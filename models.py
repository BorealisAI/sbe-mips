# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import numpy as np
import torch
from torch import nn

METHODS = ('dm', 'mips', 'mips_with_reward_supervision', 'sbe_mips')

if __package__:
    from . import odds
else:
    import odds


def pair_features(context, response):
    x, a = np.asarray(context, np.float32), np.asarray(response, np.float32)
    if x.shape != a.shape or x.ndim != 2:
        raise ValueError('Context and response embeddings must have equal matrix shapes')
    if not x.shape[1] or not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError('Embeddings must have positive width and finite values')
    return np.concatenate([x, a, x*a, np.abs(x-a)], axis=1).astype(np.float32)


def mlp(input_dim, hidden_dims, output_dim, dropout=0., dtype=torch.float32):
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(input_dim, width, dtype=dtype), nn.SiLU(), nn.Dropout(dropout)])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim, dtype=dtype))
    return nn.Sequential(*layers)


class RewardMLP(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.network = mlp(input_dim, config['hidden_dims'], 1, config['dropout'])

    def forward(self, features):
        return self.network(features).squeeze(-1)


class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=.1, residual=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_projection = (nn.Linear(input_dim, output_dim)
                                    if residual and input_dim != output_dim else None)
        self.use_residual = residual

    def forward(self, inputs):
        output = self.dropout(self.activation(self.norm(self.linear(inputs))))
        if not self.use_residual:
            return output
        return output + (inputs if self.residual_projection is None else self.residual_projection(inputs))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=.1, residual=False):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        self.backbone = nn.Sequential(*[FeedForwardBlock(a, b, dropout, residual)
                                      for a, b in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], output_dim)

    def forward(self, inputs):
        return self.output(self.backbone(inputs))


class RewardSupervisedEncoder(nn.Module):
    def __init__(self, input_dim, z_dim=64, hidden_dims=(256, 128), dropout=.1):
        super().__init__()
        self.network = MLP(input_dim, hidden_dims, z_dim, dropout, residual=True)
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, features):
        return self.norm(self.network(features))


class RatioClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), dropout=.1):
        super().__init__()
        self.network = MLP(input_dim, hidden_dims, 1, dropout, residual=True)

    def forward(self, features):
        return self.network(features).squeeze(-1)


class RewardRegressor(RatioClassifier):
    pass


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        if config.get('private_variance', False):
            raise ValueError('Separate repair variance branches are not supported')
        hidden = config['hidden_dims'][-1] if config['hidden_dims'] else input_dim
        self.trunk = mlp(input_dim, config['hidden_dims'], hidden, config['dropout'])
        self.mean_head = nn.Linear(hidden, config['z_dim'])
        self.logvar_head = nn.Linear(hidden, config['z_dim'])
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, config['logvar_init'])
        self.logvar_min, self.logvar_max = config['logvar_min'], config['logvar_max']

    def forward(self, features):
        hidden = self.trunk(features)
        return self.mean_head(hidden), self.logvar_head(hidden).clamp(self.logvar_min, self.logvar_max)


class RewardDecoder(nn.Sequential):
    def enable_skip(self, z_dim):
        self.skip_weight = nn.Parameter(next(self.parameters()).new_zeros((1, z_dim)))

    def forward(self, z):
        output = super().forward(z)
        return output + nn.functional.linear(z, self.skip_weight) if hasattr(self, 'skip_weight') else output


class RepresentationModel(nn.Module):
    def __init__(self, pair_dim, config):
        super().__init__()
        self.encoder = GaussianEncoder(pair_dim, config)
        self.decoder = RewardDecoder(*list(mlp(config['z_dim'], config['decoder_hidden_dims'],
                                              1, config['dropout']).children()))
        if config.get('joint_reward_skip', False):
            self.decoder.enable_skip(config['z_dim'])

    def decode(self, z, context_baseline):
        return self.decoder(z).squeeze(-1) + context_baseline


def _features(features):
    if not isinstance(features, dict) or set(features) != {'x', 'pair_mu', 'pair_pi'}:
        raise ValueError('Features must contain only x, pair_mu and pair_pi; no rewards')
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in features.items()}
    n = len(arrays['x'])
    if not n or any(value.ndim != 2 or len(value) != n for value in arrays.values()):
        raise ValueError('Expected nonempty aligned feature matrices')
    if arrays['pair_mu'].shape != arrays['pair_pi'].shape:
        raise ValueError('Behavior and target pair feature dimensions differ')
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite features')
    return arrays


def _batched(model, features, batch_size=512, device='cpu'):
    model.to(device).eval()
    with torch.no_grad():
        return np.concatenate([
            model(torch.from_numpy(np.ascontiguousarray(features[i:i+batch_size], dtype=np.float32))
                  .to(device)).detach().cpu().numpy()
            for i in range(0, len(features), batch_size)])


def predict_dm(checkpoint, features, device='cpu'):
    model = RewardMLP(checkpoint['pair_dim'], checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    p = checkpoint['preprocessing']
    x = ((features['pair_pi'] - np.asarray(p['pair_mean'])) / np.asarray(p['pair_scale'])).astype(np.float32)
    return _batched(model, x, device=device).astype(np.float64)*p['reward_scale'] + p['reward_offset']


def _ratio_from_logits(logits, prior=.5):
    if not 0 < prior < 1:
        raise ValueError('Classifier prior must lie strictly between zero and one')
    return np.exp(np.clip(np.asarray(logits, np.float64), -60., 60.)) * (1-prior)/prior


def predict_mips(checkpoint, features, device='cpu'):
    if checkpoint.get('estimator') != 'mips':
        raise ValueError('Expected a vanilla MIPS checkpoint')
    model = RatioClassifier(checkpoint['input_dim'], checkpoint['hidden_dims'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    return _ratio_from_logits(_batched(model, features['pair_mu'], device=device), checkpoint['rho'])


def predict_reward_mips(checkpoint, features, device='cpu'):
    cfg = checkpoint['representation_config']
    encoder = RewardSupervisedEncoder(cfg['pair_input_dim'], cfg['z_dim'], cfg['encoder_hidden_dims'], cfg['dropout'])
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    z = _batched(encoder, features['pair_mu'], device=device)
    ratio = RatioClassifier(cfg['ratio_input_dim'], cfg['ratio_hidden_dims'], cfg['dropout'])
    ratio.load_state_dict(checkpoint['ratio_state_dict'])
    if cfg.get('ratio_input_mode', 'z_x') != 'z_x':
        raise ValueError('Released reward-MIPS uses [z, x] inputs')
    inputs = np.concatenate([z, features['x']], axis=1)
    return _ratio_from_logits(_batched(ratio, inputs, device=device), float(checkpoint.get('rho', .5)))


def encode_posteriors(checkpoint, features, batch_size=512):
    model = RepresentationModel(checkpoint['pair_dim'], checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model_state_dict'])
    p, result = checkpoint['preprocessing'], {'x': features['x']}
    with torch.no_grad():
        for side in ('mu', 'pi'):
            values = ((features['pair_'+side] - p['pair_mean']) / p['pair_scale']).astype(np.float32)
            outputs = [model.encoder(torch.from_numpy(values[i:i+batch_size]))
                       for i in range(0, len(values), batch_size)]
            result[side+'_mean'] = np.concatenate([a.numpy() for a, _ in outputs])
            result[side+'_logvar'] = np.concatenate([b.numpy() for _, b in outputs])
    return result


def validate_checkpoint(checkpoint):
    if (not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != 5
            or checkpoint.get('method') != 'sbe_mips'):
        raise ValueError('Expected an SBE-MIPS checkpoint with schema_version=5')
    if not {'dm', 'mips_with_reward_supervision', 'sbe', 'task_id'} <= checkpoint.keys():
        raise ValueError('Missing baseline, SBE or task checkpoint information')
    sbe = checkpoint['sbe']
    required = {'representation', 'head', 'posterior_variance_multiplier'}
    if not isinstance(sbe, dict) or set(sbe) != required:
        raise ValueError('SBE checkpoint must contain only representation, head and posterior_variance_multiplier')
    multiplier = sbe['posterior_variance_multiplier']
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('posterior_variance_multiplier must be positive and finite')
    head = sbe['head']
    if not isinstance(head, dict) or head.get('estimator') != 'classifier_odds':
        raise ValueError('Expected a fitted classifier-odds head')
    representation = sbe['representation']
    if not isinstance(representation, dict) or not {'config', 'model_state_dict'} <= representation.keys():
        raise ValueError('Incomplete Gaussian representation checkpoint')
    if ('repair' in representation or representation['config'].get('private_variance', False)
            or any('variance_trunk' in key for key in representation['model_state_dict'])):
        raise ValueError('Encoder repair checkpoints are not supported by the head-only model')


def predict_values(checkpoint, features, behavior_rewards, device='cpu'):
    validate_checkpoint(checkpoint)
    features = _features(features)
    r_mu = np.asarray(behavior_rewards, np.float64)
    if r_mu.shape != (len(features['x']),) or not np.isfinite(r_mu).all():
        raise ValueError('Expected a finite aligned behavior-reward vector')
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        result = {'dm': predict_dm(checkpoint['dm'], features, device=device),
                  'mips_with_reward_supervision': predict_reward_mips(
                      checkpoint['mips_with_reward_supervision'], features, device=device)*r_mu}
        if 'mips' in checkpoint and checkpoint['mips'] is not None:
            result['mips'] = predict_mips(checkpoint['mips'], features, device=device)*r_mu
        sbe = checkpoint['sbe']
        posts = encode_posteriors(sbe['representation'], features)
        multiplier = sbe['posterior_variance_multiplier']
        if multiplier != 1.:
            for side in ('mu', 'pi'):
                posts[side+'_logvar'] = posts[side+'_logvar'] + np.float32(np.log(multiplier))
        weights = odds.predict_weights(sbe['head'], posts, device=device)['weights']
        result['sbe_mips'] = weights*r_mu
    if any(value.shape != r_mu.shape or not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite or mismatched prediction contributions')
    return {method: result[method] for method in METHODS if method in result}
