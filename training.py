# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from . import models, odds
else:
    import models
    import odds


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def finite_loss(loss):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')


def moments(values):
    mean = values.mean(0, dtype=np.float64).astype(np.float32)
    std = values.std(0, dtype=np.float64).astype(np.float32)
    return mean, np.where(std > 1e-6, std, 1).astype(np.float32)


def fit_dm(train, spec, seed, device):
    cfg = spec['config']
    seed_everything(seed)
    mean, scale = moments(train['pair_mu'])
    offset = float(train['r_mu'].mean(dtype=np.float64))
    reward_scale = max(float(train['r_mu'].std(dtype=np.float64)), 1e-6)
    x = torch.from_numpy((train['pair_mu'] - mean) / scale)
    y = torch.from_numpy(((train['r_mu'] - offset) / reward_scale).astype(np.float32))
    model = models.RewardMLP(x.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['lr']*cfg['min_lr_ratio'])
    shuffle = torch.Generator().manual_seed(seed + 193)
    for epoch in range(spec['epochs']):
        model.train()
        for indices in torch.randperm(len(x), generator=shuffle).split(cfg['batch_size']):
            loss = (model(x[indices].to(device)) - y[indices].to(device)).square().mean()
            finite_loss(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
    return dict(config=cfg, pair_dim=x.shape[1], model_state_dict=state(model),
                preprocessing=dict(pair_mean=torch.from_numpy(mean),pair_scale=torch.from_numpy(scale),
                                   reward_offset=offset,reward_scale=reward_scale))


def fit_reward_mips(train, monitor_rows, spec, seed, device):
    cfg = spec['config']
    dims = lambda key: tuple(map(int, cfg[key].split(',')))
    seed_everything(seed)
    pair_dim, context_dim = train['pair_mu'].shape[1], train['x'].shape[1]
    encoder = models.RewardSupervisedEncoder(pair_dim,cfg['mrs_z_dim'],dims('mrs_encoder_hidden_dims'),cfg['dropout']).to(device)
    reward = models.RewardRegressor(context_dim+cfg['mrs_z_dim'],dims('reward_hidden_dims'),cfg['dropout']).to(device)
    ratio = models.RatioClassifier(context_dim+cfg['mrs_z_dim'],dims('ratio_hidden_dims'),cfg['dropout']).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(reward.parameters())+list(ratio.parameters()),
                                 lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    if cfg['reward_target_transform'] != 'standardize':
        raise ValueError('Released reward-MIPS recipe uses standardized behavior rewards')
    offset, scale = float(train['r_mu'].mean()),max(float(train['r_mu'].std()),1e-8)
    y = ((train['r_mu']-offset)/scale).astype(np.float32)
    loader = DataLoader(TensorDataset(*(torch.from_numpy(v) for v in
        [train['pair_mu'],train['pair_pi'],train['x'],train['x'],y])),
        batch_size=cfg['batch_size'],shuffle=True)

    monitor = DataLoader(TensorDataset(torch.arange(monitor_rows)),batch_size=cfg['batch_size'],shuffle=False)
    for epoch in range(1,spec['epochs']+1):
        warmup = epoch <= cfg['mrs_warmup_epochs']
        encoder.train(); reward.train(); ratio.train(not warmup)
        for values in loader:
            mu,pi,xmu,xpi,rewards = [v.to(device) for v in values]
            optimizer.zero_grad(set_to_none=True)
            zmu=encoder(mu)
            reward_loss=nn.functional.smooth_l1_loss(reward(torch.cat([zmu,xmu],dim=1)),rewards)
            if warmup:
                with torch.no_grad(): zpi=encoder(pi)
                ratio_zmu=zmu.detach()
            else:
                zpi=encoder(pi); ratio_zmu=zmu
            lm=ratio(torch.cat([ratio_zmu,xmu],dim=1))
            lp=ratio(torch.cat([zpi,xpi],dim=1))
            logits=torch.cat([lm,lp]); labels=torch.cat([torch.zeros_like(lm),torch.ones_like(lp)])
            bce=nn.functional.binary_cross_entropy_with_logits(logits,labels)
            norm=(lm.clamp(-20,20).exp().mean()-1).square()
            loss=cfg['mrs_lambda_reward']*reward_loss
            if not warmup: loss=loss+cfg['mrs_lambda_cls']*bce+cfg['mrs_lambda_ratio_norm']*norm
            finite_loss(loss); loss.backward(); optimizer.step()
        for _ in monitor: pass
    return dict(representation_config=dict(pair_input_dim=pair_dim,x_input_dim=context_dim,z_dim=cfg['mrs_z_dim'],
        encoder_hidden_dims=list(dims('mrs_encoder_hidden_dims')),reward_hidden_dims=list(dims('reward_hidden_dims')),
        ratio_hidden_dims=list(dims('ratio_hidden_dims')),dropout=cfg['dropout'],ratio_input_dim=context_dim+cfg['mrs_z_dim'],ratio_input_mode='z_x'),
        rho=.5,encoder_state_dict=state(encoder),reward_state_dict=state(reward),ratio_state_dict=state(ratio),
        reward_target_transform='standardize',reward_transform_offset=offset,reward_transform_scale=scale)


def fit_mips(train, monitor_rows, spec, seed, device):
    """Frozen vanilla-MIPS classifier, with the original DataLoader RNG order."""
    cfg = spec['config']
    hidden = cfg['ratio_hidden_dims']
    hidden = tuple(map(int, hidden.split(','))) if isinstance(hidden, str) else tuple(hidden)
    seed_everything(seed)
    model = models.RatioClassifier(train['pair_mu'].shape[1], hidden, cfg['dropout']).to(device)
    x = torch.from_numpy(np.concatenate([train['pair_mu'], train['pair_pi']]))
    y = torch.cat([torch.zeros(len(train['x'])), torch.ones(len(train['x']))])
    loader = DataLoader(TensorDataset(x, y), batch_size=cfg['batch_size'], shuffle=True)
    monitor = DataLoader(TensorDataset(torch.arange(2 * monitor_rows)),
                         batch_size=cfg['batch_size'], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    for _ in range(spec['epochs']):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).reshape(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
            if (y < .5).any():
                norm = (logits[y < .5].clamp(-10, 10).exp().mean() - 1).square()
                loss = loss + float(cfg.get('lambda_ratio_norm', 0.) or 0.) * norm
            finite_loss(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        for _ in monitor:
            pass
    return dict(estimator='mips', input_dim=train['pair_mu'].shape[1],
                hidden_dims=list(hidden), dropout=cfg['dropout'], rho=.5,
                lambda_ratio_norm=float(cfg.get('lambda_ratio_norm', 0.) or 0.),
                state_dict=state(model))


def fit_preprocessing(train,cfg):
    pair_mean,pair_scale=moments(train['pair_mu']); xm,xs=moments(train['x'])
    offset=float(train['r_mu'].mean(dtype=np.float64)); scale=max(float(train['r_mu'].std(dtype=np.float64)),1e-6)
    coef=np.zeros(train['x'].shape[1],np.float32); intercept=0.
    if cfg['reward_mode']=='additive_context':
        x=((train['x']-xm)/xs).astype(np.float64); y=(train['r_mu'].astype(np.float64)-offset)/scale
        xc,yc=x-x.mean(0),y-y.mean()
        if xc.shape[1]<=len(xc):
            c=np.linalg.solve(xc.T@xc+cfg['context_ridge_alpha']*np.eye(xc.shape[1]),xc.T@yc)
        else:
            c=xc.T@np.linalg.solve(xc@xc.T+cfg['context_ridge_alpha']*np.eye(len(xc)),yc)
        coef=c.astype(np.float32); intercept=float(y.mean()-x.mean(0)@c)
    return dict(pair_mean=pair_mean,pair_scale=pair_scale,x_mean=xm,x_scale=xs,reward_offset=offset,reward_scale=scale,
                context_ridge_coef=coef,context_ridge_intercept=intercept)


def prepare(train,p):
    result={side:((train['pair_'+side]-p['pair_mean'])/p['pair_scale']).astype(np.float32) for side in ('mu','pi')}
    result['baseline']=(((train['x']-p['x_mean'])/p['x_scale'])@p['context_ridge_coef']+p['context_ridge_intercept']).astype(np.float32)
    result['reward']=((train['r_mu']-p['reward_offset'])/p['reward_scale']).astype(np.float32)
    return result


def model_optimizer(model,lr,weight_decay,multiplier):
    return torch.optim.AdamW([dict(params=list(model.encoder.parameters()),lr=lr*multiplier),
                             dict(params=list(model.decoder.parameters()),lr=lr)],lr=lr,weight_decay=weight_decay)


def kl(mean,logvar):
    return .5*(mean.square()+logvar.exp()-1-logvar).sum(-1).mean()


def fit_representation(train,spec,seed,device):
    cfg=spec['representation']; seed_everything(seed)
    preprocessing=fit_preprocessing(train,cfg); values=prepare(train,preprocessing)
    model=models.RepresentationModel(train['pair_mu'].shape[1],cfg).to(device)
    optimizer=model_optimizer(model,cfg['lr'],cfg['weight_decay'],cfg['encoder_lr_multiplier'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch:
        cfg['min_lr_ratio']+(1-cfg['min_lr_ratio'])*(1+math.cos(math.pi*epoch/cfg['epochs']))/2)
    shuffle=torch.Generator().manual_seed(110000+seed)
    for epoch in range(1,spec['pretrain_epochs']+1):
        model.train(); order=torch.randperm(len(train['x']),generator=shuffle).numpy()
        coefficient=cfg['kl_weight']*min(epoch/cfg['kl_warmup_epochs'],1.) if cfg['kl_warmup_epochs'] else cfg['kl_weight']
        for start in range(0,len(order),cfg['batch_size']):
            indices=order[start:start+cfg['batch_size']]
            mean,logvar=model.encoder(torch.from_numpy(values['mu'][indices]).to(device))
            noise=torch.randn(cfg['train_mc_samples'],*mean.shape,device=device,dtype=mean.dtype)
            z=mean[None]+(.5*logvar).exp()[None]*noise
            baseline=torch.from_numpy(values['baseline'][indices]).to(device)
            rewards=torch.from_numpy(values['reward'][indices]).to(device)
            if cfg.get('antithetic_pretrain', False):
                if cfg['train_mc_samples'] != 2:
                    raise ValueError('Antithetic pretraining requires two samples')
                z=mean[None]+(.5*logvar).exp()[None]*torch.stack([noise[0],-noise[0]])
            reward_loss=.5*(model.decode(z,baseline[None])-rewards[None]).square().mean()
            kl_value=kl(mean,logvar)
            if cfg.get('kl_side','behavior') == 'both':
                pi_mean,pi_logvar=model.encoder(torch.from_numpy(values['pi'][indices]).to(device))
                kl_value=.5*(kl_value+kl(pi_mean,pi_logvar))
            loss=reward_loss+coefficient*kl_value
            finite_loss(loss); optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True); optimizer.step()
        scheduler.step()
    cp=dict(config=cfg,pair_dim=train['pair_mu'].shape[1],x_dim=train['x'].shape[1],preprocessing=preprocessing,model_state_dict=state(model))
    return cp


def posteriors(checkpoint,features,device='cpu',batch_size=512):
    with torch.random.fork_rng(devices=[]):
        model=models.RepresentationModel(checkpoint['pair_dim'],checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict']); model.to(device).eval()
    p=checkpoint['preprocessing']; result={'x':features['x']}
    with torch.no_grad():
        for side in ('mu','pi'):
            means=[]; logvars=[]
            for start in range(0,len(features['x']),batch_size):
                x=((features['pair_'+side][start:start+batch_size]-p['pair_mean'])/p['pair_scale']).astype(np.float32)
                mean,logvar=model.encoder(torch.from_numpy(x).to(device))
                means.append(mean.cpu().numpy());logvars.append(logvar.cpu().numpy())
            result[side+'_mean']=np.concatenate(means);result[side+'_logvar']=np.concatenate(logvars)
    return result


def context_projection(x,dimensions):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);standard=(x-mean)/scale
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202)
        _,_,v=torch.pca_lowrank(torch.from_numpy(standard),q=min(dimensions+8,min(standard.shape)),center=False,niter=2)
    components=v[:,:dimensions].numpy().copy(); projected_scale=np.maximum((standard@components).std(0),1e-4)
    return dict(mean=mean,scale=scale,components=components,projected_scale=projected_scale)


def latent_transform(posts,projection):
    mean=.5*(posts['mu_mean'].mean(0)+posts['pi_mean'].mean(0))
    second=.5*sum((np.square(posts[side+'_mean'])+np.exp(posts[side+'_logvar'])).mean(0) for side in ('mu','pi'))
    scale=np.maximum(np.sqrt(np.maximum(second-mean**2,0)),1e-4)
    return dict(context=projection,z_mean=mean.astype(np.float32),z_scale=scale.astype(np.float32))


class JointRatioHead(nn.Module):
    def __init__(self,context_dim,z_dim):
        super().__init__(); dim=context_dim+z_dim;i,j=torch.tril_indices(dim,dim)
        self.register_buffer('i',i);self.register_buffer('j',j);self.register_buffer('center',(i==j).float())
        self.network=nn.Linear(dim+len(i),1);nn.init.zeros_(self.network.weight);nn.init.zeros_(self.network.bias)

    def forward(self,x,z):
        values=torch.cat([x[None].expand(z.shape[0],-1,-1),z],-1)
        return self.network(torch.cat([values,values[...,self.i]*values[...,self.j]-self.center],-1)).squeeze(-1)


def gradient_norm(grads):
    return torch.stack([value.square().sum() for value in grads]).sum().sqrt()


def fit_joint(train,checkpoint,spec,seed,device):
    cfg=spec['representation'];jcfg=spec['joint'];rcfg=spec['joint_ratio']
    initial_posts=posteriors(checkpoint,train,device,cfg['batch_size'])
    transform=latent_transform(initial_posts,context_projection(train['x'],8))
    seed_everything(seed)
    with torch.random.fork_rng(devices=[]): model=models.RepresentationModel(checkpoint['pair_dim'],cfg)
    model.load_state_dict(checkpoint['model_state_dict']);model.to(device)
    head=JointRatioHead(8,cfg['z_dim']).to(device)
    encoder_params=list(model.encoder.parameters());decoder_params=list(model.decoder.parameters());head_params=list(head.parameters())
    optimizer=model_optimizer(model,jcfg['encoder_decoder_lr'],cfg['weight_decay'],jcfg['encoder_lr_multiplier'])
    head_optimizer=torch.optim.AdamW(head_params,lr=jcfg['ratio_lr'],weight_decay=rcfg['weight_decay'])
    values=prepare(train,checkpoint['preprocessing']);c=transform['context']
    tx=(((train['x']-c['mean'])/c['scale']@c['components'])/c['projected_scale']).astype(np.float32)
    zmean,zscale=[torch.as_tensor(transform[key],device=device) for key in ('z_mean','z_scale')]
    shuffle=torch.Generator().manual_seed(seed+773000)
    for epoch in range(spec['joint_epochs']):
        model.train();head.train();order=torch.randperm(len(tx),generator=shuffle).numpy()
        for start in range(0,len(order),cfg['batch_size']):
            indices=order[start:start+cfg['batch_size']];x=torch.from_numpy(tx[indices]).to(device)
            mu=model.encoder(torch.from_numpy(values['mu'][indices]).to(device));pi=model.encoder(torch.from_numpy(values['pi'][indices]).to(device))
            draws=[]
            for mean,logvar in (mu,pi):
                noise=torch.randn(jcfg['train_mc_samples'],*mean.shape,device=device,dtype=mean.dtype)
                draws.append((mean[None]+(.5*logvar).exp()[None]*noise,mean[None]+(.5*logvar.detach()).exp()[None]*noise))
            baseline=torch.from_numpy(values['baseline'][indices]).to(device);rewards=torch.from_numpy(values['reward'][indices]).to(device)
            reward_loss=.5*(model.decode(draws[0][0],baseline[None])-rewards[None]).square().mean()
            kl_value=kl(*mu)
            if cfg.get('kl_side','behavior') == 'both':
                kl_value=.5*(kl_value+kl(*pi))
            weighted_kl=cfg['kl_weight']*kl_value
            lm,lp=head(x,(draws[0][1]-zmean)/zscale),head(x,(draws[1][1]-zmean)/zscale)
            ratio_loss=.5*(nn.functional.softplus(lm).mean()+nn.functional.softplus(-lp).mean())+rcfg['lambda_normalization']*(lm.clamp(rcfg['logit_min'],rcfg['logit_max']).exp().mean()-1).square()
            finite_loss(reward_loss+weighted_kl+ratio_loss)
            rg=torch.autograd.grad(reward_loss,encoder_params+decoder_params,retain_graph=True)
            kg=torch.autograd.grad(weighted_kl,encoder_params,retain_graph=True)
            raw=torch.autograd.grad(ratio_loss,encoder_params+head_params,allow_unused=True)
            bg=[torch.zeros_like(p) if g is None else g for p,g in zip(encoder_params+head_params,raw)]
            n=len(encoder_params);nr,nb=gradient_norm(rg[:n]),gradient_norm(bg[:n])
            normalizer=torch.where(nb>jcfg['gradient_epsilon'],(nr/nb.clamp_min(jcfg['gradient_epsilon'])).clamp(max=jcfg['normalizer_cap']),torch.zeros_like(nb))
            combined=[r+k+jcfg['ratio_gradient_fraction']*normalizer*b for r,k,b in zip(rg[:n],kg,bg[:n])]
            optimizer.zero_grad(set_to_none=True);head_optimizer.zero_grad(set_to_none=True)
            for p,g in zip(encoder_params+decoder_params,combined+list(rg[n:])):p.grad=g.detach()
            for p,g in zip(head_params,bg[n:]):p.grad=g.detach()
            nn.utils.clip_grad_norm_(model.parameters(),jcfg['encoder_decoder_clip'],error_if_nonfinite=True)
            nn.utils.clip_grad_norm_(head.parameters(),jcfg['ratio_clip'],error_if_nonfinite=True)
            optimizer.step();head_optimizer.step()
    checkpoint=copy.deepcopy(checkpoint);checkpoint['model_state_dict']=state(model)
    return checkpoint


def fit_classifier(posts, spec, seed, device):
    cfg = odds._config(spec['head'])
    transform = odds.fit_transform(posts)
    values = odds.transform_posts(posts, transform)
    context_dim, z_dim = values['x'].shape[1], values['mu_mean'].shape[1]
    device_obj = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = odds.OddsHead(context_dim, z_dim, cfg)
    model.to(device_obj)
    values = odds._tensors(values, model.dtype, device_obj)
    rule = (None if model.family == 'bounded_mlp'
            else odds.gaussian_quadrature(cfg['train_quadrature'], device_obj, model.dtype))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(spec.get('cosine_horizon', cfg['epochs'])), 1),
        eta_min=cfg['lr'] * cfg['min_lr_ratio'])
    shuffle = torch.Generator().manual_seed(int(seed) + 740001)
    noise_rng = torch.Generator().manual_seed(int(seed) + 820001)
    for _ in range(spec['head_epochs']):
        model.train()
        order = torch.randperm(len(values['x']), generator=shuffle)
        for indices in order.split(cfg['batch_size']):
            batch = odds._subset(values, indices.to(device_obj))
            noise = None
            if model.family == 'bounded_mlp':
                half = torch.randn(cfg['train_samples'] // 2, len(indices), z_dim,
                                   generator=noise_rng, dtype=model.dtype)
                noise = torch.stack([half, -half], dim=1).flatten(0, 1).to(device_obj)
            stats = odds._batch_statistics(model, batch, quadrature=rule, noise=noise)
            loss = odds._objective(stats, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            odds._clip_gradients(model, cfg['gradient_clip'])
            optimizer.step()
        scheduler.step()
    return dict(schema_version=1, estimator='classifier_odds', config=cfg,
                context_dim=context_dim, z_dim=z_dim, transform=transform,
                state_dict=state(model), seed=int(seed),
                integration_seeds=dict(early=int(seed) + 940001, inference=int(seed) + 940002),
                selected_epoch=spec['head_epochs'], fit_rows=len(posts['x']),
                class_prior_target=.5, training_duration='fixed_from_completed_study')


def _training_arrays(arrays):
    if set(arrays) != {'x', 'pair_mu', 'pair_pi', 'r_mu'}:
        raise ValueError('Training accepts exactly x, pair_mu, pair_pi and behavior rewards r_mu')
    result = {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}
    if result['x'].ndim != 2 or not len(result['x']):
        raise ValueError('Training requires a nonempty context matrix')
    n = len(result['x'])
    if any(value.ndim != (1 if key == 'r_mu' else 2) or len(value) != n
           or not np.isfinite(value).all() for key, value in result.items()):
        raise ValueError('Training arrays must be finite, nonempty and aligned')
    if result['pair_mu'].shape != result['pair_pi'].shape:
        raise ValueError('Behavior and target feature dimensions differ')
    return result


def validate_config(config):
    if not isinstance(config, dict) or config.get('schema_version') != 5:
        raise ValueError('Expected a head-only training configuration with schema_version=5')
    sbe = config.get('sbe')
    if not isinstance(sbe, dict):
        raise ValueError('Missing SBE training configuration')
    removed = {'branch_kind', 'head_kind', 'temperature', 'repair', 'repair_seed',
               'repair_epochs', 'integration'} & sbe.keys()
    if removed:
        raise ValueError('Removed SBE branch configuration: ' + ', '.join(sorted(removed)))
    if sbe.get('representation_stage') not in ('pretrain', 'joint'):
        raise ValueError('SBE representation_stage must be pretrain or joint')
    if sbe.get('representation', {}).get('private_variance', False):
        raise ValueError('Separate repair variance branches are not supported')


def fit_all(train, early, config, device='cpu', epoch_cap=None):
    validate_config(config)
    torch.set_num_threads(2)
    train, early = _training_arrays(train), _training_arrays(early)
    if any(train[key].shape[1:] != early[key].shape[1:] for key in train):
        raise ValueError('Train and early feature dimensions differ')
    cfg = copy.deepcopy(config)
    seed, sbe = cfg['initialization_seed'], cfg['sbe']
    if type(seed) is not int or seed < 0:
        raise ValueError('Initialization seed must be a nonnegative integer')
    if epoch_cap is not None:
        if type(epoch_cap) is not int or epoch_cap < 1:
            raise ValueError('Epoch cap must be a positive integer')
        for method in ('dm', 'reward_mips', 'mips'):
            if method in cfg:
                cfg[method]['epochs'] = min(cfg[method]['epochs'], epoch_cap)
        for key in ('pretrain_epochs', 'joint_epochs', 'head_epochs'):
            if key in sbe:
                sbe[key] = min(sbe[key], epoch_cap)
    result = dict(schema_version=5, method='sbe_mips', task_id=cfg.get('task_id', 'unregistered'))
    for name, trainer in [('dm', fit_dm), ('reward_mips', fit_reward_mips), ('mips', fit_mips)]:
        if name not in cfg:
            continue
        args = (train, cfg[name], seed, device) if name == 'dm' else (train, len(early['x']), cfg[name], seed, device)
        result['mips_with_reward_supervision' if name == 'reward_mips' else name] = trainer(*args)
        print(name + ' complete', flush=True)
    representation = fit_representation(train, sbe, seed, device)
    print('SBE pretraining complete', flush=True)
    if sbe['representation_stage'] == 'joint':
        representation = fit_joint(train, representation, sbe, seed + 1000, device)
        print('SBE joint stage complete', flush=True)
    posts = posteriors(representation, train)
    variance_scale = float(sbe.get('posterior_variance_scale', 1.))
    if not math.isfinite(variance_scale) or variance_scale <= 0:
        raise ValueError('Posterior variance scale must be positive and finite')
    if variance_scale != 1.:
        posts = {key: value.copy() for key, value in posts.items()}
        for side in ('mu', 'pi'):
            posts[side + '_logvar'] += np.float32(math.log(variance_scale))
    head = fit_classifier(posts, sbe, int(sbe.get('head_seed', seed + 23000)), device)
    result['sbe'] = dict(representation=representation, head=head,
                         posterior_variance_multiplier=variance_scale)
    result['training'] = dict(initialization_seed=seed, fixed_configuration=True,
                              target_rewards_used=False, epoch_cap=epoch_cap,
                              scheduler_horizons_preserved=True)
    print('SBE classifier complete', flush=True)
    return result
