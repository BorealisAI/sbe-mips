# Copyright (c) 2026-present, Royal Bank of Canada.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from numpy.polynomial.hermite_e import hermegauss as roots_hermitenorm
from torch import nn


SCHEMA_VERSION = 1
FAMILIES = {"affine", "bounded_affine", "bounded_mlp"}
DEFAULTS = {
    "family": "bounded_mlp",
    "bound": 5.0,
    "context_family": "mlp",
    "hidden_dims": [128, 64],
    "context_hidden_dims": [128, 64],
    "latent_hidden_dims": [128, 64],
    "lr": 0.001,
    "epochs": 40,
    "patience": 6,
    "batch_size": 512,
    "weight_decay": 0.001,
    "lambda_normalization": 0.0,
    "lambda_second_moment": 0.0,
    "gradient_clip": 1.0,
    "min_lr_ratio": 0.1,
    "min_delta": 0.0,
    "train_samples": 16,
    "early_samples": 64,
    "inference_samples": 256,
    "train_quadrature": 32,
    "early_quadrature": 64,
    "inference_quadrature": 128,
    "inference_batch_size": 128,
}


def _config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    supplied = copy.deepcopy(config)
    for alias in ("cap", "logit_bound"):
        if alias in supplied:
            supplied.setdefault("bound", supplied.pop(alias))
    if "head_family" in supplied:
        supplied.setdefault("family", supplied.pop("head_family"))
    cfg.update(supplied)
    if cfg["family"] not in FAMILIES:
        raise ValueError(f"Unknown classifier-odds family: {cfg['family']}")
    if cfg["context_family"] not in ("linear", "mlp"):
        raise ValueError("context_family must be linear or mlp")
    for key in ("bound", "lr", "gradient_clip"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("weight_decay", "lambda_normalization", "lambda_second_moment", "min_delta"):
        if not math.isfinite(float(cfg[key])) or float(cfg[key]) < 0:
            raise ValueError(f"{key} must be nonnegative and finite")
    if not 0 <= float(cfg["min_lr_ratio"]) <= 1:
        raise ValueError("min_lr_ratio must be between zero and one")
    for key in ("epochs", "patience", "batch_size", "train_samples", "early_samples",
                "inference_samples", "train_quadrature", "early_quadrature",
                "inference_quadrature", "inference_batch_size"):
        if int(cfg[key]) != cfg[key] or int(cfg[key]) < (0 if key == "epochs" else 1):
            raise ValueError(f"Invalid integer setting: {key}")
        cfg[key] = int(cfg[key])
    for key in ("train_samples", "early_samples", "inference_samples"):
        if cfg[key] < 2 or cfg[key] % 2:
            raise ValueError(f"{key} must be positive and even for antithetic draws")
    for key in ("hidden_dims", "context_hidden_dims", "latent_hidden_dims"):
        cfg[key] = list(map(int, cfg[key]))
        if any(width <= 0 for width in cfg[key]):
            raise ValueError(f"{key} widths must be positive")
    return cfg


def _posts(posts: dict[str, Any], both: bool = True) -> dict[str, np.ndarray]:
    allowed = {"x", "mu_mean", "mu_logvar", "pi_mean", "pi_logvar"}
    if set(posts).difference(allowed):
        raise ValueError("Posterior interfaces accept only context, Gaussian means and log variances")
    keys = ["x", "mu_mean", "mu_logvar"]
    if both:
        keys += ["pi_mean", "pi_logvar"]
    missing = set(keys).difference(posts)
    if missing:
        raise ValueError(f"Missing posterior arrays: {sorted(missing)}")
    values = {key: np.asarray(posts[key], dtype=np.float32) for key in keys}
    n = len(values["x"])
    if not n or any(array.ndim != 2 or len(array) != n for array in values.values()):
        raise ValueError("Posteriors must be nonempty two-dimensional paired arrays")
    z_shape = values["mu_mean"].shape
    if not z_shape[1] or not values["x"].shape[1]:
        raise ValueError("Context and latent dimensions must be positive")
    if any(values[key].shape != z_shape for key in keys if key != "x"):
        raise ValueError("Posterior mean and log-variance shapes must agree")
    if any(not np.isfinite(array).all() for array in values.values()):
        raise FloatingPointError("Nonfinite posterior or context input")
    return values


def fit_transform(posts: dict[str, Any]) -> dict[str, np.ndarray]:
    """Fit the existing full-context / pooled-Gaussian affine coordinates."""
    values = _posts(posts)
    mean = .5 * (values["mu_mean"].mean(0) + values["pi_mean"].mean(0))
    second = .5 * sum((np.square(values[side + "_mean"])
                      + np.exp(values[side + "_logvar"])).mean(0)
                     for side in ("mu", "pi"))
    transform = {
        "x_mean": values["x"].mean(0),
        "x_scale": np.maximum(values["x"].std(0), 1e-6),
        "z_mean": mean,
        "z_scale": np.maximum(np.sqrt(np.maximum(second - mean**2, 0)), 1e-4),
    }
    if any(not np.isfinite(value).all() for value in transform.values()):
        raise FloatingPointError("Nonfinite fit-only coordinate transform")
    return {key: np.asarray(value, dtype=np.float32) for key, value in transform.items()}


def transform_posts(posts: dict[str, Any], transform: dict[str, Any],
                    both: bool = True) -> dict[str, np.ndarray]:
    values = _posts(posts, both=both)
    if values["x"].shape[1] != len(transform["x_mean"]):
        raise ValueError("Context dimension does not match checkpoint")
    if values["mu_mean"].shape[1] != len(transform["z_mean"]):
        raise ValueError("Latent dimension does not match checkpoint")
    result = {"x": ((values["x"] - transform["x_mean"])
                    / transform["x_scale"]).astype(np.float32)}
    for side in (("mu", "pi") if both else ("mu",)):
        result[side + "_mean"] = ((values[side + "_mean"] - transform["z_mean"])
                                  / transform["z_scale"]).astype(np.float32)
        result[side + "_std"] = (np.exp(.5 * values[side + "_logvar"])
                                 / transform["z_scale"]).astype(np.float32)
        if (result[side + "_std"] <= 0).any():
            raise FloatingPointError("Nonpositive transformed posterior standard deviation")
    if any(not np.isfinite(array).all() for array in result.values()):
        raise FloatingPointError("Nonfinite transformed posterior")
    return result


def _mlp(input_dim: int, hidden: list[int], output_dim: int,
         dtype: torch.dtype) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers.extend([nn.Linear(input_dim, width, dtype=dtype), nn.SiLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim, dtype=dtype))
    return nn.Sequential(*layers)


class OddsHead(nn.Module):
    def __init__(self, context_dim: int, z_dim: int, config: dict[str, Any]):
        super().__init__()
        self.config = _config(config)
        self.family = self.config["family"]
        self.bound = float(self.config["bound"])
        self.dtype = torch.float32 if self.family == "bounded_mlp" else torch.float64
        self.z_dim = int(z_dim)
        if self.family == "bounded_mlp":
            widths = self.config["context_hidden_dims"]
            context_output = widths[-1] if widths else context_dim
            self.context = (_mlp(context_dim, widths[:-1], context_output, self.dtype)
                            if widths else nn.Identity())
            self.latent_network = _mlp(context_output + z_dim,
                                       self.config["latent_hidden_dims"], 1, self.dtype)
            last = self.latent_network[-1]
        else:
            hidden = (self.config["hidden_dims"]
                      if self.config["context_family"] == "mlp" else [])
            self.context = _mlp(context_dim, hidden, z_dim + 1, self.dtype)
            last = self.context[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def coefficients(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.family == "bounded_mlp":
            raise ValueError("Nonlinear head has no affine coefficients")
        outputs = self.context(x)
        return outputs[:, 0], outputs[:, 1:]

    def cap_logit(self, raw: torch.Tensor) -> torch.Tensor:
        return raw if self.family == "affine" else self.bound * torch.tanh(raw / self.bound)

    def logits(self, x: torch.Tensor, z: torch.Tensor,
               context: torch.Tensor | None = None) -> torch.Tensor:
        """z has shape [samples, rows, latent]; compute context only once."""
        context = self.context(x) if context is None else context
        if self.family == "bounded_mlp":
            expanded = context.unsqueeze(0).expand(z.shape[0], -1, -1)
            raw = self.latent_network(torch.cat([expanded, z], dim=-1)).squeeze(-1)
        else:
            raw = context[:, 0].unsqueeze(0) + (z * context[:, 1:].unsqueeze(0)).sum(-1)
        return self.cap_logit(raw)


def gaussian_quadrature(count: int, device: torch.device | str,
                        dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    if int(count) != count or count < 2:
        raise ValueError("Gaussian quadrature requires at least two nodes")
    nodes, weights = roots_hermitenorm(int(count))
    weights = weights / math.sqrt(2 * math.pi)
    if not np.isfinite(nodes).all() or not np.isfinite(weights).all():
        raise FloatingPointError("Invalid Gaussian quadrature rule")
    return (torch.as_tensor(nodes, device=device, dtype=dtype),
            torch.as_tensor(weights, device=device, dtype=dtype))


def sobol_normal(count: int, dimension: int, seed: int,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if int(count) != count or count < 2 or count % 2:
        raise ValueError("Antithetic integration requires a positive even sample count")
    engine = torch.quasirandom.SobolEngine(dimension, scramble=True, seed=int(seed))
    u = engine.draw(int(count) // 2, dtype=torch.float64)
    eps = math.sqrt(2) * torch.erfinv(2 * u.clamp(1e-12, 1 - 1e-12) - 1)
    eps = torch.stack([eps, -eps], dim=1).flatten(0, 1)
    return eps.to(device=device, dtype=dtype)


def _finite(value: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Nonfinite {label}")


def _clip_gradients(model: nn.Module, max_norm: float) -> None:
    try:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    except RuntimeError as error:
        message = str(error).lower()
        if "total norm" in message and "non-finite" in message:
            raise FloatingPointError("Nonfinite classifier gradient norm") from error
        raise


def _affine_moments(center: torch.Tensor, variance: torch.Tensor) -> dict[str, torch.Tensor]:
    log_first = center + .5 * variance
    log_second = 2 * center + 2 * variance
    first, second = log_first.exp(), log_second.exp()
    centered = torch.expm1(log_first).square() + second * (-torch.expm1(-variance))
    for name, value in (("first moment", first), ("second moment", second),
                        ("centered second moment", centered)):
        _finite(value, name)
    if bool((first <= 0).any()):
        raise FloatingPointError("Underflowing affine importance-weight moment")
    return {"weights": first, "second_moment": second,
            "centered_second_moment": centered, "log_first": log_first,
            "log_second": log_second, "logit_variance": variance}


def _batch_statistics(model: OddsHead, batch: dict[str, torch.Tensor],
                      quadrature: tuple[torch.Tensor, torch.Tensor] | None = None,
                      noise: torch.Tensor | None = None,
                      both: bool = True) -> dict[str, torch.Tensor]:
    context = model.context(batch["x"])
    sides = ("mu", "pi") if both else ("mu",)
    result: dict[str, torch.Tensor] = {}
    bce_sides = []
    for side in sides:
        mean, std = batch[side + "_mean"], batch[side + "_std"]
        if model.family != "bounded_mlp":
            c, b = context[:, 0], context[:, 1:]
            center = c + (b * mean).sum(-1)
            slope_std = b * std
            variance = slope_std.square().sum(-1)
            if model.family == "affine" and not both:
                result.update(_affine_moments(center, variance))
                continue
            if quadrature is None:
                raise ValueError("Nonanalytic affine-family statistics require quadrature")
            nodes, probabilities = quadrature
            logit_std = torch.linalg.vector_norm(slope_std, dim=-1)
            logits = model.cap_logit(center.unsqueeze(0) + nodes[:, None] * logit_std[None])
            if side == "mu":
                if model.family == "affine":
                    result.update(_affine_moments(center, variance))
                else:
                    exp_logits = logits.exp()
                    result.update({
                        "weights": (probabilities[:, None] * exp_logits).sum(0),
                        "second_moment": (probabilities[:, None] * exp_logits.square()).sum(0),
                        "centered_second_moment": (probabilities[:, None]
                                                   * torch.expm1(logits).square()).sum(0),
                    })
            if both:
                losses = nn.functional.softplus(logits if side == "mu" else -logits)
                bce_sides.append((probabilities[:, None] * losses).sum(0))
        else:
            if noise is None:
                raise ValueError("Nonlinear heads require a normal integration bank")
            draw = noise[:, None] if noise.ndim == 2 else noise
            z = mean[None] + std[None] * draw
            logits = model.logits(batch["x"], z, context=context)
            if side == "mu":
                logits64 = logits.to(torch.float64)
                result.update({
                    "weights": logits64.exp().mean(0),
                    "second_moment": (2 * logits64).exp().mean(0),
                    "centered_second_moment": torch.expm1(logits64).square().mean(0),
                })
            if both:
                bce_sides.append(nn.functional.softplus(
                    logits if side == "mu" else -logits).to(torch.float64).mean(0))
    if both:
        result["bce_rows"] = .5 * (bce_sides[0] + bce_sides[1])
    for key, value in result.items():
        _finite(value, key)
    return result


def _objective(stats: dict[str, torch.Tensor], cfg: dict[str, Any]) -> torch.Tensor:
    loss = stats["bce_rows"].mean()
    if cfg["lambda_normalization"]:
        loss = loss + cfg["lambda_normalization"] * (stats["weights"].mean() - 1).square()
    if cfg["lambda_second_moment"]:
        loss = loss + cfg["lambda_second_moment"] * stats["centered_second_moment"].mean()
    _finite(loss, "classifier objective")
    return loss


def _tensors(values: dict[str, np.ndarray], dtype: torch.dtype,
             device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=dtype, device=device)
            for key, value in values.items()}


def _subset(values: dict[str, torch.Tensor], indices: Any) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in values.items()}


def _summarize_arrays(values: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, float]:
    weights = values["weights"]
    bce = float(values["bce_rows"].mean(dtype=np.float64))
    normalization = float((weights.mean(dtype=np.float64) - 1)**2)
    centered = float(values["centered_second_moment"].mean(dtype=np.float64))
    result = {
        "objective": bce + cfg["lambda_normalization"] * normalization
                     + cfg["lambda_second_moment"] * centered,
        "bce": bce,
        "normalization": normalization,
        "centered_second_moment": centered,
        "mean_weight": float(weights.mean(dtype=np.float64)),
        "max_weight": float(weights.max()),
        "ess_fraction": float(weights.sum(dtype=np.float64)**2
                              / (len(weights) * np.square(weights).sum(dtype=np.float64))),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite aggregate head diagnostics")
    return result


def _evaluate(model: OddsHead, values: dict[str, torch.Tensor], cfg: dict[str, Any],
              quadrature: tuple[torch.Tensor, torch.Tensor] | None,
              noise: torch.Tensor | None, both: bool = True) -> dict[str, np.ndarray]:
    model.eval()
    pieces: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(values["x"]), cfg["inference_batch_size"]):
            stats = _batch_statistics(model, _subset(values, slice(start, start + cfg["inference_batch_size"])),
                                      quadrature=quadrature, noise=noise, both=both)
            for key, value in stats.items():
                pieces.setdefault(key, []).append(value.detach().cpu().numpy())
    return {key: np.concatenate(value).astype(np.float64) for key, value in pieces.items()}


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def predict_weights(checkpoint: dict[str, Any], posts: dict[str, Any],
                    device: str = "cpu", integration_samples: int | None = None) -> dict[str, Any]:
    torch.set_num_threads(2)
    if checkpoint.get("schema_version") != SCHEMA_VERSION or checkpoint.get("estimator") != "classifier_odds":
        raise ValueError("Expected a classifier-odds checkpoint")
    if checkpoint.get("class_prior_target") != .5:
        raise ValueError("These checkpoints require balanced classifier priors")
    cfg = _config(checkpoint["config"])
    device_obj = torch.device(device)
    with torch.random.fork_rng(devices=[]):
        model = OddsHead(checkpoint["context_dim"], checkpoint["z_dim"], cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device_obj).eval()
    values = _tensors(transform_posts(posts, checkpoint["transform"], both=False), model.dtype, device_obj)
    rule = noise = None
    if model.family == "bounded_mlp":
        count = cfg["inference_samples"] if integration_samples is None else integration_samples
        noise = sobol_normal(count, model.z_dim, checkpoint["integration_seeds"]["inference"],
                             device_obj, model.dtype)
        method = "sobol_antithetic"
    elif model.family == "bounded_affine":
        count = cfg["inference_quadrature"] if integration_samples is None else integration_samples
        rule = gaussian_quadrature(count, device_obj, model.dtype)
        method = "one_dimensional_gauss_hermite"
    else:
        count = 0
        method = "analytic_gaussian"
    result = _evaluate(model, values, cfg, rule, noise, both=False)
    result["integration"] = {"method": method, "samples": int(count),
                             "seed": checkpoint["integration_seeds"]["inference"]
                             if model.family == "bounded_mlp" else None}
    return result
