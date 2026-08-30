# Copyright 2026 Mehmet Turan Yardimci
# Portions Copyright 2025 The RLinf Authors.
#
# Adapted from RLinf (https://github.com/RLinf/RLinf), licensed under the Apache License, Version 2.0,
# specifically the likelihood and value recomputation in
# rlinf/models/embodiment/openpi/openpi_action_model.py, which this file follows closely.
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b)): the update grades the stored trajectory that was
# actually executed instead of a freshly drawn one, the likelihood surface is restricted to the executed part of
# the action chunk, adaptive target rescaling was added for the critic, and a non finite update guard was added.
# The reference anchor is written so that descending the penalty raises the likelihood of the executed action
# under the current policy, it is off by default, and it is refused outright when the policy carries no adapters
# to switch off, since there is then no pretrained policy left to anchor to. No entropy term is computed.
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Policy gradient update for a flow matching policy on stored trajectories.

Three details in this file exist because getting them wrong produces a run that looks healthy and learns nothing.

The update grades the action the environment executed. Collection stores the sampling trajectory, the sampling
log probability and the index of the stochastic step; the update re evaluates exactly those states. If instead a fresh
chunk were drawn at update time and paired with the stored advantage, the two would be independent, the policy
gradient term would have zero mean, and the run would perform a random walk. The visible signature of that
mistake is a likelihood ratio pinned at one with no spread, which is why the ratio and its spread are returned
from every update rather than only on failure. The trainer records them once per update cycle, taking the last
update of the cycle, so what reaches the metrics file is a sample of them rather than all of them.

The likelihood is summed over the executed surface only. A chunked policy predicts many future steps and pads the
action dimension to a fixed width, but the environment sees only the executed steps and the real degrees of
freedom. Summing over everything inflates the spread of the log ratio by roughly the square root of the ratio of
widths, which pushes most samples outside the clip range and throttles the gradient. The same slice is applied
to the new, stored and reference likelihoods, so the first update still starts at a ratio of one.

The anchor pulls toward the reference policy. The penalty is written so that descending it raises the likelihood
of the executed action under the current policy. Written with the opposite sign it becomes a force away from the
reference, which is easy to miss while the gradient is throttled and destructive once it is not.
"""

from __future__ import annotations

import torch

from reference_ops import pg_loss_reference, value_loss_reference
import torch.nn.functional as F

try:
    from .flow_sde import recompute_logprob_value
    from .lora_utils import count_adapter_layers, disable_adapters
except ImportError:
    from flow_sde import recompute_logprob_value
    from lora_utils import count_adapter_layers, disable_adapters


def _recompute_current(model, env_obs, chains, denoise_inds, noise_level):
    """Re evaluate stored trajectories under the current parameters, with gradients."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        images, img_masks, lang_tokens, lang_masks, state, _ = model.prep_inputs(env_obs)
        prefix, cache, _ = model.build_prefix_cache(images, img_masks, lang_tokens, lang_masks, state)
        velocity_fn = model.make_velocity_fn(prefix, cache)
        value_fn = model.make_value_fn()
        return recompute_logprob_value(
            velocity_fn=velocity_fn, chains=chains, denoise_inds=denoise_inds,
            num_steps=model.lerobot_cfg.num_steps,
            noise_method=model.config.noise_method,
            noise_level=noise_level,
            compute_values=True, value_fn=value_fn,
        )


def reference_is_available(model) -> bool:
    """Whether disabling adapters actually recovers the pretrained policy.

    It does only when EVERY trained parameter lives in an adapter. If any part of the policy itself was opened
    at full rank, its pretrained weights have been overwritten in place, and disabling the adapters leaves
    those changes behind: what comes back is the current policy with its adapters off, not the pretrained one.

    Counting adapters is therefore not enough, and this used to count them and stop. A regime that carries
    adapters on the language backbone WHILE training the action expert and the action head at full rank passes
    an adapter count and fails the actual requirement, so the anchor would have run against a reference that
    silently included the trained decoder. The condition is now read from the property itself: no trainable
    parameter outside an adapter.
    """
    if count_adapter_layers(model.flow_model) == 0:
        return False
    for name, p in model.flow_model.named_parameters():
        if p.requires_grad and "lora" not in name.lower():
            return False
    return True


def _recompute_reference(model, env_obs, chains, denoise_inds, noise_level):
    """Re evaluate the same trajectories under the pretrained policy, adapters disabled, without gradients.

    The whole policy is walked, not one subtree of it. Adapters sit in different places depending on what was
    opened for training, so disabling them under a fixed subtree recovers the pretrained policy in one case and
    silently returns the current one in the others. A reference that equals the current policy makes the anchor a
    structural zero, and the logged value that would reveal it reads zero by construction.
    """
    if not reference_is_available(model):
        raise RuntimeError(
            "the reference anchor needs adapters to disable, and this policy has none; "
            "with the policy open at full rank the pretrained weights are gone and no in place reference exists")
    root = model.flow_model
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16), \
            disable_adapters(root):
        images, img_masks, lang_tokens, lang_masks, state, _ = model.prep_inputs(env_obs)
        prefix, cache, _ = model.build_prefix_cache(images, img_masks, lang_tokens, lang_masks, state)
        velocity_fn = model.make_velocity_fn(prefix, cache)
        logp, _v = recompute_logprob_value(
            velocity_fn=velocity_fn, chains=chains, denoise_inds=denoise_inds,
            num_steps=model.lerobot_cfg.num_steps,
            noise_method=model.config.noise_method,
            noise_level=noise_level,
            compute_values=False, value_fn=None,
        )
    return logp


def _update_target_scale(value_head, returns, beta):
    """Advance the critic's running target statistics and reparameterise the readout to match.

    The readout is rescaled so that its de normalised output is the same before and after the statistics move.
    Without that step, changing the statistics would shift every prediction as a side effect.
    """
    flat = returns.detach().float().reshape(-1)
    old_mu = float(value_head.pa_mu)
    old_sigma = float(value_head.pa_sigma)
    new_mu = (1 - beta) * old_mu + beta * float(flat.mean())
    new_nu = (1 - beta) * float(value_head.pa_nu) + beta * float((flat ** 2).mean())
    new_sigma = float(max((new_nu - new_mu ** 2) ** 0.5, 1e-4))
    with torch.no_grad():
        readout = value_head.mlp[-1]
        readout.weight.mul_(old_sigma / new_sigma)
        if readout.bias is not None:
            readout.bias.copy_((old_sigma * readout.bias + old_mu - new_mu) / new_sigma)
    # The statistics are buffers, so they are written in place rather than rebound. Rebinding would replace the
    # buffer with a plain float and drop it out of the state dictionary again.
    value_head.set_target_scale(new_mu, new_nu, new_sigma)


def ppo_step_stored(
    model,
    env_obs: dict,
    chains: torch.Tensor,
    denoise_inds: torch.Tensor,
    prev_logp: torch.Tensor,
    noise_level: float,
    optimizer: torch.optim.Optimizer,
    *,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    kl_coef: float = 0.0,
    grad_clip_norm: float = 1.0,
    kl_anchor: bool = False,
    n_action_steps: int = 1,
    popart_beta: float = 1e-3,
    old_values: torch.Tensor | None = None,
    accum_index: int = 0,          # micro batch position inside the global batch
    accum_total: int = 1,          # micro batches per optimizer step; 1 reproduces the old behaviour exactly
    accum_state: dict | None = None,   # carries "dirty" across micro batches; the caller owns it
) -> dict:
    """One update over a stored batch. Returns a metrics dictionary; the caller decides what to log."""
    device = next(model.parameters()).device
    chains = chains.to(device, non_blocking=True)
    denoise_inds = denoise_inds.to(device)
    prev_logp = prev_logp.to(device)

    # Statistics are advanced before the value recompute so the head de normalises with the same numbers the loss
    # will re normalise with. Doing it afterwards leaves the two a step apart.
    value_head = getattr(model, "value_head", None)
    rescaling = (value_head is not None) and getattr(value_head, "popart", False)
    if rescaling:
        _update_target_scale(value_head, returns, popart_beta)

    new_logp_full, new_vals = _recompute_current(model, env_obs, chains, denoise_inds, noise_level)

    env_dim = getattr(model, "action_env_dim")
    executed = int(n_action_steps)
    new_logp_sum = new_logp_full[:, 0][:, :executed, :env_dim].flatten(start_dim=1).sum(dim=-1)
    prev_logp_sum = prev_logp[:, :executed, :env_dim].flatten(start_dim=1).sum(dim=-1)

    log_ratio = new_logp_sum - prev_logp_sum
    ratio = log_ratio.exp()
    pg_loss = pg_loss_reference(ratio, advantages, clip_range).mean()
    clip_frac = float(((ratio - 1.0).abs() > clip_range).float().mean().item())

    values = new_vals.squeeze(-1) if (new_vals.ndim == 2 and new_vals.shape[-1] == 1) \
        else new_vals.flatten(start_dim=1).mean(dim=-1)
    # Under adaptive rescaling EVERY term is carried to the normalised scale first, the stored old value
    # included, or the trust region would be a bound in one unit applied to a quantity in another.
    if rescaling:
        mu, sigma = value_head.pa_mu, value_head.pa_sigma
        v_new = (values.float() - mu) / sigma
        v_ret = (returns.float() - mu) / sigma
        v_old = None if old_values is None else (old_values.float().to(v_new.device) - mu) / sigma
    else:
        v_new, v_ret = values.float(), returns.float()
        v_old = None if old_values is None else old_values.float().to(v_new.device)
    v_loss = value_loss_reference(v_new, v_old, v_ret).mean()

    if kl_anchor:
        ref_logp_full = _recompute_reference(model, env_obs, chains, denoise_inds, noise_level)
        ref_logp_sum = ref_logp_full[:, 0][:, :executed, :env_dim].flatten(start_dim=1).sum(dim=-1)
        kl_penalty = (ref_logp_sum.detach() - new_logp_sum).mean()
        kl_measure = (new_logp_sum.detach() - ref_logp_sum).mean()
    else:
        kl_penalty = torch.zeros((), device=new_logp_sum.device, dtype=new_logp_sum.dtype)
        kl_measure = kl_penalty.detach()

    # No entropy term. The entropy of the one stochastic step depends only on the noise scale, not on any
    # trainable parameter, so a coefficient multiplying it would be a knob that cannot move anything, which is the
    # defect this repository exists to warn about. The recompute does not calculate it either, since computing a
    # quantity in order to discard it is the same defect wearing a different hat.
    loss = pg_loss + value_coef * v_loss + kl_coef * kl_penalty

    # GRADIENT ACCUMULATION. The global batch is the whole set of micro batches and the optimiser steps once
    # per global batch, not once per micro batch.
    #
    # This is worth stating precisely because the failure mode is silent: a loop that calls zero_grad and
    # step on every micro batch trains at the MICRO batch size while every banner, manifest and log line
    # reports the pooled figure. Nothing errors, the curves look ordinary, and the run is simply not the run
    # the configuration describes. accum_total=1 reproduces the per-micro-batch behaviour exactly, so the
    # default changes nothing for callers that do not opt in.
    #
    # The loss is divided by accum_total so the accumulated gradient is the MEAN over the global batch rather
    # than its sum, which is what a single large batch would have produced.
    _first = (accum_index == 0)
    _last = (accum_index >= accum_total - 1)
    if accum_state is None:
        accum_state = {}
    if _first:
        optimizer.zero_grad(set_to_none=True)
        accum_state["dirty"] = False
    (loss / float(accum_total)).backward()
    # One non finite micro batch poisons the ACCUMULATED gradient, so it is remembered until the step
    # boundary rather than judged locally. Judging locally was safe when every micro batch was its own step;
    # it is not safe once they are pooled.
    if not bool(torch.isfinite(loss).item()):
        accum_state["dirty"] = True
    if not _last:
        return {
            "loss": float(loss.detach().float().item()),
            "pg_loss": float(pg_loss.detach().float().item()),
            "v_loss": float(v_loss.detach().float().item()),
            "ratio_mean": float(ratio.detach().float().mean().item()),
            "clip_frac": clip_frac,
            "grad_norm": float("nan"),
            "step_skipped": False,
            "accumulating": True,
            "loss_finite": bool(torch.isfinite(loss).item()),
        }
    trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    # The clip returns the norm from before it acted, which is the interesting number for reading a run but
    # cannot distinguish a step that was clipped from one that was scaled to nothing. Both are recorded.
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm).item()
    grad_norm_applied = float(torch.norm(torch.stack(
        [p.grad.detach().float().norm() for p in trainable if p.grad is not None])).item()) if any(
        p.grad is not None for p in trainable) else 0.0

    # A single non finite batch must not be applied. Stepping the optimiser with non finite gradients poisons
    # every parameter it touches, after which every later step is non finite too, so what began as one bad batch
    # presents as an unrecoverable run. The batch is dropped and reported instead.
    loss_finite = bool(torch.isfinite(loss).item())
    grad_finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
    step_skipped = not (loss_finite and grad_finite) or bool(accum_state.get("dirty", False))
    if step_skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()

    return {
        "loss": float(loss.detach().float().item()),
        "pg_loss": float(pg_loss.detach().float().item()),
        "v_loss": float(v_loss.detach().float().item()),
        "kl_to_reference": float(kl_measure.float().item()),
        "ratio_mean": float(ratio.detach().float().mean().item()),
        # Population spread, not the sample estimate. The batch here is one entry per environment, and the
        # documented first run has a single task and therefore a single environment, where the corrected form is
        # a division by zero: it returns a quiet NaN, warns once per update, and lands in the metrics file as a
        # bare NaN token that strict JSON parsers reject.
        "ratio_std": float(ratio.detach().float().std(correction=0).item()),
        "ratio_median": float(ratio.detach().float().median().item()),
        "ratio_max": float(ratio.detach().float().max().item()),
        "clip_frac": clip_frac,
        "grad_norm": float(grad_norm),
        "grad_norm_applied": grad_norm_applied,
        "prev_logp_mean": float(prev_logp_sum.detach().float().mean().item()),
        "new_logp_mean": float(new_logp_sum.detach().float().mean().item()),
        "value_mean": float(values.detach().float().mean().item()),
        "chains_finite": bool(torch.isfinite(chains).all().item()),
        "logp_finite": bool(torch.isfinite(new_logp_full).all().item()),
        "loss_finite": loss_finite,
        "step_skipped": bool(step_skipped),
        "n_trainable_with_grad": int(len(trainable)),
        # Numerical evidence that adaptive rescaling is doing something, rather than a line saying it is on. Both
        # start at one and zero, so a run where they never move is a run where the mechanism is inert, and that is
        # visible in the metrics file without anyone having to instrument it afterwards.
        "pa_sigma": float(value_head.pa_sigma) if rescaling else None,
        "pa_mu": float(value_head.pa_mu) if rescaling else None,
    }
