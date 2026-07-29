# Copyright 2026 Mehmet Turan Yardimci
# Portions Copyright 2025 The RLinf Authors.
# Portions Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Adapted from RLinf (https://github.com/RLinf/RLinf), licensed under the Apache License, Version 2.0,
# specifically rlinf/models/embodiment/openpi/openpi_action_model.py, which the mathematics here follows closely
# and in places line for line. The integration grid and the deterministic Euler step come from LeRobot
# (https://github.com/huggingface/lerobot), also Apache 2.0, specifically the sampling loop in
# lerobot/policies/smolvla/modeling_smolvla.py, whose constant step expression this reproduces.
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b)): this file has been modified. The sampler takes the
# velocity field as a caller supplied callback rather than reading it from a specific model class, so it can drive
# any flow matching policy whose denoising step can be evaluated. Only one likelihood formulation is implemented.
# Relative to RLinf, the grid and the deterministic step are the policy library's constant step form rather than a
# difference between two entries of a linear space; the two are equal in exact arithmetic and not in floating
# point, which is why using the library's form is what keeps the sampler in step with the policy.
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Stochastic sampling for flow matching policies.

A flow matching policy produces an action chunk by integrating a learned velocity field from noise down to a
clean sample. Deterministic integration gives no action likelihood, so there is nothing for a policy gradient to
push on. This module adds a stochastic branch: at one step of the integration the update is treated as a Gaussian
whose mean is the deterministic step and whose scale is set by a noise parameter, which makes the sampled chunk a
draw from a distribution with a tractable log probability.

Two properties are worth stating because they are easy to lose in a reimplementation.

The deterministic branch uses the constant step size directly rather than the difference of two grid entries,
because the constant is the expression the policy library itself uses. The difference form varies in the last
places where the constant does not, which is small, and the reason to avoid it is not the size of that
difference but that matching the library's own arithmetic is the only way to know the sampler is on the policy's
path rather than near it.

The action head itself is never modified. The caller passes a callback that drives the policy's own denoising
step, so this file adds sampling behaviour around the head without touching its parameters.
"""

import random

import torch


def get_timesteps(num_steps: int, device) -> torch.Tensor:
    """Integration grid from one down to zero, with the trailing endpoint appended.

    The entries are the policy's own affine form, one plus i times the constant step, which is how LeRobot builds
    the same grid. Matching it is the point: the sampler drives the pretrained head, so it must walk the same
    points the head was trained to walk.

    A linear space would do here too, and saying otherwise would overstate it. Measured in float32 the two
    constructions are bitwise equal at the step count this repository documents, and differ by one unit in the
    last place at some other counts. The hazard this module actually guards against is one level down, in the
    step rather than the grid: taking the step as the difference between consecutive grid entries gives a value
    that varies in the last places, where the policy uses the constant. That is the invariant the module
    docstring states and the paired test checks.
    """
    dt = -1.0 / num_steps
    ts = torch.tensor([1.0 + i * dt for i in range(num_steps)], dtype=torch.float32, device=device)
    return torch.cat([ts, torch.tensor([0.0], dtype=torch.float32, device=device)])


def get_logprob_norm(sample, mu, sigma) -> torch.Tensor:
    """Element wise Gaussian log density, with zero scale contributing nothing.

    Deterministic steps have zero scale. Rather than special casing them at the call site, they are given a zero
    contribution here, so the same expression covers both branches.
    """
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    const = -torch.log(sigma_safe) - 0.5 * torch.log(2 * torch.pi * torch.ones_like(sample))
    expo = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
    lp = const + expo
    return torch.where(mask, torch.zeros_like(lp), lp)


def sample_mean_var(x_t, v_t, idx, timesteps, noise_method, noise_level):
    """Mean and scale of the next integration state.

    `idx` is an integer during sampling and a per element index tensor when a stored trajectory is re evaluated, so
    both are accepted. The deterministic branch returns a zero scale, which makes the caller's noise term vanish
    without a separate code path.
    """
    bsize = x_t.shape[0]
    device = x_t.device
    if isinstance(idx, int):
        idx = torch.tensor(idx, device=device).expand(bsize)
    t_input = timesteps[idx]
    delta = timesteps[idx] - timesteps[idx + 1]
    delta = delta[:, None, None].expand_as(x_t)
    t_input = t_input[:, None, None].expand_as(x_t)
    x0_pred = x_t - v_t * t_input
    x1_pred = x_t + v_t * (1 - t_input)

    if noise_method == "flow_ode":
        # Constant step size, not the grid difference, because the constant is what the policy library uses.
        # See the module docstring for why matching it matters more than the size of the difference.
        num_steps = timesteps.shape[0] - 1
        dt = -1.0 / num_steps
        return x_t + dt * v_t, torch.zeros_like(t_input)

    if noise_method == "flow_sde":
        # The noise scale grows with the ratio of remaining time to elapsed time, which diverges at the start of
        # the integration. The denominator is clamped to the first interior grid point so the first step stays
        # finite instead of producing a non finite scale.
        denom = torch.where(timesteps == 1, timesteps[1], timesteps)
        sigma_ratio = timesteps / (1 - denom)
        sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]
        sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
        x0_w = torch.ones_like(t_input) - (t_input - delta)
        x1_w = t_input - delta - sigma_i ** 2 * delta / (2 * t_input)
        x_t_std = torch.sqrt(delta) * sigma_i
        return x0_pred * x0_w + x1_pred * x1_w, x_t_std

    raise ValueError(f"unsupported noise_method: {noise_method}")


@torch.no_grad()
def sde_sample(velocity_fn, noise, num_steps, noise_method, noise_level,
               action_chunk, action_env_dim, mode="train", value_fn=None):
    """Integrate from noise to an action chunk, returning the trajectory and its log probability.

    In training mode a single integration step is stochastic and the rest are deterministic, which keeps the
    sampler close to the policy's own behaviour while still yielding a differentiable likelihood term. Evaluation
    mode makes every step deterministic, so evaluation is reproducible given the same starting noise.

    One surface, not several. A file offering more than one likelihood formulation is an invitation to compute a
    ratio across two of them, which is the failure this repository documents most insistently, so the surface the
    module describes is the only one it implements.

    The full state trajectory is returned because the update needs to recompute log probabilities under the
    current parameters at exactly the states that were visited.
    """
    device = noise.device
    bsize = noise.shape[0]
    timesteps = get_timesteps(num_steps, device)

    if mode == "train":
        di = random.randint(0, max(num_steps - 1, 0))
        denoise_inds = torch.tensor([di] * num_steps)
    else:
        denoise_inds = torch.tensor([-1] * num_steps)
    denoise_inds = denoise_inds[None].repeat(bsize, 1)

    x_t = noise
    chains, log_probs, values = [x_t], [], []

    for idx in range(num_steps):
        method = noise_method if idx == int(denoise_inds[0, idx]) else "flow_ode"
        t_scalar = timesteps[idx].expand(bsize)
        v_t, suffix_out = velocity_fn(x_t, t_scalar)
        x_t_mean, x_t_std = sample_mean_var(x_t, v_t, idx, timesteps, method, noise_level)
        x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
        log_probs.append(get_logprob_norm(x_t, x_t_mean, x_t_std))
        chains.append(x_t)
        values.append(value_fn(suffix_out) if (value_fn and suffix_out is not None)
                      else torch.zeros(bsize, device=device))

    chains = torch.stack(chains, dim=1)
    # Only the executed part of the chunk carries a likelihood the environment ever saw, so the log probability
    # is restricted to it here rather than at the loss, where the padding would silently inflate the sum.
    log_probs = torch.stack(log_probs, dim=1)[:, :, :action_chunk, :action_env_dim]
    log_probs = log_probs[torch.arange(bsize), denoise_inds[:, 0]]
    values = torch.stack(values, dim=1).mean(dim=1, keepdim=True)
    return {"actions": x_t, "chains": chains, "prev_logprobs": log_probs,
            "prev_values": values, "denoise_inds": denoise_inds}


def recompute_logprob_value(velocity_fn, chains, denoise_inds, num_steps,
                            noise_method, noise_level, compute_values=True, value_fn=None):
    """Re evaluate stored trajectories under the current parameters, with gradients.

    The policy gradient ratio compares the stored log probability against this one. Both must be computed on the
    same surface, over the same states and the same slice of the chunk, or the ratio measures the difference
    between two formulations instead of the difference between two policies.

    Only the step that was stochastic when the trajectory was collected is re evaluated, because it is the only
    one that carries a likelihood at all. Its index travels with the stored trajectory rather than being inferred
    here, so a change to how the sampler chooses it cannot silently put the two surfaces out of step.

    No entropy is returned. The single stochastic step has an entropy that is a function of the noise scale alone
    and not of any trainable parameter, so an entropy bonus built on it would be a term with no gradient: a knob
    that appears to do something and cannot.
    """
    device = chains.device
    bsize = chains.shape[0]
    bidx = torch.arange(bsize)
    timesteps = get_timesteps(num_steps, device)

    di = denoise_inds[:, 0]
    x_pre = chains[bidx, di]
    x_next = chains[bidx, di + 1]
    t_scalar = timesteps[di]
    v_t, suffix_out = velocity_fn(x_pre, t_scalar)
    x_mean, x_std = sample_mean_var(x_pre, v_t, di, timesteps, noise_method, noise_level)

    log_probs = torch.stack([get_logprob_norm(x_next, x_mean, x_std)], dim=1)
    if compute_values and value_fn is not None and suffix_out is not None:
        values = torch.stack([value_fn(suffix_out)], dim=1)
    else:
        values = torch.zeros(bsize, 1, device=device)
    return log_probs, values
