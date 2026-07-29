# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Generalised advantage estimation over a batch of fixed length segments.

Written for the shape this trainer produces: several environments stepped in lockstep for a fixed number of
steps, so the batch is a rectangle of environments by timesteps rather than a set of complete episodes. Two
consequences follow, and both are handled here rather than left to the caller.

An episode can end inside the segment. The terminal flag zeroes the bootstrap across that boundary, so credit
does not leak from one episode into the one that follows it in the same slot.

The segment can also end while the episode is still running. That tail is not terminal, it is merely unobserved,
so the caller supplies a value estimate for the state after the last step and it is used as the bootstrap. The
common mistake is to treat the end of a segment as an episode end, which silently tells the critic that every
segment boundary is worth zero future reward.
"""

from __future__ import annotations

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                bootstrap: torch.Tensor, gamma: float, lam_gae: float
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and value targets for one reward stream.

    Shapes: rewards, values and dones are environments by timesteps; bootstrap is one value per environment,
    the estimate for the state that follows the final stored step. Returns advantages and value targets with the
    same shape as rewards.

    The discount and trace decay are required arguments. They are not given defaults here because a default in a
    library function is the kind of value that quietly becomes a project wide constant nobody chose.
    """
    n_env, horizon = rewards.shape
    advantages = torch.zeros_like(rewards)
    running = torch.zeros(n_env, device=rewards.device, dtype=rewards.dtype)
    next_value = bootstrap
    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_value * not_done - values[:, t]
        running = delta + gamma * lam_gae * not_done * running
        advantages[:, t] = running
        next_value = values[:, t]
    return advantages, advantages + values


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardise advantages across the whole batch.

    Applied to the pooled batch rather than per environment. Standardising each environment separately would
    remove the between environment differences that a multi task batch exists to carry.

    The population form of the standard deviation is used, not the sample form. With the default correction a
    batch holding one element has zero degrees of freedom and returns NaN, and one element is reachable: a single
    environment with a single stored timestep produces exactly that. The NaN then propagates into every loss and
    the non finite guard drops every update, which the trainer now refuses rather than continuing through, so the
    consequence is a stopped run rather than a completed one that trained on nothing. The same correction is set
    for the ratio spread in the policy gradient update, for the same reason.
    """
    return (adv - adv.mean()) / (adv.std(correction=0) + eps)
