#!/usr/bin/env python3
# Copyright 2026 Mehmet Turan Yardimci
# Portions Copyright 2025 The RLinf Authors.
#
# Adapted from RLinf (https://github.com/RLinf/RLinf), licensed under the Apache License, Version 2.0.
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b)): the operators are re-expressed in the forms
# documented in the docstring, with equivalence to the reference forms proved by test rather than asserted.
"""Reference PPO operators. One implementation, imported by every PPO step in the tree.

These are the loss and reward operators written to match a published reference implementation rather than
re-derived, so that a run here can be compared against that recipe without a table of unexplained
differences.

    reference   RLinf, LIBERO openpi configuration
                pin 832db3f5c7aa6c61d72f20aaf43b7ce0f4ad2b03
                rlinf/algorithms/losses.py     compute_ppo_actor_loss
                rlinf/algorithms/utils.py      huber_loss
                rlinf/workers/env/env_worker.py
                examples/embodiment/config/libero_10_ppo_openpi.yaml
                    clip_ratio_low 0.2  clip_ratio_high 0.2  clip_ratio_c 3.0
                    value_clip 0.2      huber_delta 10.0

WHY ONE FILE RATHER THAN AN EDIT IN EACH STEP. More than one PPO step exists in this tree, and any
comparison between two runs is only meaningful if they share these operators exactly. Writing the same clip
twice is how two code paths drift apart, and a drift between compared runs is not a bug in one of them, it
is a confound in the comparison. So the operators live here once and are imported.

THE REFERENCE LINES, quoted so the claim can be checked without opening the clone:

    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio
    policy_loss = torch.max(policy_loss1, policy_loss2)
    if clip_ratio_c is not None:
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        policy_loss = torch.min(policy_loss, policy_loss3)

Equivalence with the form written below is proved on a grid in `tests/test_reference_parity.py`, not
asserted here.
"""
from __future__ import annotations

import torch

DUAL_CLIP_C = 3.0
VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def pg_loss_reference(ratio, advantages, clip_range, dual_clip_c=DUAL_CLIP_C):
    """Per-sample policy loss, NOT reduced. The caller takes the mean.

    The single-clip core is written as `-min(r*A, clip(r)*A)`, which is the same number as the reference's
    `max(-r*A, -clip(r)*A)`.

    DUAL CLIP. `clip_ratio_c` bounds the loss when the advantage is NEGATIVE and the ratio has run far above
    one, which is the regime where standard PPO's gradient grows without limit. It engages nowhere else: on
    a positive advantage the ordinary clip already caps the loss. In a flow-matching policy whose log-ratio
    is a sum over denoising steps, ratios well above one do occur, so this term is not decoration.
    """
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    loss = -torch.minimum(unclipped, clipped)
    if dual_clip_c is not None:
        if not dual_clip_c > 1.0:
            raise ValueError("dual_clip_c must exceed 1.0, got %r" % (dual_clip_c,))
        loss = torch.minimum(loss, torch.sign(advantages) * dual_clip_c * advantages)
    return loss


def huber(err, delta=HUBER_DELTA):
    """Quadratic within delta, linear beyond, and continuous at the join."""
    a = err.abs()
    return torch.where(a <= delta, 0.5 * err ** 2, delta * (a - 0.5 * delta))


def value_loss_reference(new_values, old_values, returns,
                         value_clip=VALUE_CLIP, huber_delta=HUBER_DELTA):
    """Per-sample value loss, NOT reduced.

    VALUE CLIP. The new value is held inside a trust region of `value_clip` around the old one and the
    PESSIMISTIC max of the two errors is taken. This only bites when the value moves TOWARD its return past
    the bound: moving away leaves the unclipped error larger, and the max picks it, which is what an
    unclipped loss would have given anyway.

    HUBER. Beyond `huber_delta` the error enters linearly, so one outlying return cannot dominate a batch.
    Without it the loss is unbounded in the error; at an error of 19.5 the plain form gives 190.125 against
    the reference's 145.0.

    `old_values` may be None, in which case the clip is skipped and only the Huber applies. That path exists
    for callers that genuinely have no stored value, NOT as a convenience: passing None silently is how a
    patch becomes a knob that does nothing, so the caller is expected to pass the buffer's own stored values.
    """
    unclipped = huber(new_values - returns, huber_delta)
    if old_values is None or value_clip is None:
        return unclipped
    clipped_v = old_values + torch.clamp(new_values - old_values, -value_clip, value_clip)
    return torch.maximum(unclipped, huber(clipped_v - returns, huber_delta))


def truncation_compensated_rewards(rewards, values, bootstrap, truncations, gamma):
    """Add gamma*V(s_next) to the reward of every step that ended by TIMEOUT rather than by success.

    reference   rlinf/workers/env/env_worker.py
                    adjusted_rewards[:, -1] += self.cfg.algorithm.gamma * final_values
                active under auto_reset True and bootstrap_type standard, both of which hold in
                libero_10_ppo_openpi.yaml.

    WHY IT MATTERS, and it is easy to get wrong in a way that looks correct. `dones = terminations OR
    truncations` is NOT itself a deviation; the reference does the same OR. The deviation is that the
    reference then pays back, on the reward side, the value that the OR masked away. Without this
    compensation a timeout is scored as if the episode had ended with nothing ahead of it, so the policy is
    pushed away from behaviour that was merely unfinished, and on a long-horizon manipulation benchmark the
    episodes that time out are exactly the long ones.

    rewards      [B, T]
    values       [B, T]   V(s_t)
    bootstrap    [B]      V(s_T), the trailing-horizon estimate
    truncations  [B, T]   1 where step t ended by timeout and NOT by termination
    """
    if truncations is None:
        return rewards
    out = rewards.clone()
    T = rewards.shape[1]
    for t in range(T):
        m = truncations[:, t].to(rewards.dtype)
        if float(m.sum()) == 0.0:
            continue
        # V(s_{t+1}): the next stored value, or the bootstrap when the timeout lands on the last step.
        v_next = values[:, t + 1] if t + 1 < T else bootstrap
        out[:, t] = out[:, t] + gamma * v_next * m
    return out
