# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Collect a fixed length segment of experience, storing everything the update will need.

The defining choice here is that the sampling trajectory is kept, not just the action. A flow matching policy
reaches its action through a sequence of intermediate states, and the update has to re evaluate the likelihood at
exactly those states under the current parameters. Storing only the action and redrawing at update time pairs a
fresh sample with an advantage computed for a different one, which makes the policy gradient zero mean.

That costs memory: the whole trajectory for every environment and every step. It is kept on the host rather than
the accelerator, since it is written once and read once per epoch, and on a single consumer card the space it
would otherwise occupy is the difference between fitting and not.

The reward is sparse and terminal. There is no shaping term and no place to add one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    from .env_libero import adapt_observation, terminal_success
except ImportError:
    from env_libero import adapt_observation, terminal_success


def exec_action(model, sampled: dict, env_post, step: int = 0) -> np.ndarray:
    """Turn one planned chunk into the single action the environment executes at chunk position `step`.

    The policy's own output transform runs on the whole chunk before the position is selected. That order is
    deliberate: the transform is elementwise, so the result is the same either way, and running it on the chunk
    keeps this function agnostic to how many positions the caller intends to execute.
    """
    transformed = model.output_transform({"actions": sampled["actions"]})
    chunk = transformed["actions"]
    selected = chunk[:, step, :]
    out = env_post({"action": selected})["action"]
    return out.detach().to("cpu").numpy().astype(np.float32)


@torch.no_grad()
def collect_rollout(model, ve, env_pre, env_post, task_prompts, n_steps, *,
                    initial_obs, device, mode: str = "train", n_action_steps: int = 1) -> dict:
    """Step every environment in lockstep for a fixed number of decisions.

    A decision may execute several actions from one plan. Rewards and terminations are folded across those
    actions so the stored segment stays one row per decision: a success anywhere inside the chunk counts once,
    and a termination anywhere inside it marks the boundary. Without that folding the stored arrays would not
    line up with the likelihoods, which are per decision.

    A bootstrap value for the state after the final step is computed and returned. The segment usually ends while
    episodes are still running, and treating that as a terminal state tells the critic every segment boundary is
    worth nothing.
    """
    n_envs = ve.num_envs
    env_obs_list, prev_values = [], []
    chains_list, dinds_list, plogp_list = [], [], []
    rewards = np.zeros((n_envs, n_steps), dtype=np.float32)
    dones = np.zeros((n_envs, n_steps), dtype=np.float32)

    raw_obs = initial_obs
    use_cuda = torch.device(device).type == "cuda"

    for t in range(n_steps):
        env_obs = adapt_observation(raw_obs, env_pre, task_prompts)
        env_obs_list.append(env_obs)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
            sampled = model.sample_actions(env_obs, mode=mode)

        chains_list.append(sampled["chains"].detach().float().cpu())
        dinds_list.append(sampled["denoise_inds"].detach().cpu())
        plogp_list.append(sampled["prev_logprobs"].detach().float().cpu())

        v = sampled["prev_values"].detach()
        if v.ndim == 2 and v.shape[-1] == 1:
            v = v.squeeze(-1)
        elif v.ndim == 3 and v.shape[1] == 1:
            v = v[:, 0]
        prev_values.append(v.float().cpu())

        reward_acc = np.zeros(n_envs, dtype=np.float32)
        done_acc = np.zeros(n_envs, dtype=np.float32)
        for k in range(max(1, int(n_action_steps))):
            raw_obs, _r, term, trunc, info = ve.step(exec_action(model, sampled, env_post, step=k))
            term_a = np.asarray(term).reshape(-1).astype(bool)
            trunc_a = np.asarray(trunc).reshape(-1).astype(bool)
            reward_acc = np.maximum(reward_acc, terminal_success(term_a, info))
            done_acc = np.maximum(done_acc, (term_a | trunc_a).astype(np.float32))
        rewards[:, t] = reward_acc
        dones[:, t] = done_acc

    bootstrap_obs = adapt_observation(raw_obs, env_pre, task_prompts)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
        boot = model.sample_actions(bootstrap_obs, mode="eval")
    vb = boot["prev_values"].detach()
    if vb.ndim == 2 and vb.shape[-1] == 1:
        vb = vb.squeeze(-1)
    elif vb.ndim == 3 and vb.shape[1] == 1:
        vb = vb[:, 0]

    return {
        "env_obs": env_obs_list,
        "chains": torch.stack(chains_list, dim=1),
        "denoise_inds": torch.stack(dinds_list, dim=1),
        "prev_logprobs": torch.stack(plogp_list, dim=1),
        "values": torch.stack(prev_values, dim=1),
        "bootstrap": vb.float().cpu(),
        "rewards": torch.from_numpy(rewards),
        "dones": torch.from_numpy(dones),
        "last_raw_obs": raw_obs,
        "n_envs": n_envs,
        "n_steps": n_steps,
    }
