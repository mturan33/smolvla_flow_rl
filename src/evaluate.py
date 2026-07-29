# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Deterministic evaluation that writes one record per episode.

Three properties make an evaluation comparable across runs, and all three are decisions rather than accidents.

Sampling is deterministic. Every integration step is taken without noise, so two evaluations of the same weights
under the same seed produce the same episodes, and a difference between two evaluations is therefore a difference
in the weights. Evaluating with exploration noise on measures the noise as much as the policy.

Initial states are pinned by batch index rather than drawn. The same batch index gives the same starting
configuration in every run, which is what makes two evaluation runs comparable episode by episode.

The step at which an episode succeeded is recorded, not merely whether it did. That single number lets one
evaluation be read at several budgets afterwards, so asking how the policy does under a shorter horizon does not
require running anything again. It also separates two episode records that share an outcome but not a
trajectory, which the outcome alone does not.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from .env_libero import adapt_observation, set_initial_states, terminal_success
    from .rollout import exec_action
except ImportError:
    from env_libero import adapt_observation, set_initial_states, terminal_success
    from rollout import exec_action

N_INIT_STATES = 50  # how many stored starting configurations the suite provides


@torch.no_grad()
def evaluate(model, ve, env_pre, env_post, task_prompts, tasks: Sequence[Tuple[str, int]], *,
             episodes_per_task: int, seed: int, horizon: int, device,
             events_path: Optional[str] = None, n_action_steps: int = 1,
             cycle: Optional[int] = None) -> float:
    """Run a fixed number of episodes per task and return the pooled success rate.

    One episode per task runs in each batch, so the batch index and the task together identify an episode. Every
    episode is written out as it finishes, which means a run interrupted halfway leaves a file that is honestly
    short rather than one that looks complete.

    The cycle label is written into every record and identifies which set of weights produced it. A run that
    evaluates periodically appends to the same file, so without it the records of two checkpoints are
    indistinguishable, and a reader keyed on task and batch alone silently keeps whichever arrived last while
    reporting the file as complete.

    `horizon` counts ENVIRONMENT STEPS, the same unit the episode cap uses, and so does the recorded success step.
    That has to be said because the loop below is over decisions, and each decision executes several environment
    steps: measuring the budget in one unit and capping the episode in the other silently shortens every episode
    by that factor while the record still claims the full length, and any budget read off the file afterwards is
    then wrong by the same factor.
    """
    n_envs = ve.num_envs
    was_training = model.training
    prev_method = model.config.noise_method
    prev_steps = model.lerobot_cfg.num_steps
    model.eval()
    model.config.noise_method = "flow_ode"

    use_cuda = torch.device(device).type == "cuda"
    # Offset so that evaluation episodes never draw the same seeds as collection does. The value is arbitrary;
    # only its being fixed and far from zero matters.
    eval_seed_base = seed + 7000
    successes = np.zeros(len(tasks), dtype=np.int64)
    attempts = np.zeros(len(tasks), dtype=np.int64)

    steps_per_decision = max(1, int(n_action_steps))
    n_decisions = max(1, int(horizon) // steps_per_decision)
    if int(horizon) % steps_per_decision:
        print("  [eval] horizon %d is not a whole number of decisions at %d steps each; scoring %d decisions, "
              "which is %d environment steps" % (horizon, steps_per_decision, n_decisions,
                                                 n_decisions * steps_per_decision), flush=True)

    try:
        for batch in range(episodes_per_task):
            set_initial_states(ve, "eval", n_envs, per_task=1, eval_batch=batch, n_init_states=N_INIT_STATES)
            raw, _ = ve.reset(seed=[eval_seed_base + batch * n_envs + i for i in range(n_envs)])

            solved = np.zeros(n_envs, dtype=bool)
            finished = np.zeros(n_envs, dtype=bool)
            success_step = np.full(n_envs, -1, dtype=int)

            for t in range(n_decisions):
                obs = adapt_observation(raw, env_pre, task_prompts)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda):
                    sampled = model.sample_actions(obs, mode="eval")
                for k in range(steps_per_decision):
                    raw, _r, term, trunc, info = ve.step(exec_action(model, sampled, env_post, step=k))
                    term_a = np.asarray(term).reshape(-1).astype(bool)
                    trunc_a = np.asarray(trunc).reshape(-1).astype(bool)
                    hit = terminal_success(term_a, info).astype(bool)
                    first = hit & ~finished & (success_step < 0)
                    # Environment steps, not decisions, so that a budget read off this file later means what the
                    # episode cap and the horizon argument mean.
                    success_step[first] = t * steps_per_decision + k + 1
                    solved |= hit & ~finished
                    finished |= term_a | trunc_a
                if finished.all():
                    break

            for i in range(n_envs):
                successes[i] += int(solved[i])
                attempts[i] += 1
            if events_path:
                os.makedirs(os.path.dirname(events_path) or ".", exist_ok=True)
                with open(events_path, "a") as fh:
                    for i in range(n_envs):
                        fh.write(json.dumps({
                            "cycle": None if cycle is None else int(cycle),
                            "task_idx": int(i), "batch": int(batch), "env": int(i),
                            "seed": int(eval_seed_base + batch * n_envs + i),
                            "success": bool(solved[i]), "succ_step": int(success_step[i]),
                            # Both units, written down, so no reader has to work out which one a number is in.
                            "horizon": int(n_decisions * steps_per_decision),
                            "decisions": int(n_decisions),
                            "steps_per_decision": int(steps_per_decision)}) + "\n")
    finally:
        model.config.noise_method = prev_method
        model.lerobot_cfg.num_steps = prev_steps
        if was_training:
            model.train()

    rates = [successes[i] / max(1, attempts[i]) for i in range(len(tasks))]
    return float(sum(rates) / len(rates))
