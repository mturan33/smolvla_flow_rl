#!/usr/bin/env python3
# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Score one named checkpoint, and refuse to print a rate unless the evaluation finished.

This exists so that scoring a checkpoint is something a person can run, rather than something that only happens
inside a training loop. A library function that only the trainer calls cannot be pointed at a checkpoint someone
else produced, which is most of what scoring is for.

Three things are deliberate.

The checkpoint is named by file, not by directory, so the weights that get scored are the ones asked for rather
than whichever file happens to be newest. The loader reports how many tensors changed, and refuses to continue if
that number is zero, because a load that matched nothing leaves the policy at its initialisation while every line
of output still carries the checkpoint's name.

The regime is an argument. The trainable surface determines the shape of the state dictionary, so a checkpoint
trained under one regime will not load into a model built under another. Passing the wrong one fails loudly here
instead of producing a near miss.

No rate is printed until the completeness gate passes. Once it does, the same evaluation is read at every
requested budget, which is what the recorded success step is for: asking how the policy would have scored under a
shorter horizon costs nothing and needs no second run.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch

# Set before the first repository import; it only affects imports that follow it. See the note in
# src/train_flow_rl.py: a run must not leave compiled copies that make the tree fail its own artefact gate.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from action_model import SmolVLAConfig, SmolVLAForRLActionPrediction  # noqa: E402
from checkpoint import load_policy_only  # noqa: E402
from completeness_gate import EvalReading, gate  # noqa: E402
from env_libero import build_env, get_task_prompts  # noqa: E402
from env_processors import make_processors  # noqa: E402
from evaluate import evaluate  # noqa: E402
from model_setup import build_model  # noqa: E402
from train_flow_rl import (ADAPTER_ALPHA_DEFAULT, ADAPTER_RANK_DEFAULT,  # noqa: E402
                           check_regime_args, check_task_args, parse_tasks)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", required=True,
                   help="comma separated suite:index pairs, one simulator process is started per pair")
    p.add_argument("--base_policy", required=True, help="path to the pretrained policy directory")
    p.add_argument("--checkpoint", required=True, help="path to one checkpoint file, not a directory")
    p.add_argument("--output_dir", required=True, help="where the per episode records are written")

    p.add_argument("--episodes", type=int, required=True, help="episodes per task")
    p.add_argument("--horizon", type=int, required=True, help="environment steps per episode")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--denoise_steps", type=int, required=True, help="integration steps of the flow policy")
    p.add_argument("--action_steps", type=int, default=1,
                   help="actions executed from one plan, must match how the run was trained")

    p.add_argument("--regime", required=True, choices=("frozen", "lora", "full"),
                   help="the regime the checkpoint was trained under, since it sets the state dictionary shape")
    # The same defaults the trainer uses, taken from it rather than respelled, so the two entry points cannot
    # drift apart and leave a checkpoint that loads under one and not the other.
    p.add_argument("--adapter_rank", type=int, default=ADAPTER_RANK_DEFAULT)
    p.add_argument("--adapter_alpha", type=int, default=ADAPTER_ALPHA_DEFAULT)
    p.add_argument("--critic_layernorm", action="store_true")
    p.add_argument("--popart", action="store_true")

    p.add_argument("--budgets", default="",
                   help="comma separated environment step budgets to read the same evaluation at, in addition to "
                        "the horizon")
    p.add_argument("--events_name", default="events_eval.jsonl",
                   help="file the per episode records are appended to, inside the output directory")
    return p.parse_args(argv)


def parse_budgets(spec: str, horizon: int):
    """Read the budgets to score at, refusing any that the episodes cannot answer.

    A budget above the horizon cannot be read from episodes that were never run that long. What comes back is the
    horizon's own number under a label that is not true of it, which reads as a result at that length and is
    guaranteed to understate the policy there. The trainer refuses the same shape of argument for the same reason.
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            raise SystemExit("--budgets takes whole numbers of environment steps separated by commas; %r is not "
                             "one of those." % part)
        # Bounded at both ends for one reason. Nothing can be scored at a budget of zero or fewer either, and the
        # rate printed for one would be a well formed zero under a label no episode could have earned.
        if value < 1:
            raise SystemExit(
                "--budgets %d is not a length any episode could be scored at, so the rate under it would be zero "
                "by construction. Use budgets of at least 1." % value)
        out.append(value)
    over = sorted(b for b in out if b > int(horizon))
    if over:
        raise SystemExit(
            "--budgets %s exceeds --horizon %d, and no episode ran that long, so a rate read at that budget would "
            "restate the horizon's own number under a label no episode earned. Lower the budgets, or evaluate at a "
            "longer horizon." % (", ".join(str(b) for b in over), int(horizon)))
    out.append(int(horizon))
    return sorted(set(out))


def check_matches_the_run(args, produced_under: dict) -> None:
    """Refuse arguments that disagree with the run the checkpoint came from.

    The regime already fails loudly, because the trainable surface sets the shape of the state dictionary and a
    mismatch cannot load. The rest are silent: a chunk execution length or a step count that differs from the one
    trained scores a policy the run never produced, and every line of the output still names the checkpoint. The
    trainer records what it used, so the comparison is available rather than left to the reader's memory.

    A checkpoint written before this was recorded carries nothing, and is evaluated with a line saying so rather
    than refused, since refusing would make older checkpoints unreadable for no gain in truth.
    """
    if not produced_under:
        print("  [checkpoint] records nothing about the run that produced it, so the arguments below are "
              "unchecked", flush=True)
        return
    disagree = []
    for name in ("regime", "denoise_steps", "action_steps"):
        # Only what this command actually takes is compared. The trainer records more than that, and comparing a
        # field this command has no argument for would read its absence as a disagreement.
        if name not in produced_under or not hasattr(args, name):
            continue
        want, got = produced_under[name], getattr(args, name)
        if want != got:
            disagree.append("--%s is %r here and was %r in the run" % (name, got, want))
    if disagree:
        raise SystemExit(
            "the checkpoint was produced under different arguments, so this would score a policy that run never "
            "produced: %s. Match them, or evaluate a checkpoint from the run these arguments describe."
            % "; ".join(disagree))
    print("  [checkpoint] arguments match the run that produced it", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if os.path.isdir(args.checkpoint):
        raise SystemExit("--checkpoint must be a file, not a directory: %s" % args.checkpoint)
    if not os.path.isfile(args.checkpoint):
        raise SystemExit("--checkpoint names no file: %s" % args.checkpoint)
    # Zero episodes measures nothing and, because the expectation is the product of tasks and episodes, leaves
    # the completeness gate with nothing to require. Records from an earlier run at the same checkpoint step are
    # then read as this run's, and a rate is printed under a name that did not earn it.
    if args.episodes < 1:
        raise SystemExit(
            "--episodes %d runs no episode, so nothing would be measured and any rate printed would come from "
            "records this run did not write. Use at least 1." % args.episodes)
    # The same bound the budgets carry. Without it the horizon walks past that check, since it is appended to the
    # budget list after the test, and a rate at a horizon of zero is a well formed zero under a length nothing ran.
    if args.horizon < 1:
        raise SystemExit(
            "--horizon %d is not a length any episode could run for, so every rate under it would be zero by "
            "construction. Use at least 1." % args.horizon)
    check_regime_args(args)
    # Read here rather than where they are used, which is after every episode has run. An argument that will be
    # refused should be refused before the work, not after it, and the reading is what refuses it.
    budgets = parse_budgets(args.budgets, args.horizon)
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  [device] %s" % ("cuda" if device.type == "cuda" else
                             "cpu, no CUDA device was visible; this will be very slow"), flush=True)
    tasks = parse_tasks(args.tasks)
    check_task_args(tasks)

    cfg = SmolVLAConfig(
        model_path=args.base_policy,
        num_steps=args.denoise_steps,
        noise_method="flow_ode",
        noise_level=0.0,
        device=str(device),
        add_value_head=True,
        value_head_layernorm=args.critic_layernorm,
        value_head_popart=args.popart,
    )
    model = build_model(SmolVLAForRLActionPrediction, cfg, device, args.regime,
                        args.adapter_rank, args.adapter_alpha)

    step, changed, produced_under = load_policy_only(args.checkpoint, model)
    print("  [checkpoint] %s at step %d, %d tensors changed on load"
          % (os.path.basename(args.checkpoint), step, changed), flush=True)
    check_matches_the_run(args, produced_under)

    ve, _task_of, _init_obs, suite_limit = build_env(tasks, args.horizon, args.seed)
    env_pre, env_post = make_processors(tasks[0][0], model.lerobot_cfg)
    prompts = get_task_prompts(ve)
    print("  [env] %d workers, horizon %d, suite limit %s"
          % (ve.num_envs, args.horizon, suite_limit if suite_limit else "unknown"), flush=True)
    if suite_limit and args.horizon < suite_limit:
        print("  [env] scoring at %d, below the suite's own limit of %d, so tasks needing more steps count as "
              "failures here" % (args.horizon, suite_limit), flush=True)

    # The records of this evaluation are labelled with the checkpoint's own step, so that appending a second
    # checkpoint's episodes to the same file leaves two readable groups rather than one silently merged one.
    events_path = os.path.join(args.output_dir, args.events_name)
    try:
        evaluate(model, ve, env_pre, env_post, prompts, tasks,
                 episodes_per_task=args.episodes, seed=args.seed, horizon=args.horizon,
                 device=device, events_path=events_path, n_action_steps=args.action_steps,
                 cycle=step)
    finally:
        ve.close()

    reading = EvalReading(events_path, list(range(len(tasks))), args.episodes, cycle=step)
    if not gate({os.path.basename(args.checkpoint): reading}):
        return 1

    print("success rate by budget")
    print("-" * 60)
    for budget in budgets:
        rate = reading.success_rate(budget=budget)
        print("  budget %5d   pooled %.4f" % (budget, rate))
    print()
    print("EVALUATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
