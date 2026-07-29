#!/usr/bin/env python3
# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Train a flow matching policy with a policy gradient on a simulated manipulation suite.

The loop is deliberately plain: collect a fixed segment, estimate advantages, run a few epochs of updates over
the stored trajectories, evaluate periodically, checkpoint. Everything interesting is in the pieces it calls.

On defaults. Fourteen arguments are required. Nine describe the run: the task list, the base policy path, the
output directory, the number of update cycles, the rollout length, the episode length, the number of integration
steps, the seed, and which part of the policy trains. Five are tuning knobs: the exploration noise scale, the
discount, the trace decay and both learning rates.

None of them has a default. A default would quietly become a constant that later runs inherit without anyone
having chosen it, and an inherited constant is harder to notice than a missing argument. The trainable surface is
in that list for the same reason and not as an oversight: opening a surface other than the one intended produces a
run whose arguments look right and whose manifest is the only place the difference shows.

Everything that does carry a default is a conventional value, not a tuned one. The reference anchor weight
defaults to zero, so the plain objective is what runs unless it is asked for explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

# Set before the first repository import, because it only affects imports that come after it. Importing this
# tree otherwise leaves compiled copies beside the source, which the repository's own gate treats as a failure,
# so a run would make the tree fail its own check. Putting it here rather than only in the gate runner means it
# holds however the file is invoked.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from action_model import SmolVLAConfig, SmolVLAForRLActionPrediction  # noqa: E402
from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from env_libero import build_env, get_task_prompts, set_initial_states  # noqa: E402
from env_processors import make_processors  # noqa: E402
from gae import compute_gae, normalize_advantages  # noqa: E402
from model_setup import build_model, build_optimizer  # noqa: E402
from ppo_step import ppo_step_stored, reference_is_available  # noqa: E402
from rollout import collect_rollout  # noqa: E402
from tripwire import TRIPPED_EXIT_CODE, Tripwire  # noqa: E402
from evaluate import evaluate  # noqa: E402


# How far the first update's likelihood ratio may sit from one before the run is stopped. Wide enough to absorb
# reduced precision arithmetic on the recompute, narrow enough that a genuinely different likelihood surface
# cannot hide inside it.
RATIO_TOLERANCE = 1e-2

# Named so that a run under a regime which builds no adapters can tell a value that was passed from one that was
# defaulted, and refuse the first.
ADAPTER_RANK_DEFAULT = 8
ADAPTER_ALPHA_DEFAULT = 16


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    p.add_argument("--tasks", required=True,
                   help="comma separated suite:index pairs, one simulator process is started per pair")
    p.add_argument("--base_policy", required=True, help="path to the pretrained policy directory")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--noise_level", type=float, required=True, help="exploration noise scale")
    p.add_argument("--actor_lr", type=float, required=True, help="learning rate for the policy")
    p.add_argument("--critic_lr", type=float, required=True, help="learning rate for the critic head")
    p.add_argument("--gamma", type=float, required=True, help="discount")
    p.add_argument("--gae_lambda", type=float, required=True, help="advantage trace decay")

    # No default. This selects the trainable surface, and a default here would silently give every run that omits
    # the flag one specific configuration that nobody chose.
    p.add_argument("--regime", required=True, choices=("frozen", "lora", "full"),
                   help="which part of the policy trains")
    p.add_argument("--adapter_rank", type=int, default=ADAPTER_RANK_DEFAULT,
                   help="adapter rank, read only by the frozen and lora regimes")
    p.add_argument("--adapter_alpha", type=int, default=ADAPTER_ALPHA_DEFAULT,
                   help="adapter scaling, read only by the frozen and lora regimes")

    p.add_argument("--updates", type=int, required=True, help="number of collect and update cycles")
    p.add_argument("--rollout_steps", type=int, required=True, help="decisions per environment per cycle")
    p.add_argument("--action_steps", type=int, default=1, help="actions executed from one plan")
    p.add_argument("--epochs", type=int, default=1, help="passes over the stored segment per cycle")

    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kl_coef", type=float, default=0.0, help="reference anchor weight, off by default")
    p.add_argument("--popart", action="store_true", help="adaptive rescaling of critic targets")
    p.add_argument("--critic_layernorm", action="store_true")

    p.add_argument("--episode_length", type=int, required=True)
    p.add_argument("--denoise_steps", type=int, required=True, help="integration steps of the flow policy")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--eval_every", type=int, default=0, help="cycles between evaluations, zero disables")
    p.add_argument("--eval_episodes", type=int, default=0, help="episodes per task at each evaluation")
    p.add_argument("--eval_horizon", type=int, default=0, help="scoring budget, zero means the episode length")
    p.add_argument("--ckpt_every", type=int, default=0)
    p.add_argument("--resume", action="store_true")

    # The guard's thresholds have no defaults. A threshold that fits one setting will eventually kill a healthy
    # run in another, and a number inherited from someone else's setup is worse than no guard, because it looks
    # deliberate. Switching the guard on therefore means choosing them.
    p.add_argument("--tripwire", action="store_true", help="enable the stability guard, requires its thresholds")
    p.add_argument("--trip_ratio_median", type=float, default=None)
    p.add_argument("--trip_ratio_max", type=float, default=None)
    p.add_argument("--trip_success_floor", type=float, default=None)
    p.add_argument("--trip_window", type=int, default=None,
                   help="how many update cycles the guard watches, not optimiser steps")
    return p.parse_args(argv)


def check_horizon_args(args) -> None:
    """Refuse an evaluation budget the episodes cannot reach.

    Collection and evaluation share one vector environment, and its episode cap is the training episode length.
    An evaluation asked for a longer horizon than that cannot run it: the episodes truncate at the cap while the
    written records are labelled with the number that was asked for, and a rate read at that budget afterwards
    describes a length no episode ever had. Refused here rather than corrected silently, because raising the cap
    would change the training data as a side effect.
    """
    if args.eval_horizon and args.eval_horizon > args.episode_length:
        raise SystemExit(
            "--eval_horizon %d exceeds --episode_length %d, and the two share one environment whose episodes are "
            "capped at the shorter. Lower the evaluation horizon, or raise the episode length knowing that it "
            "changes what collection sees." % (args.eval_horizon, args.episode_length))

    # Evaluation needs all three of its arguments. Asking for a schedule or a budget while leaving the episode
    # count at zero produces a run that evaluates nothing and says nothing about it, while the horizon check above
    # and the guard's armed line both still print, so the log reads like a configured evaluation.
    # Tested for sign before anything is tested for truth, because a negative reads as present everywhere below
    # and as something else entirely downstream: a negative episode count loops zero times and still prints a
    # pooled success of zero, which the guard cannot tell from a measured one; a negative horizon scores a single
    # environment step and records it under the number that was asked for; and a negative schedule fires on every
    # cycle, since any integer is divisible by minus one.
    for name in ("eval_episodes", "eval_horizon", "eval_every", "ckpt_every"):
        value = getattr(args, name, 0)
        if value is not None and value < 0:
            raise SystemExit(
                "--%s %d is negative. Counts and schedules here are zero to switch off and positive to switch "
                "on; a negative one reads as switched on and behaves as neither." % (name, value))
    if args.episode_length < 1:
        raise SystemExit("--episode_length %d gives an episode no steps to run. Use at least 1."
                         % args.episode_length)

    if (args.eval_every or args.eval_horizon) and not args.eval_episodes:
        raise SystemExit(
            "--eval_every or --eval_horizon was given without --eval_episodes, which leaves the episode count at "
            "zero and means no evaluation runs at all. Set --eval_episodes, or drop the other two.")
    # The mirror of the case above, and it fails the same way. Evaluation runs from the schedule, so an episode
    # count without one is a budget nothing ever spends: the run reports a configured evaluation and performs none.
    if args.eval_episodes and not args.eval_every:
        raise SystemExit(
            "--eval_episodes was given without --eval_every, which leaves the schedule empty and means no "
            "evaluation runs at all. Set --eval_every, or drop the episode count.")
    # A schedule longer than the run never comes round, so the evaluation is configured, reported as configured
    # and never performed. One step further out than the two checks above, and the same failure.
    if args.eval_every and args.eval_every > args.updates:
        raise SystemExit(
            "--eval_every %d is longer than the %d update cycles this run has, so no evaluation would ever run. "
            "Lower it, or raise --updates." % (args.eval_every, args.updates))
    if args.trip_success_floor is not None and not (args.eval_episodes and args.eval_every):
        raise SystemExit(
            "--trip_success_floor was given but no evaluation will run, so the floor can never be tested. Set "
            "--eval_episodes and --eval_every, or drop the floor.")


def check_regime_args(args) -> None:
    """Refuse adapter settings under the regime that builds no adapters.

    They are read only by the two regimes that build adapters. Under full they are accepted and inert: a run at
    rank 8 and one at rank 512 produce the same trainable count and the same manifest line, so nothing in the
    output distinguishes a run that meant to pass a rank from one that did not. Refused for the same reason every
    other incoherent combination here is, and refused before the policy is built rather than after, since the
    failure is in the arguments and needs no weights to find.
    """
    if args.regime == "full" and (args.adapter_rank != ADAPTER_RANK_DEFAULT
                                  or args.adapter_alpha != ADAPTER_ALPHA_DEFAULT):
        raise SystemExit(
            "--adapter_rank or --adapter_alpha was given with --regime full, which trains the action expert at "
            "full rank and builds no adapters, so neither value reaches anything. Use --regime frozen or lora, "
            "or drop them.")

    # The integration needs at least one step; at zero the step size is a division by the count and the sampler
    # raises from inside the grid, far from the argument that caused it.
    if args.denoise_steps < 1:
        raise SystemExit(
            "--denoise_steps %d leaves the integration no steps to take, and the step size is one divided by that "
            "count. Use at least 1." % args.denoise_steps)

    # Zero or fewer grades an empty surface. The likelihood is then a sum over nothing, the ratio is exactly one,
    # the policy gradient is identically zero and only the critic moves, while collection still executes one
    # action per decision, so the graded surface and the executed one disagree. Every diagnostic reads healthy,
    # including the first update invariant, which a ratio of one satisfies trivially. This is the failure the
    # rest of this file is built to make impossible, reachable through a single argument.
    if args.action_steps < 1:
        raise SystemExit(
            "--action_steps %d grades no part of the chunk, so the likelihood ratio is one by construction and "
            "the policy gradient is zero, while collection still executes an action. Use at least 1."
            % args.action_steps)


def check_trainer_args(args) -> None:
    """Refuse arguments that would train nothing. Only the trainer takes these.

    Kept apart from the shared checks on purpose. The evaluation command calls the shared function and declares
    none of these, so folding them together crashed that command on every invocation until it was split back out.
    A check that is right for one entry point and absent from the other is a shared function; a check that only
    one entry point can satisfy is not.
    """
    # A scale of zero makes the sampler deterministic, so the likelihood is a constant, the ratio is one and the
    # surrogate carries no gradient. A clip of zero scales every gradient to nothing while the number logged as
    # evidence is the norm from before the clip, so the log cannot show it. Both leave the critic training and
    # every printed diagnostic reading healthy.
    if args.noise_level <= 0:
        raise SystemExit(
            "--noise_level %g makes the sampler deterministic, so the likelihood is constant, the ratio is one "
            "by construction and the policy gradient is zero. Use a positive scale." % args.noise_level)
    if args.grad_clip <= 0:
        raise SystemExit(
            "--grad_clip %g scales every gradient to zero, and the norm reported afterwards is the one from "
            "before the clip, so nothing in the log would show it. To disable clipping give a large finite "
            "value or inf." % args.grad_clip)
    if args.updates < 1:
        raise SystemExit("--updates %d runs no cycle at all. Use at least 1." % args.updates)
    if args.epochs < 1:
        raise SystemExit(
            "--epochs %d runs the update loop zero times per cycle, so a whole collection cycle would be paid "
            "for and nothing graded. Use at least 1." % args.epochs)
    if args.rollout_steps < 1:
        raise SystemExit(
            "--rollout_steps %d stores no timestep, so the cycle would grade nothing and collection would stack "
            "an empty list after the weights are loaded. Use at least 1." % args.rollout_steps)
    # The critic appears in the loss only through this coefficient, so at zero its gradient is exactly zero: the
    # learning rate the user was required to choose reaches nothing, the advantages are computed against a head
    # that never trains, and the line printed to prove the critic was built still prints.
    if args.value_coef <= 0:
        raise SystemExit(
            "--value_coef %g removes the only term the critic appears in, so --critic_lr reaches nothing and "
            "every advantage is measured against a head that never trains. Use a positive weight."
            % args.value_coef)
    # At zero the clipped objective is flat wherever the ratio has moved at all, so the gradient vanishes as soon
    # as the policy does anything; negative is worse, since the clip bounds cross.
    if args.clip_range <= 0:
        raise SystemExit(
            "--clip_range %g leaves no width around a ratio of one, so the objective is flat and the gradient "
            "vanishes as soon as the policy moves. Use a positive width." % args.clip_range)
    # A rate of zero leaves the surface the manifest has just reported as trainable exactly where it started,
    # while the run collects, grades, writes metrics, saves a checkpoint and reports a completed training. The
    # optimiser proof prints the rate, so the evidence is on screen, but the refusal a line above this one already
    # says a critic that never trains is not a configuration worth allowing, and a rate of zero reaches it too.
    for name in ("actor_lr", "critic_lr"):
        rate = getattr(args, name)
        if rate <= 0:
            raise SystemExit(
                "--%s %g leaves that surface where it started while the run reports a completed training. A "
                "negative rate ascends the objective rather than descending it. Use a positive rate."
                % (name, rate))
    if not 0 < args.gamma <= 1:
        raise SystemExit("--gamma %g is outside (0, 1]; a discount outside that range does not discount."
                         % args.gamma)
    if not 0 <= args.gae_lambda <= 1:
        raise SystemExit("--gae_lambda %g is outside [0, 1]; the trace decay is a weight, not a gain."
                         % args.gae_lambda)


def check_resume_is_not_a_noop(start: int, updates: int) -> None:
    """Refuse a resume that lands at or past the requested end.

    The loop would run no cycle at all, fall through to the final save and print a completed training, which is
    the same end state the run refuses when a cycle applies none of its updates, reached by a different road. The
    ordinary way to arrive here is extending a run and forgetting to raise the count.
    """
    if start >= updates:
        raise SystemExit(
            "--resume found a checkpoint at step %d and --updates is %d, so no cycle would run and the run would "
            "report a completed training without training. Raise --updates above %d, or drop --resume."
            % (start, updates, start))


TRIP_THRESHOLDS = ("trip_ratio_median", "trip_ratio_max", "trip_success_floor", "trip_window")


def check_tripwire_args(args) -> None:
    """Refuse a guard that is half configured, in either direction.

    Thresholds without the flag are the worse half and were the silent one. The guard is then constructed mute:
    it prints no armed line, and the armed line is the only evidence in a run's output that a guard exists at
    all, so a run with no guard is indistinguishable from a guarded one. Dropping or mistyping the bare flag
    while keeping the four numbers is the ordinary way to get there, and it produces exactly the unwatched run
    the guard was written to prevent.
    """
    if not args.tripwire:
        given = [name for name in TRIP_THRESHOLDS if getattr(args, name) is not None]
        if given:
            raise SystemExit(
                "%s given without --tripwire, so the guard is never armed and none of them reaches anything. Add "
                "--tripwire, or drop them." % ", ".join("--" + g for g in given))
        return
    missing = [name for name in TRIP_THRESHOLDS if getattr(args, name) is None]
    if missing:
        raise SystemExit("--tripwire requires: " + ", ".join("--" + m for m in missing))
    # A window wider than the run never closes, so the ratio limits are never tested and the line that reports
    # the window closing, which is the evidence the guard observed anything, never prints.
    if args.trip_window > args.updates:
        raise SystemExit(
            "--trip_window %d is wider than the %d update cycles this run has, so the window never closes and "
            "the ratio limits are never tested. Lower it, or raise --updates."
            % (args.trip_window, args.updates))


def parse_tasks(spec: str):
    """Read the task list, refusing a malformed entry rather than raising from inside the parse.

    These were the only arguments in this file that answered a mistake with a traceback: a suite with no index
    unpacked one value where two were wanted, and a non numeric index failed inside int(). Both are the first
    required argument, so they are the first thing a reader gets wrong.
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                "--tasks entry %r has no index. Each entry is a suite and a task index separated by a colon, as "
                "in libero_10:0, and several are separated by commas." % part)
        suite, idx = part.rsplit(":", 1)
        if not suite:
            raise SystemExit("--tasks entry %r names no suite before the colon." % part)
        try:
            out.append((suite, int(idx)))
        except ValueError:
            raise SystemExit(
                "--tasks entry %r has %r where a task index was wanted, which must be a whole number."
                % (part, idx))
    if not out:
        raise SystemExit("--tasks parsed to nothing from %r; give at least one suite:index entry." % spec)
    return out


def check_task_args(tasks) -> None:
    """Refuse a suite this harness cannot drive, or an index the simulator does not have, before the policy loads.

    Otherwise both are discovered after the weights are read, inside a spawned worker, by an exception naming
    neither the argument nor this repository. The suite names are already in scope; the task count per suite is
    read where it lives, and a failure to read it leaves the index unchecked rather than stopping the run, since
    that is a fact about the installed simulator rather than about these arguments.
    """
    try:
        from env_libero import TASK_SUITE_MAX_STEPS
    except ImportError:
        from .env_libero import TASK_SUITE_MAX_STEPS

    known = sorted(TASK_SUITE_MAX_STEPS)
    for suite, idx in tasks:
        if suite not in TASK_SUITE_MAX_STEPS:
            # The list is the step limit table this harness reads, not the simulator's own registry, and the two
            # are not the same set: the simulator carries suites this table has no published limit for.
            raise SystemExit("--tasks names the suite %r, which this harness does not drive, because the step "
                             "limit table it reads has no entry for it. It drives: %s."
                             % (suite, ", ".join(known)))
        if idx < 0:
            raise SystemExit("--tasks names the index %d in %s; indices start at zero." % (idx, suite))
        try:
            from libero.libero import benchmark as libero_benchmark
            n = len(libero_benchmark.get_benchmark_dict()[suite]().tasks)
        except Exception:
            continue
        if idx >= n:
            raise SystemExit("--tasks names index %d in %s, which has %d tasks, numbered 0 to %d."
                             % (idx, suite, n, n - 1))


def main(argv=None):
    args = parse_args(argv)
    check_tripwire_args(args)
    check_horizon_args(args)
    check_regime_args(args)
    check_trainer_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # Seed before anything constructs a module or an environment. Sampling consumes the global streams on every
    # step, so where this line sits determines the stream position the first update sees.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Printed rather than assumed. This repository states a card as a requirement, and a run that quietly fell
    # back to the host looks identical in its arguments to one that did not, while taking long enough that the
    # difference is noticed hours later.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  [device] %s" % ("cuda" if device.type == "cuda" else
                             "cpu, no CUDA device was visible; this will be very slow"), flush=True)
    tasks = parse_tasks(args.tasks)
    check_task_args(tasks)

    cfg = SmolVLAConfig(
        model_path=args.base_policy,
        num_steps=args.denoise_steps,
        noise_method="flow_sde",
        noise_level=args.noise_level,
        device=str(device),
        add_value_head=True,
        value_head_layernorm=args.critic_layernorm,
        value_head_popart=args.popart,
    )
    model = build_model(SmolVLAForRLActionPrediction, cfg, device, args.regime,
                        args.adapter_rank, args.adapter_alpha)
    optimizer = build_optimizer(model, args.actor_lr, args.critic_lr)

    # The executed chunk cannot be longer than the chunk the policy emits, which the wrapper reads off the policy
    # rather than carrying as a constant. Left unchecked this raises an index error inside collection, after the
    # environments are spawned and the first cycle has begun, which is loud but far from its cause.
    if args.action_steps > model.action_chunk:
        raise SystemExit(
            "--action_steps %d is longer than the chunk this policy emits, which is %d, so collection would index "
            "past the end of it. Lower it to at most %d."
            % (args.action_steps, model.action_chunk, model.action_chunk))

    # Checked here rather than at the first update, because the failure is a configuration error and the run
    # would otherwise spend a collection cycle before finding out. The anchor recovers the pretrained policy by
    # switching adapters off, so it needs adapters to exist.
    if args.kl_coef != 0.0 and not reference_is_available(model):
        raise SystemExit(
            "--kl_coef %g was given, but regime %r trains the policy in place and leaves no adapters to switch "
            "off, so there is no pretrained policy left to anchor to. Use --regime frozen or lora, or set "
            "--kl_coef 0." % (args.kl_coef, args.regime))

    # What the checkpoints will record about the run that produced them. The evaluation command takes the same
    # arguments and cannot tell from the weights alone which ones were used, so scoring a checkpoint under a
    # different chunk length or step count measures a policy this run never produced while every printed line
    # still names the checkpoint. Recorded rather than left to whoever remembers the command.
    provenance = {"regime": args.regime, "denoise_steps": args.denoise_steps,
                  "action_steps": args.action_steps, "noise_level": args.noise_level}

    start = load_checkpoint(args.output_dir, model, optimizer) if args.resume else 0

    check_resume_is_not_a_noop(start, args.updates)

    # Effect proof for the critic options. Both are flags, and a flag that is parsed and never reaches anything
    # looks identical in the arguments and in any configuration dump to one that works, so what is printed here
    # is what the constructed head actually contains.
    if model.value_head is not None:
        n_linear = sum(1 for m in model.value_head.mlp if isinstance(m, torch.nn.Linear))
        n_norm = sum(1 for m in model.value_head.mlp if isinstance(m, torch.nn.LayerNorm))
        print("  [critic] %d linear layers, %d normalisation layers, rescaling %s, scale %.4g offset %.4g"
              % (n_linear, n_norm, "on" if model.value_head.popart else "off",
                 float(model.value_head.pa_sigma), float(model.value_head.pa_mu)), flush=True)

    ve, _task_of, init_obs, suite_limit = build_env(tasks, args.episode_length, args.seed)
    env_pre, env_post = make_processors(tasks[0][0], model.lerobot_cfg)
    prompts = get_task_prompts(ve)
    n_envs = ve.num_envs
    print("  [env] %d workers, episode length %d, suite limit %s"
          % (n_envs, args.episode_length, suite_limit if suite_limit else "unknown"), flush=True)
    # Truncating below the suite's own limit is a legitimate choice and a silent trap. A short cap makes long
    # horizon tasks unsolvable, and the run then reports no reward movement, which reads exactly like a setup
    # that cannot learn. The comparison is against the suite's published limit, not against what the environment
    # reports: an environment built with an explicit length reports that length back, so comparing with it asks
    # the run whether it agrees with itself, which it always does.
    if suite_limit and args.episode_length < suite_limit:
        print("  [env] episodes are cut at %d, below the suite's own limit of %d. Tasks that need more steps "
              "cannot succeed here, so an absence of reward may be the cap rather than the policy."
              % (args.episode_length, suite_limit), flush=True)

    guard = Tripwire(args.tripwire, args.trip_ratio_median, args.trip_ratio_max,
                     args.trip_success_floor, args.trip_window, args.output_dir)
    guard.arm()

    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    rolling = init_obs
    ratio_checked = False

    for cycle in range(start, args.updates):
        set_initial_states(ve, "train", n_envs, per_task=1, eval_batch=0, n_init_states=n_envs)
        model.train()
        t0 = time.time()
        buf = collect_rollout(model, ve, env_pre, env_post, prompts, args.rollout_steps,
                              initial_obs=rolling, device=device, mode="train",
                              n_action_steps=args.action_steps)
        rolling = buf["last_raw_obs"]

        adv, returns = compute_gae(buf["rewards"].to(device), buf["values"].to(device),
                                   buf["dones"].to(device), buf["bootstrap"].to(device),
                                   gamma=args.gamma, lam_gae=args.gae_lambda)
        adv = normalize_advantages(adv)

        # One stored timestep per optimiser step. The observation the update re evaluates against belongs to a
        # single timestep, so a wider slice would pair one observation with several timesteps' trajectories.
        last = {}
        stepped = skipped = 0
        for _epoch in range(args.epochs):
            for t in range(args.rollout_steps):
                last = ppo_step_stored(
                    model, buf["env_obs"][t],
                    chains=buf["chains"][:, t],
                    denoise_inds=buf["denoise_inds"][:, t],
                    prev_logp=buf["prev_logprobs"][:, t],
                    noise_level=args.noise_level, optimizer=optimizer,
                    advantages=adv[:, t], returns=returns[:, t],
                    clip_range=args.clip_range, value_coef=args.value_coef, kl_coef=args.kl_coef,
                    grad_clip_norm=args.grad_clip, kl_anchor=args.kl_coef != 0.0,
                    n_action_steps=args.action_steps)

                if last.get("step_skipped"):
                    skipped += 1
                else:
                    stepped += 1

                # The first update of the run happens before any parameter has moved, so the recomputed
                # likelihood must equal the stored one and the ratio must be one. Any other value means the two
                # are computed on different surfaces, in which case every update afterwards is driven by that
                # discrepancy rather than by the advantage, and nothing downstream would say so. Checked once,
                # here, rather than left to a reader noticing a printed number.
                if not ratio_checked:
                    ratio_checked = True
                    r = float(last.get("ratio_mean", float("nan")))
                    print("  [invariant] first update ratio %.6f, tolerance %g" % (r, RATIO_TOLERANCE),
                          flush=True)
                    if not abs(r - 1.0) <= RATIO_TOLERANCE:
                        ve.close()
                        raise SystemExit(
                            "first update ratio was %.6f, not one within %g. The stored and recomputed "
                            "likelihoods are on different surfaces; stopping rather than training on the "
                            "difference between two formulations." % (r, RATIO_TOLERANCE))

        # The clip range is recorded alongside the statistics it shapes. A reader that has to assume it cannot
        # tell a run that used a different one, and would draw the band in the wrong place without saying so.
        record = {"cycle": cycle, "seconds": round(time.time() - t0, 1),
                  "clip_range": args.clip_range, "updates_stepped": stepped, "updates_skipped": skipped,
                  "reward_mean": float(buf["rewards"].mean()), **last}
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print("  cycle %d/%d  ratio %.3f  clip %.3f  grad %.2f  reward %.4f"
              % (cycle, args.updates, last.get("ratio_mean", float("nan")),
                 last.get("clip_frac", float("nan")), last.get("grad_norm", float("nan")),
                 record["reward_mean"]), flush=True)

        # A cycle in which the non finite guard dropped every update trained on nothing. Left alone the run
        # continues, reaches the end, prints success and saves a checkpoint no gradient ever touched, which is not
        # a shorter version of a trained run but a differently shaped failure that reads like one.
        if stepped == 0:
            ve.close()
            raise SystemExit(
                "cycle %d applied none of its %d updates, every one dropped for a non finite loss or gradient. "
                "Continuing would report a completed run whose parameters never moved. Check the manifest, the "
                "batch size implied by --rollout_steps and the number of environments, and the reward scale."
                % (cycle, skipped))

        if guard.observe_update(last.get("ratio_median", 1.0), last.get("ratio_max", 1.0)):
            ve.close()
            return TRIPPED_EXIT_CODE

        if args.eval_every and (cycle + 1) % args.eval_every == 0 and args.eval_episodes:
            # The chunk execution length is forwarded on purpose. Evaluating under a different regime than the
            # one trained measures a policy the run never produced, and nothing in the output would say so.
            rate = evaluate(model, ve, env_pre, env_post, prompts, tasks,
                            episodes_per_task=args.eval_episodes, seed=args.seed,
                            horizon=args.eval_horizon or args.episode_length,
                            events_path=os.path.join(args.output_dir, "events.jsonl"), device=device,
                            n_action_steps=args.action_steps, cycle=cycle + 1)
            print("  [eval] cycle %d pooled success %.3f" % (cycle + 1, rate), flush=True)
            if guard.observe_eval(rate):
                ve.close()
                return TRIPPED_EXIT_CODE
            set_initial_states(ve, "train", n_envs, per_task=1, eval_batch=0, n_init_states=n_envs)
            rolling, _ = ve.reset()

        if args.ckpt_every and (cycle + 1) % args.ckpt_every == 0:
            save_checkpoint(args.output_dir, cycle + 1, model, optimizer, extra=provenance)

    save_checkpoint(args.output_dir, args.updates, model, optimizer, extra=provenance)
    ve.close()
    print("TRAINING_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
