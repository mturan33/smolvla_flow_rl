# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. See the LICENSE file at the root of this repository.
"""Executable form of the invariants the protocol requires, plus the negative tests for the gates.

These need no GPU and step no simulator. They cover the parts that can be checked that way: the likelihood
surface used by the ratio, the completeness gate, and the checkpoint round trip. One invariant that needs a
policy on a device, the first update ratio, is checked by the trainer itself and printed by the contact example.
The rest of those, behavioural parity and cross process reproducibility, are marked manual in the protocol and
are for a reader to perform; nothing here or in the example performs them.

Some of them import the trainer, which imports the simulator package, so that package must be installed and its
one time question already answered. `scripts/run_gates.sh` checks for that and prints the one line remedy; run
this file directly and you get the simulator's own error instead.

Every gate here is also given the defect it exists to catch and required to fail on it. A gate that has only
ever passed has not been tested.

Usage: python tests/test_invariants.py
"""

import argparse
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch

# Set before the first repository import; it only affects imports that follow it. Without it, running this file
# directly, which is how its own usage line says to run it, leaves compiled copies beside the source and the very
# next artefact check fails on a tree that was clean a moment earlier.
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from completeness_gate import EvalReading  # noqa: E402
from flow_sde import (get_logprob_norm, get_timesteps, recompute_logprob_value,  # noqa: E402
                      sample_mean_var, sde_sample)


def check(name, ok, detail=""):
    print("  %-46s %s%s" % (name, "PASS" if ok else "FAIL", (" " + detail) if detail else ""))
    return ok


def test_deterministic_branch_matches_constant_step():
    """The deterministic branch must step by the constant, not by the difference of two grid entries.

    The two agree in exact arithmetic. This checks the implementation took the form that also agrees in floating
    point, since the other one drifts away from the policy's own integration.
    """
    torch.manual_seed(0)
    # A step count whose reciprocal is not exactly representable. With a power of two every grid entry is exact,
    # the two forms agree bit for bit, and the negative case below could not tell them apart.
    n = 10
    ts = get_timesteps(n, torch.device("cpu"))
    x = torch.randn(3, 4, 5)
    v = torch.randn(3, 4, 5)
    mean, std = sample_mean_var(x, v, 2, ts, "flow_ode", 0.0)
    expected = x + (-1.0 / n) * v
    ok = torch.equal(mean, expected) and torch.count_nonzero(std) == 0

    # Negative case. The defect this exists to catch is stepping by the difference of two grid entries, which is
    # equal in exact arithmetic and not in floating point. If that form were indistinguishable here, a passing
    # test would say nothing, so the test itself would be worthless.
    defective = x + (ts[3] - ts[2]) * v
    distinguishable = not torch.equal(defective, expected)
    return check("deterministic step uses the constant", ok and distinguishable,
                 "defect distinguishable: %s" % distinguishable)


def test_zero_scale_contributes_nothing():
    """A deterministic step has zero scale and must contribute zero log density, not a divergence."""
    sample = torch.randn(4, 3)
    mu = sample.clone()
    zeros = torch.zeros_like(sample)
    lp = get_logprob_norm(sample, mu, zeros)
    ok = bool(torch.isfinite(lp).all()) and float(lp.abs().sum()) == 0.0

    # Negative case. Written without the guard, the same quantity is a division by zero and a log of zero, so it
    # is not finite. If it were finite anyway the guard would be untested decoration.
    naive = -0.5 * ((sample - mu) / zeros) ** 2 - torch.log(zeros)
    defect_shows = not bool(torch.isfinite(naive).all())
    return check("zero scale gives a zero contribution", ok and defect_shows,
                 "unguarded form non finite: %s" % defect_shows)


def _stub_velocity_field():
    """A velocity field that is a fixed, deterministic function of the state.

    Standing in for the policy lets the likelihood surface be exercised on a machine with no GPU and no
    simulator. What is being checked is the arithmetic that connects sampling to the update, not the policy.
    """
    bias = torch.randn(8, 5)

    def velocity_fn(x, _t):
        return x * 0.5 + bias, None

    return velocity_fn


def test_stored_and_recomputed_logprob_agree():
    """The invariant the whole update rests on: unchanged parameters must give a ratio of exactly one.

    The stored log probability comes from the sampler, the recomputed one from the update path. They are two
    separate pieces of code walking the same arithmetic, so nothing but a test keeps them on one surface. When
    they drift apart the run does not fail; it trains on the difference between two formulations, and the only
    symptom is a number that looks plausible.
    """
    torch.manual_seed(5)
    random.seed(5)
    n_steps, chunk, env_dim = 6, 4, 3
    velocity_fn = _stub_velocity_field()

    out = sde_sample(velocity_fn, torch.randn(2, 8, 5), n_steps, "flow_sde", 0.2,
                     action_chunk=chunk, action_env_dim=env_dim, mode="train", value_fn=None)
    lp, _values = recompute_logprob_value(velocity_fn, out["chains"], out["denoise_inds"], n_steps,
                                          "flow_sde", 0.2, compute_values=False, value_fn=None)

    stored = out["prev_logprobs"].flatten(start_dim=1).sum(dim=-1)
    recomputed = lp[:, 0][:, :chunk, :env_dim].flatten(start_dim=1).sum(dim=-1)
    ratio = (recomputed - stored).exp()
    ok = bool(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5))

    # Negative case. Recomputing at a different integration step is the defect this guards against, and it must
    # be visible as a ratio away from one, otherwise the positive result above would mean nothing.
    wrong_step = (out["denoise_inds"] + 1) % (n_steps - 1)
    lp_wrong, _ = recompute_logprob_value(velocity_fn, out["chains"], wrong_step, n_steps,
                                          "flow_sde", 0.2, compute_values=False, value_fn=None)
    wrong = lp_wrong[:, 0][:, :chunk, :env_dim].flatten(start_dim=1).sum(dim=-1)
    defect_shows = not bool(torch.allclose((wrong - stored).exp(), torch.ones_like(ratio), atol=1e-5))

    return check("stored and recomputed likelihoods agree", ok and defect_shows,
                 "ratio %s, wrong step distinguishable: %s"
                 % (["%.6f" % float(v) for v in ratio], defect_shows))


def test_evaluation_sampling_is_deterministic():
    """Two evaluations of the same weights from the same noise must produce the same chunk."""
    torch.manual_seed(7)
    random.seed(7)
    velocity_fn = _stub_velocity_field()
    noise = torch.randn(2, 8, 5)
    a = sde_sample(velocity_fn, noise, 6, "flow_sde", 0.3, action_chunk=4, action_env_dim=3,
                   mode="eval", value_fn=None)
    b = sde_sample(velocity_fn, noise, 6, "flow_sde", 0.3, action_chunk=4, action_env_dim=3,
                   mode="eval", value_fn=None)
    same = bool(torch.equal(a["actions"], b["actions"]))

    # Negative case: the training mode of the same call is stochastic, so the two modes must not be confusable.
    c = sde_sample(velocity_fn, noise, 6, "flow_sde", 0.3, action_chunk=4, action_env_dim=3,
                   mode="train", value_fn=None)
    differs = not bool(torch.equal(a["actions"], c["actions"]))
    return check("evaluation sampling is deterministic", same and differs,
                 "train mode differs: %s" % differs)


def _write_events(path, tasks, per_task, drop=0):
    n = 0
    with open(path, "w") as fh:
        for b in range(per_task):
            for i, _t in enumerate(tasks):
                if drop and n >= len(tasks) * per_task - drop:
                    break
                fh.write(json.dumps({"task_idx": i, "batch": b, "succ_step": 5 if i == 0 else -1}) + "\n")
                n += 1
    return n


def test_completeness_gate():
    """Complete input passes and yields a number; short input fails and yields none."""
    tasks, per_task = [1, 2, 3], 4
    with tempfile.TemporaryDirectory() as d:
        full = os.path.join(d, "full.jsonl")
        _write_events(full, tasks, per_task)
        r_full = EvalReading(full, tasks, per_task)

        short = os.path.join(d, "short.jsonl")
        _write_events(short, tasks, per_task, drop=3)
        r_short = EvalReading(short, tasks, per_task)

        ok = (r_full.complete and r_full.success_rate() is not None
              and not r_short.complete and r_short.success_rate() is None)
        return check("completeness gate accepts full, refuses short", ok,
                     "%s | %s" % (r_full.status, r_short.status))


def test_completeness_gate_ignores_duplicates():
    """A rewritten batch must not let one episode count twice."""
    tasks, per_task = [1, 2], 3
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "dup.jsonl")
        _write_events(p, tasks, per_task)
        with open(p, "a") as fh:
            fh.write(json.dumps({"task_idx": 0, "batch": 0, "succ_step": 5}) + "\n")
        r = EvalReading(p, tasks, per_task)
        return check("duplicate rows are reported, not counted",
                     r.complete and r.episodes == len(tasks) * per_task and r.duplicates == 1)


def test_completeness_gate_refuses_mixed_cycles():
    """Two checkpoints in one file must not be silently merged into a single rate.

    A training run that evaluates periodically appends to the same file. Keyed on task and batch alone, the later
    evaluation lands on the earlier one's keys, the count still reaches the expected total, and the reading looks
    complete while holding whichever weights wrote last.
    """
    tasks, per_task = [1, 2], 3
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mixed.jsonl")
        with open(p, "w") as fh:
            for cycle, step in ((5, 4), (10, -1)):
                for b in range(per_task):
                    for i, _t in enumerate(tasks):
                        fh.write(json.dumps({"cycle": cycle, "task_idx": i, "batch": b,
                                             "succ_step": step if i == 0 else -1}) + "\n")

        blind = EvalReading(p, tasks, per_task)
        picked = EvalReading(p, tasks, per_task, cycle=5)
        other = EvalReading(p, tasks, per_task, cycle=10)

        ok = (not blind.complete and blind.success_rate() is None
              and picked.complete and other.complete
              and picked.success_rate() != other.success_rate())
        return check("mixed cycles refused unless one is named", ok,
                     "%s | cycle5=%s cycle10=%s" % (blind.status, picked.success_rate(),
                                                    other.success_rate()))


def test_line_ending_gate_sees_files_without_an_extension(tmp_root=None):
    """The gate must judge a file by its bytes, not by its name.

    It used to select text files from an extension list, so files with no extension at all were never looked at,
    and it printed that the tree's line endings were consistent while two committed files carried carriage
    returns. That is the failure this whole suite exists to catch, one level up: a check that covers part of its
    input space and reports total success.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import check_syntax

    with tempfile.TemporaryDirectory() as d:
        # Written with the newline translation switched off. In text mode Python rewrites every newline to the
        # platform's own, so on Windows this fixture would arrive carrying the very defect the test then asks the
        # gate not to find, and the suite would fail on a machine where nothing is wrong.
        with open(os.path.join(d, "module.py"), "w", newline="") as fh:
            fh.write("x = 1\n")
        extensionless = os.path.join(d, "NOTICE")
        with open(extensionless, "wb") as fh:
            fh.write(b"a line\nanother line\n")

        argv = sys.argv
        try:
            sys.argv = ["check_syntax", d]
            before = check_syntax.main()
            with open(extensionless, "wb") as fh:
                fh.write(b"a line\r\nanother line\r\n")
            after = check_syntax.main()
            # A binary file with the same byte pattern must not be reported, or the gate becomes noise.
            with open(os.path.join(d, "blob.bin"), "wb") as fh:
                fh.write(b"\x00\x01\r\n\x00")
            with open(extensionless, "wb") as fh:
                fh.write(b"a line\nanother line\n")
            with_binary = check_syntax.main()
        finally:
            sys.argv = argv

        return check("line ending gate reads bytes, not names",
                     before == 0 and after == 1 and with_binary == 0,
                     "clean=%d planted=%d binary ignored=%s" % (before, after, with_binary == 0))


def test_import_closure_gate_catches_a_missing_module():
    """The closure gate must fail on a tree whose entry point imports something that is not there."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import check_import_closure as closure

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        with open(os.path.join(d, "requirements.txt"), "w") as fh:
            fh.write("numpy\n")
        with open(os.path.join(d, "src", "present.py"), "w") as fh:
            fh.write("import os\n")
        with open(os.path.join(d, "src", "entry.py"), "w") as fh:
            fh.write("import os\nimport numpy\nfrom present import thing\n")

        argv = sys.argv
        try:
            sys.argv = ["check_import_closure", d]
            before = closure.main()
            # The gate has two verdicts a stranger depends on, and both are planted here. A relative import of a
            # module the repository does not hold is MISSING; a third party import no line of requirements.txt
            # provides is UNACCOUNTED. Testing only the first leaves the second able to stop working unnoticed.
            with open(os.path.join(d, "src", "entry.py"), "a") as fh:
                fh.write("from . import absent_module\n")
            after_missing = closure.main()

            with open(os.path.join(d, "src", "entry.py"), "w") as fh:
                fh.write("import os\nimport numpy\nimport undeclared_third_party\n")
            after_undeclared = closure.main()
        finally:
            sys.argv = argv

        return check("closure gate passes clean, fails on either kind of gap",
                     before == 0 and after_missing == 1 and after_undeclared == 1,
                     "clean=%d missing=%d undeclared=%d" % (before, after_missing, after_undeclared))


def test_import_closure_gate_counts_only_tracked_files():
    """A module that exists on disk but is not tracked must not satisfy an import.

    This is the gate's whole reason for existing, and until this test was written it was the one case the gate did
    not cover: the walk read the filesystem, so a module sitting beside the checkout and never committed looked
    exactly like one a clone would receive. Exercised through repo_modules with an explicit tracked set rather
    than by building a repository, so the test needs no git and cannot pass for an accidental reason.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import check_import_closure as closure

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "src"))
        for name in ("committed.py", "untracked.py"):
            with open(os.path.join(d, "src", name), "w") as fh:
                fh.write("VALUE = 1\n")

        tracked = {os.path.normpath(os.path.join("src", "committed.py"))}
        with_git = closure.repo_modules(d, tracked)
        without_git = closure.repo_modules(d, None)

        return check("closure gate counts tracked files, not files on disk",
                     "committed" in with_git and "untracked" not in with_git
                     and "committed" in without_git and "untracked" in without_git,
                     "tracked=%s filesystem=%s" % (sorted(with_git), sorted(without_git)))


def test_evaluation_arguments_are_refused_in_both_directions():
    """Every incomplete evaluation configuration must be refused, not only the half that was thought of first.

    An episode count without a schedule and a schedule without an episode count fail identically: the run reports
    a configured evaluation and performs none. The success floor is the sharper case, because its own error text
    named both arguments while the condition tested one.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    import train_flow_rl

    def refused(**over):
        args = argparse.Namespace(eval_horizon=0, episode_length=520, eval_every=0, eval_episodes=0,
                                  trip_success_floor=None, updates=10)
        for k, v in over.items():
            setattr(args, k, v)
        try:
            train_flow_rl.check_horizon_args(args)
            return False
        except SystemExit:
            return True

    cases = {
        "schedule without episodes": refused(eval_every=5),
        "horizon without episodes": refused(eval_horizon=520),
        "episodes without schedule": refused(eval_episodes=4),
        "floor without either": refused(trip_success_floor=0.3),
        "floor with episodes but no schedule": refused(eval_episodes=4, trip_success_floor=0.3),
        "horizon above the episode length": refused(eval_every=5, eval_episodes=4, eval_horizon=999),
        "schedule longer than the run": refused(eval_every=20, eval_episodes=4, eval_horizon=520),
    }
    accepted = not refused(eval_every=5, eval_episodes=4, eval_horizon=520)
    bad = [name for name, was in cases.items() if not was]

    return check("incomplete evaluation arguments refused, complete one accepted",
                 not bad and accepted,
                 "unrefused=%s complete_accepted=%s" % (bad or "none", accepted))


def test_adapter_arguments_are_refused_where_they_do_nothing():
    """An adapter setting under the regime that builds no adapters must be refused, not silently ignored.

    Under full, a run at rank 8 and a run at rank 512 produce the same trainable count and the same manifest, so
    the argument is inert and nothing in the output says so.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    import train_flow_rl

    # Built by the real parser, not by hand. A hand written namespace carries exactly the fields the test author
    # thought of, so a check that starts reading a new argument passes here and fails for a user; asking the
    # parser means the test sees what the command sees.
    def refused(regime, rank=train_flow_rl.ADAPTER_RANK_DEFAULT, alpha=train_flow_rl.ADAPTER_ALPHA_DEFAULT,
                action_steps=1, noise_level=0.1, grad_clip=1.0, updates=10, rollout_steps=4,
                value_coef=0.5, clip_range=0.2, denoise_steps=10, actor_lr=1e-5, gamma=0.99):
        args = train_flow_rl.parse_args([
            "--tasks", "libero_10:0", "--base_policy", "/nonexistent", "--output_dir", "/tmp",
            "--noise_level", str(noise_level), "--actor_lr", str(actor_lr), "--critic_lr", "1e-3",
            "--gamma", str(gamma), "--gae_lambda", "0.95", "--regime", regime,
            "--updates", str(updates), "--rollout_steps", str(rollout_steps), "--episode_length", "520",
            "--denoise_steps", str(denoise_steps), "--seed", "0",
            "--adapter_rank", str(rank), "--adapter_alpha", str(alpha),
            "--action_steps", str(action_steps), "--grad_clip", str(grad_clip),
            "--value_coef", str(value_coef), "--clip_range", str(clip_range),
        ])
        try:
            # Both, because this is the trainer's own path and it calls both. The split between them is about
            # which arguments the evaluation command also declares, not about which ones matter here.
            train_flow_rl.check_regime_args(args)
            train_flow_rl.check_trainer_args(args)
            return False
        except SystemExit:
            return True

    # Zero grades an empty surface: the ratio is one by construction and the gradient is zero, while collection
    # still executes an action. Every printed diagnostic reads healthy, so nothing but this refuses it.
    ok = (refused("full", rank=512) and refused("full", alpha=99)
          and refused("lora", action_steps=0) and refused("lora", action_steps=-1)
          and refused("lora", noise_level=0.0) and refused("lora", grad_clip=0.0)
          and refused("lora", updates=0) and refused("lora", rollout_steps=0)
          and refused("lora", value_coef=0.0) and refused("lora", clip_range=0.0)
          and refused("lora", denoise_steps=0) and refused("lora", actor_lr=0.0)
          and refused("lora", actor_lr=-1.0) and refused("lora", gamma=5.0)
          and not refused("full") and not refused("lora", rank=512) and not refused("frozen", rank=512))
    return check("arguments that would train nothing are refused", ok,
                 "rank=%s alpha=%s steps0=%s steps-1=%s noise0=%s clip0=%s updates0=%s sane=%s"
                 % (refused("full", rank=512), refused("full", alpha=99), refused("lora", action_steps=0),
                    refused("lora", action_steps=-1), refused("lora", noise_level=0.0),
                    refused("lora", grad_clip=0.0), refused("lora", updates=0), not refused("full")))


def test_half_configured_tripwire_is_refused_in_both_directions():
    """Thresholds without the flag are refused, not accepted and ignored.

    The guard prints an armed line and nothing else, so a guard that was never armed leaves a run that reads
    exactly like a guarded one.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    import train_flow_rl

    def refused(tripwire, **thresholds):
        args = argparse.Namespace(tripwire=tripwire, updates=10,
                                  **{k: None for k in train_flow_rl.TRIP_THRESHOLDS})
        for k, v in thresholds.items():
            setattr(args, k, v)
        try:
            train_flow_rl.check_tripwire_args(args)
            return False
        except SystemExit:
            return True

    full = dict(trip_ratio_median=2.0, trip_ratio_max=10.0, trip_success_floor=0.0, trip_window=5)
    ok = (refused(False, **full)                       # thresholds without the flag
          and refused(False, trip_window=5)            # even one of them
          and refused(True, trip_window=5)             # the flag without the rest
          and not refused(True, **full)                # complete
          and not refused(False))                      # neither
    return check("a half configured guard is refused from either side", ok,
                 "thresholds_only=%s one_only=%s flag_only=%s complete=%s neither=%s"
                 % (refused(False, **full), refused(False, trip_window=5), refused(True, trip_window=5),
                    refused(True, **full), refused(False)))


def test_every_shared_check_is_satisfiable_by_both_entry_points():
    """A check shared by the two commands must only read arguments both of them declare.

    This exists because a trainer only check was once folded into the shared validator, and the evaluation
    command, which declares none of those arguments, then raised AttributeError on every invocation. Unit testing
    the check with a hand built namespace could not catch that; asking the real parser could.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    import eval_cli
    from train_flow_rl import check_regime_args

    args = eval_cli.parse_args([
        "--tasks", "libero_10:0", "--base_policy", "/nonexistent", "--checkpoint", "/nonexistent.pt",
        "--output_dir", "/tmp", "--episodes", "1", "--horizon", "100", "--seed", "0",
        "--denoise_steps", "10", "--regime", "full",
    ])
    missing = None
    try:
        check_regime_args(args)
    except SystemExit:
        pass          # a refusal is fine; reading an argument that does not exist is not
    except AttributeError as exc:
        missing = str(exc)

    return check("the shared argument checks read only what both commands declare", missing is None,
                 missing or "no attribute reached that the evaluation parser does not produce")


def test_the_download_helper_refuses_an_output_inside_the_tree():
    """The one mechanism protecting the claim that no weights are distributed here had no test behind it.

    The ignore rules are not a second line: the same download writes processor configurations that no rule
    matches, so if this refusal were weakened or moved below the fetch, a single add would commit them.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "scripts"))
    from download_base_policy import refuse_in_tree

    def refused(path):
        try:
            refuse_in_tree(path)
            return False
        except SystemExit:
            return True

    with tempfile.TemporaryDirectory() as d:
        outside = os.path.join(d, "base")
        link = os.path.join(d, "link")
        try:
            os.symlink(root, link)
            through_a_link = refused(os.path.join(link, "base"))
        except (OSError, NotImplementedError):
            through_a_link = True  # no symlinks on this filesystem; the case cannot arise here either

        inside = [root, os.path.join(root, "base"), os.path.join(root, "scripts", "base"),
                  os.path.join(root, "..", os.path.basename(root), "base")]
        ok = all(refused(p) for p in inside) and through_a_link and not refused(outside)
        return check("the download helper refuses an output inside the tree", ok,
                     "inside=%s symlink=%s outside_allowed=%s"
                     % ([refused(p) for p in inside], through_a_link, not refused(outside)))


def test_a_hub_identifier_is_not_mistaken_for_a_path():
    """The base policy guard must catch a mistyped path without rejecting repository identifiers.

    When it was first written it rejected every namespaced hub id, because those carry a slash, and then told the
    reader to pass one. The distinction is what this checks.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from action_model import _is_local_path

    identifiers = ["HuggingFaceVLA/smolvla_libero", "lerobot/smolvla_base", "gpt2"]
    paths = ["/tmp/base", "./base", "../base", "~/base", "a/b/c", "C:\\base", "/base/"]
    wrong = ([i for i in identifiers if _is_local_path(i)]
             + [p for p in paths if not _is_local_path(p)])
    return check("a hub identifier is not mistaken for a path", not wrong,
                 "misclassified=%s" % (wrong or "none"))


def test_resume_that_would_run_nothing_is_refused():
    """A resume landing at or past the requested end must stop, not print a completed training."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    import train_flow_rl

    def refused(start, updates):
        try:
            train_flow_rl.check_resume_is_not_a_noop(start, updates)
            return False
        except SystemExit:
            return True

    ok = refused(10, 10) and refused(12, 10) and not refused(9, 10) and not refused(0, 10)
    return check("a resume that would run no cycle is refused", ok,
                 "at=%s past=%s before=%s fresh=%s"
                 % (refused(10, 10), refused(12, 10), refused(9, 10), refused(0, 10)))


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)


def _draws():
    return (float(torch.randn(1)), float(np.random.rand()), random.random())


def test_checkpoint_restores_random_state():
    """Saving and loading must continue the same random sequence, on all three streams."""
    with tempfile.TemporaryDirectory() as d:
        m = _Tiny()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)

        torch.manual_seed(11), np.random.seed(11), random.seed(11)
        _ = _draws()
        expected = _draws()

        torch.manual_seed(11), np.random.seed(11), random.seed(11)
        _ = _draws()
        save_checkpoint(d, 3, m, opt)

        torch.manual_seed(99), np.random.seed(99), random.seed(99)
        _ = [_draws() for _ in range(4)]

        # Negative case first, from the same displaced position: without the restore the next draw does not
        # continue the saved sequence. Taken before the load so it reads the state the load is about to replace.
        without_restore = _draws()
        defect_shows = any(abs(a - b) > 1e-12 for a, b in zip(expected, without_restore))

        torch.manual_seed(99), np.random.seed(99), random.seed(99)
        _ = [_draws() for _ in range(4)]

        m2, opt2 = _Tiny(), None
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-4)
        step = load_checkpoint(d, m2, opt2)
        got = _draws()

        ok = step == 3 and all(abs(a - b) < 1e-12 for a, b in zip(expected, got))
        return check("checkpoint restores every random stream", ok and defect_shows,
                     "diverges without the restore: %s" % defect_shows)


def test_checkpoint_round_trip_on_the_real_device():
    """Repeat the round trip with the model on whatever device the run would use.

    The test above puts everything on the host, which is the one arrangement where the restore cannot go wrong.
    A resume on a machine with an accelerator loads the saved random state onto that device, where the numpy keys
    cannot be converted, and the failure names numpy rather than the device. This is the same shape as every
    other defect in this suite's history: a check that only ever saw the branch that works.
    """
    if not torch.cuda.is_available():
        return check("checkpoint round trip on the run's device", True, "SKIP, no accelerator on this host")

    dev = torch.device("cuda")
    with tempfile.TemporaryDirectory() as d:
        m = _Tiny().to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)

        torch.manual_seed(23), np.random.seed(23), random.seed(23)
        _ = _draws()
        expected = _draws()

        torch.manual_seed(23), np.random.seed(23), random.seed(23)
        _ = _draws()
        save_checkpoint(d, 5, m, opt)

        torch.manual_seed(77), np.random.seed(77), random.seed(77)
        _ = [_draws() for _ in range(3)]

        m2 = _Tiny().to(dev)
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-4)
        step = load_checkpoint(d, m2, opt2)
        got = _draws()

        on_device = all(p.device.type == "cuda" for p in m2.parameters())
        ok = step == 5 and on_device and all(abs(a - b) < 1e-12 for a, b in zip(expected, got))
        return check("checkpoint round trip on the run's device", ok,
                     "parameters stayed on the accelerator: %s" % on_device)


def test_checkpoint_tolerates_missing_random_state():
    """Files written before random state existed must still load, and must say so."""
    with tempfile.TemporaryDirectory() as d:
        m = _Tiny()
        opt = torch.optim.Adam(m.parameters(), lr=1e-4)
        path = save_checkpoint(d, 7, m, opt)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload.pop("rng_state")
        torch.save(payload, path)

        m2 = _Tiny()
        opt2 = torch.optim.Adam(m2.parameters(), lr=1e-4)
        step = load_checkpoint(d, m2, opt2)
        return check("checkpoint without random state still loads", step == 7)


def main():
    print("invariants and gate negative tests")
    print("-" * 62)
    ok = True
    ok &= test_deterministic_branch_matches_constant_step()
    ok &= test_zero_scale_contributes_nothing()
    ok &= test_stored_and_recomputed_logprob_agree()
    ok &= test_evaluation_sampling_is_deterministic()
    ok &= test_completeness_gate()
    ok &= test_completeness_gate_ignores_duplicates()
    ok &= test_completeness_gate_refuses_mixed_cycles()
    ok &= test_import_closure_gate_catches_a_missing_module()
    ok &= test_import_closure_gate_counts_only_tracked_files()
    ok &= test_evaluation_arguments_are_refused_in_both_directions()
    ok &= test_adapter_arguments_are_refused_where_they_do_nothing()
    ok &= test_a_hub_identifier_is_not_mistaken_for_a_path()
    ok &= test_every_shared_check_is_satisfiable_by_both_entry_points()
    ok &= test_the_download_helper_refuses_an_output_inside_the_tree()
    ok &= test_resume_that_would_run_nothing_is_refused()
    ok &= test_half_configured_tripwire_is_refused_in_both_directions()
    ok &= test_line_ending_gate_sees_files_without_an_extension()
    ok &= test_checkpoint_restores_random_state()
    ok &= test_checkpoint_round_trip_on_the_real_device()
    ok &= test_checkpoint_tolerates_missing_random_state()
    print()
    print("INVARIANTS_%s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
