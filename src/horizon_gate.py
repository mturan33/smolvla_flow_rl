#!/usr/bin/env python3
# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Single horizon gate. Enforced rather than described.

THE RULE. Training, the in-training evaluations and the run-end canonical evaluation all run at ONE horizon,
and that horizon comes from ONE field. The reference uses 480 for both training and evaluation
(`libero_10_ppo_openpi.yaml:91` and `:98`). If the paths that read a horizon disagree, the run does not
start.

WHY IT IS A GATE AND NOT A CONVENTION. An arrangement with two knobs, `--episode_length` and
`--eval_horizon`, where the evaluation cap is `min(max_steps, eval_horizon)`, is enough for ONE run to carry
TWO horizons: intermediate points measured at one value and the run-end canonical at another, because the
canonical is usually a separate invocation with its own flags. A series measured at two horizons is not a
series, and nothing in the code says so while it happens.

The failure is not hypothetical and it is not symmetric. Pulling the training horizon down to the reference's
480 through `--episode_length` alone silently drops the in-training evaluations with it while a canonical
launched separately stays where it was. The number that comes out looks like a result.

A horizon difference is also not recoverable by comparison later: a longer horizon gives every episode more
steps in which to succeed, so a difference in success rate between two horizons contains that difference as
well as whatever the policy did. If episode outcomes record the step at which they succeeded, a shorter
horizon can be recovered from a longer one by exact recount, but that is a repair, not a substitute for
measuring one series on one ruler.

Usage:
    from horizon_gate import assert_single_horizon
    assert_single_horizon(args, horizon=HORIZON)     # raises SystemExit if the paths disagree

    python horizon_gate.py --selftest
"""
import sys

DEFAULT_HORIZON = 480


def check(episode_length, eval_horizon, horizon):
    """Returns (ok, message). Pure, so the selftest can plant values without an argparse namespace."""
    vals = {"--episode_length": int(episode_length), "--eval_horizon": int(eval_horizon),
            "HORIZON": int(horizon)}
    distinct = sorted(set(vals.values()))
    if len(distinct) == 1:
        return True, "SINGLE HORIZON: every path reads %d." % distinct[0]
    detail = ", ".join("%s=%d" % (k, v) for k, v in vals.items())
    return False, ("HORIZON SPLIT: the paths disagree (%s). One horizon is required; a run whose training, "
                   "in-training evaluations and canonical do not share a ruler produces a series that is not "
                   "a series. Set them from the one field or change that field." % detail)


def assert_single_horizon(args, horizon=DEFAULT_HORIZON):
    ok, msg = check(getattr(args, "episode_length"), getattr(args, "eval_horizon"), horizon)
    print("  [horizon gate] %s" % msg, flush=True)
    if not ok:
        raise SystemExit(2)
    return True


def selftest():
    """PLANTED VIOLATIONS: each way the paths can drift apart, plus the cases that must pass."""
    cases = [
        ("all three agree at 480",                   (480, 480, 480), True),
        ("episode_length left at an older 600",      (600, 480, 480), False),
        ("eval_horizon left at an older 520",        (480, 520, 480), False),
        ("both flags agree but the field differs",   (520, 520, 480), False),
        ("all three agree at some other value",      (300, 300, 300), True),
    ]
    ok = True
    print("SINGLE HORIZON GATE, planted violations")
    for title, (el, eh, h), want in cases:
        got, msg = check(el, eh, h)
        good = got == want
        ok = ok and good
        print("   %-42s expected %-6s got %-6s  %s"
              % (title, "PASS" if want else "REFUSE", "PASS" if got else "REFUSE",
                 "OK" if good else "*** FAIL ***"))
        if not good:
            print("      message was: %s" % msg)
    print("")
    print("  SINGLE HORIZON SELFTEST %s"
          % ("PASSED" if ok else "*** FAILED, do not trust this gate ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
