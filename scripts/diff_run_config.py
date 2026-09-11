#!/usr/bin/env python3
# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Two runs are not a comparison until their configurations have been diffed field by field.

Any sentence that attributes a difference between two runs to one variable should be preceded by this diff.
Without it the honest language is relational: "these two runs differ by this much", not "X caused this".
This is the tool that makes that discipline executable rather than aspirational.

Exit codes: 0 identical or only non-critical fields differ, 1 a CRITICAL field differs, 2 usage,
read error, or SCHEMA MISMATCH (zero fields compared, which is blindness rather than agreement). A critical difference is not an error to suppress; it is the thing worth surfacing before a claim.

Usage:
    python diff_run_config.py A/run_config.json B/run_config.json
"""
import io
import json
import sys

# Fields where a difference invalidates a single-variable claim. Everything else is reported without
# raising the exit code, because a run directory or a timestamp differing proves nothing about the science.
CRITICAL = {
    "lr", "value_lr", "kl_coef", "ent_coef", "ppo_epochs", "accum_rounds", "n_rollout_steps",
    "num_steps", "n_action_steps", "per_task", "ensemble_n", "sigma_mode", "fixed_sigma",
    "episode_length", "eval_horizon", "k_outer", "tasks", "bc_ckpt", "seed",
}
CRITICAL_ENV = {"FORCE_DETERMINISM", "STOP_ON_NONFINITE", "ROLLOUT_TRAIN_MODE", "ACCUM_OFF",
                "EXPERT_TRAIN", "VLM_LORA_RANK", "EXPERT_LORA_RANK"}
# Constants that live in a module rather than in argparse still decide what a run is. A reward coefficient
# or an evaluation horizon defined here and not compared is a difference nobody would see.
CRITICAL_CONST = {"REWARD_COEF", "HORIZON", "GRAD_CLIP", "CLIP", "VCOEF", "LR", "SEED",
                  "NOISE_METHOD", "ACTION_CHUNK", "N_ENVS"}


def load(p):
    return json.load(io.open(p, encoding="utf-8"))


def cmp_block(a, b, name, critical):
    """Returns (rows, critical_count, compared_field_count).

    THE FIELD COUNT IS RETURNED because a verdict without its denominator is not a measurement.
    See the schema refusal in main(): a comparison of zero fields used to print "identical on
    every compared field", which is vacuously true and reads as verification.
    """
    rows, bad = [], 0
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
        if va == vb:
            continue
        mark = "CRITICAL" if k in critical else "note"
        if k in critical:
            bad += 1
        rows.append("  %-8s %-28s %-28s %s" % (mark, "%s.%s" % (name, k), str(va)[:28], str(vb)[:28]))
    return rows, bad, len(set(a) | set(b))


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    try:
        A, B = load(sys.argv[1]), load(sys.argv[2])
    except Exception as e:
        print("READ FAILED: %s: %s" % (type(e).__name__, e))
        print("A missing run_config is not 'no differences'. It means the comparison cannot be made.")
        return 2

    BLOCKS = ("args", "derived_CLAIMED", "measured_OBSERVED", "module_constants", "env_flags")
    rows, bad, compared = [], 0, 0
    for blk, crit in zip(BLOCKS, (CRITICAL, set(),
                                  {"optimizer_steps_per_outer", "samples_per_step"},
                                  CRITICAL_CONST, CRITICAL_ENV)):
        r, b, n = cmp_block(A.get(blk, {}), B.get(blk, {}), blk, crit)
        rows += r
        bad += b
        compared += n

    # ZERO COMPARED FIELDS IS BLINDNESS, NOT AGREEMENT. This tool reads the blocks named above. A
    # run_config written against a different schema carries none of them, so every block compared
    # the empty dict against the empty dict, `rows` stayed empty, and the tool printed "identical
    # on every compared field" followed by "a single-variable claim is available for the fields
    # checked". Measured seeded violation: two configs differing in task, arm and seed were
    # green-lit.
    #
    # The qualifier "for the fields checked" is what made it dangerous. Over an empty checked set
    # that sentence is vacuously TRUE and reads as verification. That is an empty-set failure
    # sitting inside the one tool whose whole purpose is to refuse unverified comparability. A
    # tool that cannot see the fields must say it is BLIND rather than say they agree.
    if compared == 0:
        print("DIFF %s against %s" % (sys.argv[1], sys.argv[2]))
        print("  SCHEMA MISMATCH: 0 fields compared. This is NOT a verification.")
        print("  expected blocks: %s" % ", ".join(BLOCKS))
        # The parenthesised fallbacks are built BEFORE the formatting, not after it with `or`.
        # Written as `"...%s" % ", ".join(sorted(A)) or "(none)"` the `or` binds to the whole
        # formatted string, which is always truthy, so the fallback can never appear and an empty
        # file prints a bare colon -- the same ambiguity this refusal exists to remove.
        print("  A top-level keys: %s" % (", ".join(sorted(A)) or "(none)"))
        print("  B top-level keys: %s" % (", ".join(sorted(B)) or "(none)"))
        print("A comparison of zero fields agrees on everything and measures nothing. Compare the")
        print("fields these files actually carry, or write the comparison for this schema.")
        return 2

    print("DIFF %s against %s" % (sys.argv[1], sys.argv[2]))
    if not rows:
        print("  identical on every compared field")
    else:
        print("  %-8s %-28s %-28s %s" % ("kind", "field", "A", "B"))
        for x in rows:
            print(x)
    print()
    if bad:
        print("VERDICT: %d CRITICAL field(s) differ. A single-variable claim between these two runs is NOT"
              % bad)
        print("available. Either equalise and repeat, or state the difference and keep the language")
        print("relational. Note that a field can be critical by name and inert in context, for example a")
        print("fixed-sigma value in a run that does not read it; the gate stays strict and the explanation")
        print("belongs in the report, not in a silent exception here.")
        return 1
    print("VERDICT: no critical field differs among %d compared fields. A single-variable claim is"
          % compared)
    print("available FOR THOSE FIELDS. The denominator is printed because a verdict without one is")
    print("not a measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
