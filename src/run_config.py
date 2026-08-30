#!/usr/bin/env python3
# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""run_config.json: every run states what it was, before it produces a number.

The gap this closes is easy to leave open. Two runs get compared, someone asks which variable differs, and
the answer has to be reconstructed from shell history and memory. If neither run wrote down its own
configuration, the comparison cannot be checked even in principle, and a rule that cannot be executed is not
a rule.

WHAT GOES IN, and the third block is the one that matters most.

  1. ARGUMENTS. Every argparse value, verbatim. No filtering, because the field that turns out to matter is
     always the one someone decided was uninteresting.
  2. DERIVED ARITHMETIC. The quantities a reader would otherwise compute in their head: pooled timesteps,
     buffer samples, expected optimizer steps. These are CLAIMS and are labelled as such.
  3. MEASURED ARITHMETIC. Filled in after the first outer FROM A LIVE COUNTER, never computed. Block 2 is
     what was intended; block 3 is what happened. When they disagree the run is not what its own banner says
     it is, and that disagreement is the thing worth catching: gradient accumulation, batch size and
     optimizer-step counts are all quantities a training script will happily print from configuration while
     doing something else.
  4. PROVENANCE. Repository revision, library versions, host, time.

  Plus MODULE CONSTANTS and ENVIRONMENT FLAGS, because argparse is not the whole configuration. A reward
  coefficient, a gradient-clip norm or an evaluation horizon defined as a module constant decides what a run
  IS and would appear in none of the blocks above. So does an environment variable that switches a value
  head's normalisation on. Both are recorded here for the same reason: the field that matters is the one
  nobody thought to record.
"""
import io
import json
import os
import subprocess
import sys
import time

# Environment variables that change what a run is. Extend this rather than assuming a flag is harmless.
TRACKED_ENV = (
    "FORCE_DETERMINISM", "STOP_ON_NONFINITE", "ROLLOUT_TRAIN_MODE", "ACCUM_OFF",
    "EXPERT_TRAIN", "VLM_LORA_RANK", "EXPERT_LORA_RANK", "CUBLAS_WORKSPACE_CONFIG",
)


def _git_sha(path):
    try:
        out = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unavailable"


def module_constants(module_name):
    """Every upper-case scalar in a constants module, read at run time rather than listed by hand.

    Listing them by hand is how one gets left out, and the one left out is the one that mattered.
    """
    if not module_name:
        return {}
    try:
        mod = __import__(module_name)
    except Exception as e:
        return {"ERROR": "%s: %s" % (type(e).__name__, e)}
    out = {}
    for k in dir(mod):
        if not k.isupper():
            continue
        v = getattr(mod, k)
        if isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
    return out


def dump_run_config(args, output_dir, derived=None, extra=None,
                    constants_module=None, repo_paths=None):
    """Write run_config.json into output_dir. Returns the path, or raises."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_config.json")

    try:
        import torch
        versions = {"torch": torch.__version__, "cuda": str(torch.version.cuda),
                    "python": sys.version.split()[0],
                    "arch_list": ",".join(torch.cuda.get_arch_list()) if torch.cuda.is_available() else "cpu"}
    except Exception:
        versions = {"python": sys.version.split()[0]}

    repos = repo_paths or {"repo": os.path.dirname(os.path.abspath(__file__))}

    doc = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                 for k, v in sorted(vars(args).items())},
        "derived_CLAIMED": derived or {},
        "measured_OBSERVED": {"note": "filled after the first outer from the live counter; empty means the "
                                      "run did not reach one optimizer step"},
        "provenance": {
            "git": {name: _git_sha(p) for name, p in repos.items()},
            "host": os.environ.get("HOSTNAME", os.uname().nodename if hasattr(os, "uname") else "?"),
            "versions": versions,
        },
        "module_constants": module_constants(constants_module),
        "env_flags": {k: os.environ.get(k, "") for k in TRACKED_ENV},
    }
    if extra:
        doc.update(extra)
    io.open(path, "w", encoding="utf-8").write(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    return path


def record_measured(output_dir, measured):
    """Fill block 3 after the first outer. A separate call because it is MEASURED, not predicted."""
    path = os.path.join(output_dir, "run_config.json")
    if not os.path.exists(path):
        return None
    doc = json.load(io.open(path, encoding="utf-8"))
    doc["measured_OBSERVED"] = measured
    claimed = doc.get("derived_CLAIMED", {})
    disagree = {k: {"claimed": claimed[k], "measured": measured[k]}
                for k in claimed if k in measured and claimed[k] != measured[k]}
    doc["claim_vs_measurement_DISAGREEMENTS"] = disagree
    io.open(path, "w", encoding="utf-8").write(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    return disagree


def assert_written(output_dir):
    """The GATE. Training does not start without this file.

    Plant the violation to check the gate: skip the write and confirm this refuses. A gate that has never
    been seen to refuse is not a gate.
    """
    path = os.path.join(output_dir, "run_config.json")
    if not os.path.exists(path):
        raise SystemExit(
            "RUN_CONFIG GATE FAILED: %s was not written. Training does not start without it, because a run "
            "that cannot state what it was cannot be compared with anything later." % path)
    return path
