# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Save and restore a run, including the random state.

Storing the model and the optimiser is not enough to resume a run faithfully. Sampling consumes the global random
streams on every step, so a resumed run that starts from fresh streams draws a different sequence than an
uninterrupted one would have drawn at the same point. Nothing errors, both runs look reasonable, and the resumed
one is simply no longer reproducible from its seed.

The random state is therefore stored too, and stored in plain tensors and numbers. The array based generator
state cannot be written directly, because the loader below refuses to unpickle arbitrary objects and would then
be unable to read the very files this function writes. Restricting the payload to tensors and primitives keeps
the safe loader usable, which is worth more than the convenience of storing the state object as it comes.

Checkpoints written before random state existed are still readable. Which case was taken is printed rather than
assumed, so a resume that silently restarted the streams cannot be mistaken for a faithful one.
"""

from __future__ import annotations

import glob
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch


def _capture_rng() -> dict:
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "numpy_keys": torch.from_numpy(np_state[1].astype(np.int64)),
        "numpy_meta": [str(np_state[0]), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _host(t):
    """A tensor on the host, whatever device it arrived on.

    Every consumer of a random state lives on the host: the generator setters take host byte tensors, and the
    numpy keys have to become a numpy array. A checkpoint loaded onto a device brings its random state with it,
    and converting a device tensor to numpy raises with a message about numpy rather than about the device.
    """
    return t.cpu() if hasattr(t, "cpu") else t


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(_host(state["torch"]))
    name, pos, has_gauss, cached = state["numpy_meta"]
    np.random.set_state((str(name), _host(state["numpy_keys"]).numpy().astype(np.uint32),
                         int(pos), int(has_gauss), float(cached)))
    py = state["python"]
    random.setstate(tuple(py) if isinstance(py, list) else py)
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_host(s) for s in state["cuda"]])


def save_checkpoint(ckpt_dir: str, step: int, model, optimizer, extra: Optional[dict] = None) -> str:
    """Write one checkpoint and a pointer to the newest one. Returns the path written."""
    os.makedirs(ckpt_dir, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _capture_rng(),
        "extra": extra or {},
    }
    path = os.path.join(ckpt_dir, "step_%06d.pt" % step)
    torch.save(payload, path)
    with open(os.path.join(ckpt_dir, "latest.txt"), "w") as fh:
        fh.write(os.path.basename(path))
    return path


def load_checkpoint(ckpt_dir: str, model, optimizer) -> int:
    """Resume from the newest readable checkpoint in a directory. Returns the step it was written at.

    Unreadable files are skipped rather than fatal. A crash during a write leaves a truncated file, and losing
    the whole run because the most recent of many checkpoints is damaged is a poor trade.

    The payload is read onto the host regardless of where the model lives. Loading it straight onto the device
    puts the saved random state there too, where it is unusable, and both loaders below move what they need
    themselves: a state dictionary copies into parameters that are already on the right device, and the optimiser
    places its state alongside the parameters it belongs to.
    """
    paths = glob.glob(os.path.join(ckpt_dir, "step_*.pt"))
    if not paths:
        # Every other exit from this function announces itself, and this one used to be the exception: a resume
        # against a mistyped directory started from zero in silence, which looks identical in the log to a resume
        # that worked. A run that believes it is continuing a long one deserves to be told that it is not.
        print("  [checkpoint] no checkpoint in %s, starting fresh" % ckpt_dir, flush=True)
        return 0
    paths.sort(key=lambda p: int(os.path.basename(p)[5:-3]), reverse=True)

    for path in paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            print("  [checkpoint] %s unreadable (%s), trying an older one"
                  % (os.path.basename(path), type(exc).__name__), flush=True)
            continue

        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        rng = payload.get("rng_state")
        if rng:
            _restore_rng(rng)
            note = "random state restored"
        else:
            note = "no random state in this checkpoint, streams restart"
        print("  [checkpoint] resumed from %s at step %d, %s"
              % (os.path.basename(path), int(payload["step"]), note), flush=True)
        return int(payload["step"])

    print("  [checkpoint] no readable checkpoint, starting fresh", flush=True)
    return 0


def load_policy_only(path: str, model) -> Tuple[int, int, dict]:
    """Load policy weights from one specific file for evaluation, and prove the load happened.

    A file path is required rather than a directory, so the evaluated weights are the ones named and not
    whichever happens to be newest. The number of tensors that changed is returned alongside the step, because a
    load that quietly matched nothing leaves the model at its initialisation while every log line still reports
    the checkpoint's name.

    Whatever the writer recorded about the run is returned as well, so that a caller can compare the arguments it
    was given against the ones the checkpoint was produced under. Reading it here rather than in a second pass
    keeps a large file from being loaded twice.

    Read onto the host, like the resume path, and copied into parameters that already live wherever the model was
    put.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in payload:
        raise KeyError("checkpoint %s has no model_state_dict" % path)

    before = {k: float(v.detach().float().sum()) for k, v in model.state_dict().items()}
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError("state dict mismatch: %d missing, %d unexpected" % (len(missing), len(unexpected)))
    after = {k: float(v.detach().float().sum()) for k, v in model.state_dict().items()}
    changed = sum(1 for k in before if before[k] != after.get(k))

    if changed == 0:
        raise RuntimeError("loading %s changed no tensor; refusing to evaluate an unloaded policy" % path)
    extra = payload.get("extra") or {}
    return int(payload.get("step", -1)), changed, dict(extra)
