#!/bin/bash
# Copyright 2026 Mehmet Turan Yardimci
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
#
# Environment wrapper for a training run.
#
# The variables set here are the ones that make a headless machine behave, and getting any of them wrong
# produces a failure far from its cause: a renderer that silently falls back to software, an allocator that
# fragments until a long run dies, or a hub lookup that hangs on a machine without a network.
#
# Everything experiment shaped is passed through to the trainer rather than set here, so this file never becomes
# a second place where configuration lives.
#
# Usage: bash scripts/launch.sh --tasks ... --base_policy ... [any trainer flag]
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Hardware accelerated offscreen rendering. Without this the simulator may fall back to a software renderer,
# which does not fail, it just runs slowly enough to make a long run infeasible.
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
# The simulator asserts at import that this names a device inside CUDA_VISIBLE_DEVICES, before any GL context
# exists, so a hardcoded zero fails outright on a machine driving any card but the first. Taken from
# CUDA_VISIBLE_DEVICES when that is set.
_egl_device=0
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && _egl_device=${CUDA_VISIBLE_DEVICES%%,*}
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-$_egl_device}

# Long runs allocate and free tensors of changing shapes. Expandable segments keep the allocator from
# fragmenting into a state where a request fails despite enough free memory in total.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Deliberately NOT offline by default. Offline is the right setting for a long run on a machine whose caches are
# already warm and the wrong setting for a first run: if the installed simulator distribution does not carry its
# assets in the package, the first environment it builds fetches them from the hub, and with the offline flag set
# that download fails, the failure is swallowed, and construction continues against an asset directory that does
# not exist. What surfaces later is an error from inside the renderer that names neither this file nor the flag.
# Which behaviour the declared distribution has was not established here; see the first run note in README.
#
# So set them yourself once the caches are warm, which this wrapper will honour:
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 bash scripts/launch.sh ...
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}

# Some distributions ship only python3. Overridable for the same reason the gate runner is.
PY="${TRAIN_PY:-}"
[ -n "$PY" ] || { command -v python >/dev/null 2>&1 && PY=python || PY=python3; }

exec "$PY" "$ROOT/src/train_flow_rl.py" "$@"
