#!/bin/bash
# Copyright 2026 Mehmet Turan Yardimci
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
#
# Contact test: can this setup learn one task at all?
#
# Run this first. It establishes that the pieces are connected end to end, which is worth knowing before reading
# anything into what a longer run produces.
#
# What to look for, in order:
#   the manifest line, which says what is training. If the count is not what you intended, stop here.
#   the optimiser line, which says at what rate, read from inside the optimiser rather than from the arguments.
#   the first update's ratio, which must be one. Anything else means the stored and recomputed likelihoods live
#   on different surfaces, and nothing after that point is meaningful.
#   the reward, which should move. If it does not, read the environment line first: a cap below the suite's own
#   limit makes long tasks unsolvable, and the trainer says so explicitly when that is the case.
#
# Usage: bash examples/single_task_contact.sh /path/to/base/policy [output_dir]
set -eu
BASE=${1:?path to the pretrained policy is required}
OUT=${2:-runs/contact}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# One task, a short run.
#
# Every value below is ILLUSTRATIVE, the regime included. They are conventional textbook values, chosen to make
# the command readable, and they are not a tuned recipe and not a recommendation. The one exception is the episode
# length, which is the step limit this task suite itself publishes, so the contact test is not answered by a cap.
# Expect to change the rest: the useful setting depends on the policy, the task and the card, and the point of a
# contact test is to see whether anything moves at all, not to be tuned.
#
# On disk: a checkpoint holds the whole policy, not just the trainable surface, so each one is roughly 1.6 GB
# under this regime. Periodic checkpointing is off here for that reason, and the run still writes one at the end,
# so the first command anyone runs costs about 1.6 GB rather than a multiple of it. Pass --ckpt_every N if you
# want intermediate ones and have the room.
#
# On the clock, measured on one consumer card: a training cycle is about fifty seconds, so the ten of them are
# roughly eight minutes, and an evaluation episode at this horizon is about ten minutes. Evaluation therefore
# dominates whatever it is set to, which is why there is one of it here rather than several. Everything the list
# above tells you to read is on screen within the first few minutes; the evaluation at the end is a bonus, not
# the answer.
PY="${TRAIN_PY:-}"
[ -n "$PY" ] || { command -v python >/dev/null 2>&1 && PY=python || PY=python3; }

"$PY" "$ROOT/src/train_flow_rl.py" \
  --tasks libero_10:0 \
  --base_policy "$BASE" \
  --output_dir "$OUT" \
  --noise_level 0.1 \
  --actor_lr 1e-5 --critic_lr 1e-3 \
  --gamma 0.99 --gae_lambda 0.95 \
  --regime full \
  --updates 10 --rollout_steps 32 --action_steps 1 \
  --episode_length 520 --denoise_steps 10 \
  --eval_every 10 --eval_episodes 2 --eval_horizon 520 \
  --ckpt_every 0 --seed 0 \
  --tripwire --trip_ratio_median 2 --trip_ratio_max 10 --trip_success_floor 0 --trip_window 5
