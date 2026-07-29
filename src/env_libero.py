# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Vectorised LIBERO environments, one simulator process per task.

Each environment runs in its own subprocess. The simulator and its rendering backend keep global state, so
several tasks constructed inside one process can interfere with each other in ways that look like a property of
the policy rather than of the harness. Process isolation removes that class of problem at the cost of some
startup time, which is a good trade for anything whose numbers will be reported.

The reward is sparse and terminal: an episode is worth something only if the task was solved. No shaping term is
computed anywhere in this repository. Shaping changes what is being optimised, and a shaped run is not
comparable to an unshaped one, so making it available as a flag would invite exactly that comparison.

The wrapper below exists for two defects in the layer underneath, both of which are silent.

The environment overwrites the episode termination signal with its own success check, so reaching the time limit
arrives as neither a termination nor a truncation. The vector wrapper never sees an episode boundary, so an
environment that never succeeds runs forever, and every later reset in that slot starts from a stale state. The
cap is therefore enforced inside the subprocess.

Attributes do not cross the wrapper boundary by themselves in either direction: a plain wrapper forwards neither
reads nor writes, so the initial state selector set on a wrapper would sit there while the environment underneath
kept its own default, and every worker would silently share one initial state. The vector environment's
`set_attr` reaches the inner environment through gymnasium's own stack walk, so on the versions tested that path
already works; the explicit forwarding here is a second route rather than a repair, and it is kept because the
consequence of the boundary being crossed silently is an evaluation that is not reproducible and does not say so.
"""

from __future__ import annotations

import functools

import gymnasium as gym
import numpy as np
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, _make_env_fns
from lerobot.envs.utils import preprocess_observation
from libero.libero import benchmark as libero_benchmark

CAMERAS = ["agentview_image", "robot0_eye_in_hand_image"]


class EpisodeCapWrapper(gym.Wrapper):
    """Enforce the time limit in the worker and forward the control attribute writes the base class drops."""

    _FORWARD_ATTRS = ("init_state_id", "_reset_stride")

    def __init__(self, env, horizon):
        super().__init__(env)
        self._horizon = int(horizon)
        self._t = 0

    def __setattr__(self, name, value):
        if name in EpisodeCapWrapper._FORWARD_ATTRS and "env" in self.__dict__:
            setattr(self.env, name, value)
        else:
            super().__setattr__(name, value)

    def reset(self, **kw):
        out = self.env.reset(**kw)
        self._t = 0
        return out

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._t += 1
        if self._t >= self._horizon:
            trunc = True
        return obs, rew, term, trunc, info


def make_wrapped_env(suite_name, task_id, episode_length):
    """Build one environment. Defined at module level so it can be sent to a spawned subprocess."""
    suite = libero_benchmark.get_benchmark_dict()[suite_name]()
    cfg = LiberoEnvConfig(task=suite_name)
    fns = _make_env_fns(suite=suite, suite_name=suite_name, task_id=task_id, n_envs=1,
                        camera_names=CAMERAS, episode_length=episode_length,
                        init_states=True, gym_kwargs=cfg.gym_kwargs, control_mode="relative")
    return EpisodeCapWrapper(fns[0](), horizon=episode_length)


def suite_step_limit(suite_name):
    """The step limit the task suite itself defines, independent of what a run asks for.

    Read from the library's own table rather than from a constructed environment. An environment built with an
    explicit episode length reports that length back as its limit, so asking the environment is asking the run
    what the run already decided; the answer is always yes and tells nobody anything.
    """
    return int(TASK_SUITE_MAX_STEPS.get(suite_name, 0)) or None


def build_env(tasks, episode_length, seed):
    """Create one subprocess per task.

    `tasks` is a sequence of suite name and task index pairs. Returns the vector environment, the task index of
    each slot, the first observation, and the step limit the task suite defines for the first task's suite.
    """
    fns = [functools.partial(make_wrapped_env, sn, tid, episode_length) for sn, tid in tasks]
    ve = gym.vector.AsyncVectorEnv(fns, context="spawn", autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs, _ = ve.reset(seed=[seed + i for i in range(len(fns))])
    return ve, np.arange(len(tasks)), obs, suite_step_limit(tasks[0][0])


def set_initial_states(ve, mode, n_envs, per_task, eval_batch, n_init_states):
    """Choose which stored initial state each worker starts from.

    Training spreads the workers across distinct initial states and advances them, so successive collections do
    not keep starting from the same configuration. Evaluation pins every worker to the state determined by the
    batch index, which is what makes two evaluation runs comparable episode by episode.

    The number of available initial states is passed in rather than assumed, since it is a property of the task
    suite and not of this harness.
    """
    if mode == "train":
        ve.set_attr("init_state_id", list(range(n_envs)))
        ve.set_attr("_reset_stride", [n_envs] * n_envs)
    else:
        ids = [(eval_batch * per_task + (i % per_task)) % n_init_states for i in range(n_envs)]
        ve.set_attr("init_state_id", ids)
        ve.set_attr("_reset_stride", [0] * n_envs)


def get_task_prompts(ve):
    """Read each worker's language instruction across the process boundary."""
    return list(ve.call("task_description"))


def adapt_observation(raw_obs, env_pre, task_prompts):
    """Turn a raw vector observation into the dictionary the policy expects.

    The camera tensors do not exist under these names on the raw observation; they are produced by the
    preprocessing step, which must therefore run before anything reads them. Reading the processed names off the
    raw dictionary returns nothing at all, and a caller that does not check gets an empty result rather than an
    error.
    """
    o = preprocess_observation(raw_obs)
    o["task"] = list(task_prompts)
    o = env_pre(o)
    return {
        "main_images": o.get("observation.images.image"),
        "wrist_images": o.get("observation.images.image2"),
        "states": o.get("observation.state"),
        "task_descriptions": list(o.get("task")) if o.get("task") is not None else None,
    }


def terminal_success(term, info):
    """Sparse terminal reward: one when the task was solved on this step, zero otherwise.

    Kept as a named function so the reward definition has exactly one place in the repository. A success
    criterion spread across call sites is how two parts of a system end up disagreeing about what counts.
    """
    solved = np.asarray(term).reshape(-1).astype(bool)
    if isinstance(info, dict) and "is_success" in info:
        flag = np.asarray(info["is_success"]).reshape(-1).astype(bool)
        solved = solved | flag
    return solved.astype(np.float32)
