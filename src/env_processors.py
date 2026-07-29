# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Observation and action transforms, taken from the policy library rather than reimplemented.

The transforms that resize camera images, order their channels and rescale the action are part of how the
pretrained policy was trained. Writing a second version of them here would be an invitation to drift: the two
would agree at first and then not, and the symptom would be a policy that performs worse than it should for
reasons that look like the fine tuning rather than like the pipeline.

So they are fetched from the library, parameterised by the suite and the policy configuration, and used
unchanged by both training and evaluation. That is the property the preprocessing parity rule in the protocol
requires: the same raw frame must produce the same tensor everywhere it is consumed.
"""

from __future__ import annotations

from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors


def make_processors(suite_name: str, policy_config):
    """Return the observation preprocessor and the action postprocessor for one suite.

    Both are returned together and are meant to be used together. Taking one from here and writing the other by
    hand reintroduces exactly the mismatch this module exists to avoid.
    """
    env_cfg = LiberoEnvConfig(task=suite_name)
    return make_env_pre_post_processors(env_cfg, policy_config)
