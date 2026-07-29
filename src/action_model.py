# Copyright 2026 Mehmet Turan Yardimci
# Portions Copyright 2025 The RLinf Authors.
# Portions Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Derived from RLinf (https://github.com/RLinf/RLinf), specifically
# rlinf/models/embodiment/openpi/openpi_action_model.py, and from LeRobot (https://github.com/huggingface/lerobot),
# specifically lerobot/policies/smolvla/modeling_smolvla.py, both licensed under the Apache License, Version 2.0.
# The prefix cache construction follows LeRobot's own policy implementation closely, so that the pretrained head is
# driven exactly as it was trained.
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b)): this file has been modified. Relative to RLinf it
# wraps a different policy family, and captures the features the critic reads through a forward pre hook so the
# pretrained action head is never edited. Relative to LeRobot the prefix cache construction is exposed so an
# external sampler can drive the integration, the output is cropped to the executed part of the chunk and the
# real degrees of freedom, and the policy's own postprocessing is applied to the action tensor rather than to the
# whole output dictionary.
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""A reinforcement learning wrapper around a pretrained flow matching policy, which never modifies it.

The whole design follows from one constraint: the pretrained action head must stay bit identical, so that a
deterministic rollout through this wrapper reproduces the policy's own behaviour exactly, which is what makes
the behavioural parity rule in the protocol checkable. Without it, a difference in behaviour cannot be attributed
to the update rather than to the wrapper.

Three consequences.

The head is driven, not edited. Sampling calls the policy's own denoising step through a callback, so the
sampler adds behaviour around the head rather than inside it.

The critic reads the head's features through a forward pre hook rather than a new output. A hook lives on the
module instance and is not part of the state dictionary or the forward computation, so registering one changes
neither the weights nor the arithmetic. The alternative, adding a branch to the head, would make the parity
claim false the moment it was written.

Normalisation statistics come from the saved processors, not from the weights. Loading the weights alone gives a
policy that runs and acts subtly wrong, which is a difficult failure to attribute later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import os

import torch
import torch.nn as nn
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import (OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS,
                                     POLICY_POSTPROCESSOR_DEFAULT_NAME,
                                     POLICY_PREPROCESSOR_DEFAULT_NAME)

try:
    from .flow_sde import sde_sample
    from .value_head import ValueHead, compute_value_from_suffix
except ImportError:
    from flow_sde import sde_sample
    from value_head import ValueHead, compute_value_from_suffix



# What the simulator harness in this repository hands the policy and expects back. A checkpoint that declares
# anything else cannot drive it, and the failure that follows names an image key or an action width deep inside a
# spawned worker rather than the checkpoint, so it is checked here instead.
REQUIRED_IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
REQUIRED_STATE_DIM = 8
REQUIRED_ACTION_DIM = 7


@dataclass
class SmolVLAConfig:
    model_path: str
    num_steps: int
    noise_level: float
    device: str = "cpu"
    noise_method: str = "flow_sde"
    add_value_head: bool = True
    detach_critic_input: bool = True
    value_head_layernorm: bool = False
    value_head_popart: bool = False


def _is_local_path(s: str) -> bool:
    """Whether a string can only mean a place on this filesystem, never a hub repository identifier.

    Hub identifiers are one slash between two names, or a single bare name. Anything anchored to a root or a home
    directory, anything carrying a backslash, and anything nested more deeply than one slash is a path. Deciding
    it this way rather than by looking for a separator keeps the guard from rejecting every real repository id,
    which is what it did when it was first written.
    """
    return (s.startswith((".", "/", "~")) or "\\" in s or s.count("/") > 1
            or (s.count("/") == 1 and any(part == "" for part in s.split("/"))))


class SmolVLAForRLActionPrediction(nn.Module):
    """Wraps a pretrained policy with the pieces a policy gradient needs, and nothing else."""

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        # A path that does not exist is otherwise handed on to the hub, which reads it as a repository identifier
        # and answers that a repository identifier may not look like that, telling a reader who passed a path to
        # pass a repository identifier. Only strings that cannot be repository identifiers are caught here: a hub
        # id is one slash between two names, so anything anchored, backslashed or more deeply nested is a path,
        # and anything else is left to the loader, which is the only thing that can tell a typo from a repository
        # that exists.
        model_path = str(config.model_path)
        if _is_local_path(model_path) and not os.path.isdir(model_path):
            raise SystemExit(
                "--base_policy %r is a path and no directory is there. Point it at a downloaded policy directory, "
                "which scripts/download_base_policy.py writes, or pass a hub repository identifier such as "
                "HuggingFaceVLA/smolvla_libero." % model_path)

        self.policy = SmolVLAPolicy.from_pretrained(config.model_path)
        self.policy.eval()
        # A shorthand for a module that is already registered under the policy, bound outside the module registry
        # on purpose. Assigning it normally registers the same submodule a second time: parameter iteration
        # deduplicates by identity and is unaffected, but the state dictionary does not, so every checkpoint
        # would carry the whole policy twice, under two prefixes, at roughly double the size and with nothing
        # reporting it.
        object.__setattr__(self, "flow_model", self.policy.model)
        self.lerobot_cfg = self.policy.config
        # Read off the policy rather than configured, for the reason the action width is: a value written down
        # here is right for the checkpoint it was written against and silently wrong for the next one, and this
        # one decides how much of the emitted chunk is graded.
        self.action_chunk = int(self.lerobot_cfg.chunk_size)
        self.lerobot_cfg.num_steps = config.num_steps

        self._check_checkpoint_fits_the_harness()
        # Read off the policy rather than configured. The width the environment accepts decides both the crop
        # applied to the emitted chunk and the surface the likelihood is summed over, so a wrong value is wrong
        # twice over.
        self.action_env_dim = int(self.lerobot_cfg.action_feature.shape[0])

        # The two pipelines take different things and must be built differently. The preprocessor is handed a
        # dictionary of observations and returns one, so the library's default converters are right for it. The
        # postprocessor is handed the action tensor itself, and the loader does not restore converters from the
        # saved configuration: it falls back to the dictionary ones, which reject a tensor outright. Without the
        # action converters the first decision of the first cycle raises, for every regime, task and seed.
        #
        # The device is overridden rather than taken from the saved configuration. A checkpoint records the
        # device it was written on, so a checkpoint written on a machine with a card refuses to construct on one
        # without, raising about a processor step while the real cause is the absent card. Every other part of
        # this repository has a working processor path, and this is what made it unreachable.
        device_override = {"device_processor": {"device": str(config.device)}}
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            config.model_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            overrides=device_override)
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            config.model_path, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition, to_output=transition_to_policy_action,
            overrides=device_override)

        # Capture the features entering the head's final projection. Read only, and cleared before each call so
        # a stale capture from a previous step can never be read as if it belonged to this one.
        self._suffix = {}

        def _capture(_module, args):
            self._suffix["out"] = args[0]

        self._suffix_handle = self.flow_model.action_out_proj.register_forward_pre_hook(_capture)

        self.value_head = None
        if config.add_value_head:
            # The width is read off the layer whose input the hook captures, rather than written down here. A
            # constant would be right for one pretrained policy and produce a shape error deep inside the critic
            # for any other, on the first forward, with nothing in the message naming the base policy.
            expert_feature_dim = self.flow_model.action_out_proj.in_features
            self.value_head = ValueHead(input_dim=expert_feature_dim,
                                        layernorm=config.value_head_layernorm,
                                        bias_last=config.value_head_popart)
            self.value_head.popart = bool(config.value_head_popart)

    def _check_checkpoint_fits_the_harness(self) -> None:
        """Refuse a checkpoint whose declared features this harness cannot drive.

        Checked here because the alternative is a failure far from its cause. A checkpoint expecting different
        camera names raises inside the policy's image preparation, naming keys the caller never wrote; one with a
        different action width gets past every stage in this process and fails as an assertion inside a spawned
        simulator worker. Both read as bugs in this repository rather than as the wrong checkpoint.
        """
        declared = set(self.lerobot_cfg.input_features or {})
        missing = [k for k in REQUIRED_IMAGE_KEYS if k not in declared]
        state = self.lerobot_cfg.input_features.get("observation.state")
        state_dim = int(state.shape[0]) if state is not None else None
        action_dim = int(self.lerobot_cfg.action_feature.shape[0])

        problems = []
        if missing:
            problems.append("expects image keys %s but declares %s"
                            % (", ".join(missing), ", ".join(sorted(k for k in declared if "images" in k))))
        if state_dim != REQUIRED_STATE_DIM:
            problems.append("expects a %d dimensional state, checkpoint declares %s"
                            % (REQUIRED_STATE_DIM, state_dim))
        if action_dim != REQUIRED_ACTION_DIM:
            problems.append("expects %d dimensional actions, checkpoint declares %d"
                            % (REQUIRED_ACTION_DIM, action_dim))
        if problems:
            raise SystemExit(
                "the checkpoint at %s does not fit this harness: %s. A policy conditioned on this task suite is "
                "required, not a generic base checkpoint; see the README." % (self.config.model_path,
                                                                              "; ".join(problems)))

    # observation plumbing -------------------------------------------------------------------------------

    def obs_processor(self, env_obs: dict) -> dict:
        """Rename the harness's keys to the ones the policy library expects."""
        return {
            "observation.images.image": env_obs["main_images"],
            "observation.images.image2": env_obs["wrist_images"],
            "observation.state": env_obs["states"],
            "task": env_obs["task_descriptions"],
        }

    def prep_inputs(self, env_obs: dict):
        """Produce the tensors the policy consumes, using the library's own preparation helpers."""
        processed = self.preprocessor(self.obs_processor(env_obs))
        images, img_masks = self.policy.prepare_images(processed)
        state = self.policy.prepare_state(processed)
        return (images, img_masks, processed[OBS_LANGUAGE_TOKENS],
                processed[OBS_LANGUAGE_ATTENTION_MASK], state, processed)

    # driving the pretrained head ------------------------------------------------------------------------

    def build_prefix_cache(self, images, img_masks, lang_tokens, lang_masks, state):
        """Run the observation through the backbone once and keep the cache for the whole integration.

        The backbone does not change across integration steps, so computing it once is what makes many step
        sampling affordable at all on a single card.
        """
        m = self.flow_model
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_out, past_key_values = m.vlm_with_expert.forward(
            attention_mask=att_2d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=m.config.use_cache, fill_kv_cache=True)
        return prefix_pad_masks, past_key_values, prefix_out

    def make_velocity_fn(self, prefix_pad_masks, past_key_values):
        """A callback that evaluates the policy's own denoising step and returns the captured features."""
        m = self.flow_model
        suffix = self._suffix

        def velocity_fn(x_t, t):
            suffix.pop("out", None)
            v_t = m.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values,
                                 x_t=x_t, timestep=t)
            return v_t, suffix.get("out")

        return velocity_fn

    def make_value_fn(self):
        if self.value_head is None:
            return None
        head, detach = self.value_head, self.config.detach_critic_input

        def value_fn(suffix_out):
            return compute_value_from_suffix(head, suffix_out, detach_critic_input=detach)

        return value_fn

    # sampling -------------------------------------------------------------------------------------------

    def sample_actions(self, env_obs: dict, mode: Literal["train", "eval"] = "eval",
                       noise: Optional[torch.Tensor] = None) -> dict:
        """Draw an action chunk, returning the trajectory and log probability the update will need."""
        images, img_masks, lang_tokens, lang_masks, state, _ = self.prep_inputs(env_obs)
        if noise is None:
            noise = self.flow_model.sample_noise(
                (state.shape[0], self.lerobot_cfg.chunk_size, self.lerobot_cfg.max_action_dim), state.device)
        prefix_pad_masks, past_key_values, _prefix_out = self.build_prefix_cache(
            images, img_masks, lang_tokens, lang_masks, state)
        return sde_sample(
            velocity_fn=self.make_velocity_fn(prefix_pad_masks, past_key_values),
            noise=noise, num_steps=self.lerobot_cfg.num_steps,
            noise_method=self.config.noise_method, noise_level=self.config.noise_level,
            action_chunk=self.action_chunk, action_env_dim=self.action_env_dim, mode=mode,
            value_fn=self.make_value_fn())

    def output_transform(self, outputs: dict) -> dict:
        """Crop the padded chunk to what the environment accepts, then undo the action normalisation.

        The order is not free. The head emits a chunk padded out to the model's maximum action width, while the
        normalisation statistics are stored at the environment's real width, so unnormalising first multiplies a
        wide tensor by a narrow one and raises. The policy library does the same two steps in the same order, and
        this follows it.

        The postprocessor also takes the action tensor itself rather than the whole dictionary, and it has to be
        built with the converters that accept one; that happens where it is constructed.
        """
        actions = outputs["actions"][:, :self.action_chunk, :self.action_env_dim]
        outputs["actions"] = self.postprocessor(actions)
        return outputs
