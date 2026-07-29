# Copyright 2026 Mehmet Turan Yardimci
# Portions Copyright 2025 The RLinf Authors.
#
# Adapted from RLinf (https://github.com/RLinf/RLinf), licensed under the Apache License, Version 2.0, from two
# files: rlinf/models/embodiment/modules/value_head.py for the head itself, and the suffix pooling in
# rlinf/models/embodiment/openpi/openpi_action_model.py for the function that turns policy features into a value.
# NOTICE OF MODIFICATION (Apache License 2.0, Section 4(b)): this file has been modified. Optional adaptive target
# rescaling was added, so the head can learn on a normalised scale while every caller reads real units, and its
# running statistics are buffers so they travel in the state dictionary. Optional normalisation between the hidden
# layers was added, applied to the hidden layers only. The initialisation gain for the smooth rectifier
# activations is mapped to the rectifier entry, because the upstream passes the activation name straight to the
# gain table, which does not accept the head's own default. The upstream crop of the suffix to the action chunk is
# not repeated, because the policy library performs it before the projection whose input the critic reads.
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Critic head for a flow matching policy, kept outside the action head.

The head is a separate module attached alongside the policy rather than a branch inside it. Keeping it separate
is what allows the action head to stay bit identical to the pretrained one, which in turn is what makes the
behavioural parity rule in the protocol checkable.

Two options are worth explaining rather than leaving as flags.

Normalisation between the hidden layers is offered because a critic trained on sparse terminal rewards sees mostly
identical targets with a few far away, which is the regime where unnormalised hidden activations drift. Whether it
helps on a given setup is something a run has to show, and this file makes no claim about that; it is applied to
the hidden layers only, since normalising the scalar readout would remove the very scale the readout exists to
produce.

Adaptive rescaling lets the head learn against targets of roughly unit scale while the outside world keeps seeing
the original units. When the running statistics change, the final layer is reparameterised so its de normalised
output is unchanged, which means the update to the statistics does not by itself move any prediction.
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Small feed forward critic over pooled policy features."""

    def __init__(self, input_dim, hidden_sizes=(512, 128), output_dim=1,
                 activation="gelu", bias_last=False, layernorm=False):
        super().__init__()
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation.lower()]
        layers, in_dim = [], input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))
        self.mlp = nn.Sequential(*layers)

        # Running statistics for adaptive target rescaling. Inert until enabled by the trainer, in which case the
        # update lives in the optimisation step, next to the targets it tracks.
        #
        # These are buffers rather than plain floats so that they travel in the state dictionary. As attributes
        # they were absent from every checkpoint, and since the readout is reparameterised by the running scale,
        # a resumed run would reload a readout divided by a scale it no longer had, while the statistics silently
        # restarted at zero and one. Every de normalised value after that point is wrong by that factor, the
        # advantages built from it are wrong, and nothing reports an error.
        self.register_buffer("pa_on", torch.zeros(()))
        self.register_buffer("pa_mu", torch.zeros(()))
        self.register_buffer("pa_nu", torch.ones(()))
        self.register_buffer("pa_sigma", torch.ones(()))

        self._init_weights(activation.lower())

    @property
    def popart(self) -> bool:
        return bool(self.pa_on.item())

    @popart.setter
    def popart(self, value) -> None:
        with torch.no_grad():
            self.pa_on.fill_(1.0 if value else 0.0)

    def set_target_scale(self, mu: float, nu: float, sigma: float) -> None:
        """Write the running statistics in place, which is how a buffer is updated."""
        with torch.no_grad():
            self.pa_mu.fill_(float(mu))
            self.pa_nu.fill_(float(nu))
            self.pa_sigma.fill_(float(sigma))

    def _init_weights(self, nonlinearity="relu"):
        # The gain table does not cover every activation this head accepts; the smooth rectifier variants are
        # mapped to the rectifier entry, whose gain is close enough that the difference is not observable at this
        # depth. The readout is initialised small so early predictions start near zero rather than at whatever the
        # fan in happens to produce.
        gain_key = "relu" if nonlinearity in ("gelu", "silu") else nonlinearity
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if m is self.mlp[-1]:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=gain_key)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


def compute_value_from_suffix(value_head, suffix_out, detach_critic_input=True):
    """Pool policy features into a single state value.

    The critic input is detached by default so critic error does not flow back into the policy backbone. Letting
    it flow couples two objectives that are meant to be separable, and the coupling is hard to notice afterwards
    because both losses still go down.

    De normalisation happens here rather than at each call site, so everything downstream, the advantage
    estimator and any logging, sees real units without having to know whether rescaling is on.

    Everything captured is pooled, with no crop. The upstream this derives from crops the suffix to the action
    chunk before projecting it, but the policy library used here already does that crop itself, immediately
    before the projection whose input this function receives. A second crop would therefore be a no op in the
    ordinary case and a silent change of meaning in any other, which is worse than not having one. That the
    library crops was read out of its source rather than assumed.
    """
    s = suffix_out.mean(dim=1)
    if detach_critic_input:
        s = s.detach()
    v = value_head(s)[:, 0]
    if getattr(value_head, "popart", False):
        v = v * float(value_head.pa_sigma) + float(value_head.pa_mu)
    return v
