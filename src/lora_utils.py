# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Low rank adapters on the language backbone, and a way to switch them off in place.

Adapters are injected into the existing modules rather than wrapping the model in a new object. The policy's own
forward pass indexes the backbone's layer list directly, and a wrapper hides that attribute, so an injection that
preserves the module tree is the only variant that leaves the policy working.

Everything is frozen before injection, so the only trainable parameters afterwards are the ones just added. That
ordering matters: freezing after injection would freeze the adapters too and produce a run that trains nothing
while reporting no error.

The assertions here are deliberate. Adapter target names are matched by substring, so a name that also occurs
inside the action head would silently make the head trainable and break any claim that the pretrained head is
untouched. Rather than trusting the target list, the head is checked afterwards, and the check also verifies that
the injection produced some trainable parameters at all, which catches a target list that matched nothing.

Switching adapters off in place gives a reference forward pass without holding a second copy of the model in
memory, which on a single consumer card is often the difference between fitting and not.
"""

from contextlib import contextmanager
from typing import List, Optional

import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Linear layers belonging to the pretrained action head. None of these may be adapted or left trainable.
ACTION_HEAD_LINEARS = (
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "state_proj",
)


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def apply_text_model_lora(
    flow_model: nn.Module,
    rank: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Freeze the policy, inject adapters into the language backbone attention projections, and verify.

    Returns the module the adapters were injected into, which is the same object that was passed in, since the
    injection is in place.
    """
    freeze_module(flow_model)
    text_model = flow_model.vlm_with_expert.vlm.model.text_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        lora_dropout=0.0,
        bias="none",
    )
    inject_adapter_in_model(cfg, text_model)

    for name in ACTION_HEAD_LINEARS:
        mod = getattr(flow_model, name)
        assert isinstance(mod, nn.Linear) and not isinstance(mod, LoraLayer), (
            f"action head layer '{name}' was adapted (type {type(mod).__name__}); "
            f"the target list reached the action head"
        )
        assert not mod.weight.requires_grad, f"action head layer '{name}' weight is trainable after injection"
        if mod.bias is not None:
            assert not mod.bias.requires_grad, f"action head layer '{name}' bias is trainable after injection"

    n_trainable = sum(p.numel() for p in flow_model.parameters() if p.requires_grad)
    assert n_trainable > 0, "injection produced no trainable parameters; the target list matched nothing"
    return text_model


def count_adapter_layers(model: nn.Module) -> int:
    """How many adapted layers exist under a module.

    Used to decide whether switching adapters off recovers anything. Zero means the policy was trained in place
    and there is no pretrained copy left inside the model to compare against.
    """
    return sum(1 for _, m in model.named_modules() if isinstance(m, LoraLayer))


@contextmanager
def disable_adapters(root: nn.Module):
    """Temporarily switch every adapter under `root` off, restoring the previous state on exit.

    The previous per layer state is saved and restored rather than being set to a fixed value, so nesting this
    context or entering it when adapters are already off does not leave the model in a different state than it
    started in.
    """
    flipped = []
    for _, m in root.named_modules():
        if isinstance(m, LoraLayer):
            flipped.append((m, m._disable_adapters))
            m._disable_adapters = True
    try:
        yield
    finally:
        for m, prev in flipped:
            m._disable_adapters = prev
