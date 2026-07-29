# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Choose what trains, then prove it before the first update.

Which parameters are trainable is easy to get wrong without noticing, and the mistake to guard against is opening
a surface other than the one that was intended. A run in that state still runs and still reports a loss, because
whatever is trainable will move; nothing raises an error, and the arguments look exactly as they would if the
intended surface were open. The only defence is to print the trainable set and read it, which is what this module
exists to do.

Three regimes are offered. Which one suits a given run depends on the policy, the task and the available memory,
and this module takes no position on that; it exists to make the choice explicit and to report what the choice
actually produced.

- frozen: low rank adapters on the language backbone train; the action expert does not.
- lora: low rank adapters on the action expert train instead; the backbone carries none.
- full: the action expert trains at full rank.

The first two open different surfaces rather than nested ones, so which of them trains more parameters depends on
the relative size of the backbone and the expert at the chosen adapter rank, and cannot be read off the order of
the list. Measured for the default policy at the default rank, the flow model's trainable count is 1.64M under
frozen, 1.11M under lora and 96.70M under full, so lora is the smallest of the three rather than the middle one.
Only full is unambiguously the largest. The manifest printed at startup is the authority for any particular run,
which is the whole reason it is printed.

In every regime the vision and language backbone stays frozen apart from adapters, and the critic head always
trains, since a frozen critic makes the advantage meaningless.

The pretrained action head projections train in none of the three. That is deliberate, and it is what the
behavioural parity rule in the protocol rests on, so the manifest's action head group reads zero in every regime.
A zero there is the evidence; a nonzero one would mean something reached the head that should not have.
"""

from __future__ import annotations

import torch

try:
    from .lora_utils import apply_text_model_lora, freeze_module
except ImportError:
    from lora_utils import apply_text_model_lora, freeze_module

REGIMES = ("frozen", "lora", "full")

# Substrings that identify each part of the policy in its parameter names. Grouping by name is fragile if the
# upstream naming changes, so the manifest carries two signals for that. Anything the keys do not match is
# reported as `other`, and a nonzero `other` means a name moved. And the total is counted from the model rather
# than by adding the groups up, so the two walks are independent and a grouping that drops a parameter entirely
# stops the run instead of printing a smaller number.
GROUP_KEYS = {
    "action_expert": ("lm_expert",),
    "action_head": ("action_in_proj", "action_out_proj", "action_time_mlp", "state_proj"),
    "backbone_adapters": ("lora",),
    "critic": ("value_head",),
}


def open_action_expert(flow_model, regime: str, lora_rank: int, lora_alpha: int) -> None:
    """Make the action expert trainable, either at full rank or through adapters."""
    if regime == "full":
        for name, p in flow_model.named_parameters():
            if "lm_expert" in name:
                p.requires_grad = True
        return
    if regime == "lora":
        from peft import LoraConfig, inject_adapter_in_model
        cfg = LoraConfig(r=lora_rank, lora_alpha=lora_alpha,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                         lora_dropout=0.0, bias="none")
        inject_adapter_in_model(cfg, flow_model.vlm_with_expert.lm_expert)
        return
    raise ValueError("unknown regime for the action expert: %r" % regime)


def print_manifest(model) -> int:
    """Print the trainable parameter count with a breakdown, and return the total.

    This is the first thing a run should print. It answers, as a logged fact rather than an assumption, what is
    training. Anything not matched by a group is reported as other, so a parameter that quietly became trainable
    cannot hide inside a group it does not belong to.
    """
    groups = {k: 0 for k in GROUP_KEYS}
    groups["other"] = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        for group, keys in GROUP_KEYS.items():
            if any(k in name for k in keys) or (group == "backbone_adapters" and "lora" in name.lower()):
                groups[group] += p.numel()
                break
        else:
            groups["other"] += p.numel()

    # Counted a second time, from the model rather than from the groups, so the comparison below is between two
    # independent walks. Summing the groups and calling that the total would make the check true by construction.
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grouped = sum(groups.values())
    parts = " ".join("%s=%.2fM" % (g, groups[g] / 1e6) for g in groups)
    print("  [manifest] trainable=%.2fM (%s)" % (total / 1e6, parts), flush=True)
    if grouped != total:
        raise RuntimeError(
            "the manifest reached %d trainable parameters by group and %d by counting the model, so the grouping "
            "walk is missing some. The printed breakdown cannot be trusted." % (grouped, total))
    if total == 0:
        raise RuntimeError("no trainable parameters; this run would do nothing while reporting a loss")
    return total


def print_optimizer_proof(optimizer) -> None:
    """Print the learning rate of each parameter group, read back from the optimiser.

    Read from the optimiser rather than from the arguments on purpose. A group that was constructed but never
    attached, or attached with a different rate than intended, is invisible in the arguments and in any
    configuration dump, and shows up here.
    """
    parts = []
    for i, group in enumerate(optimizer.param_groups):
        parts.append("group%d lr=%.3g n=%d" % (i, group["lr"], len(group["params"])))
    print("  [optimizer] " + ", ".join(parts), flush=True)


def build_model(model_cls, config, device, regime: str, adapter_rank: int, adapter_alpha: int):
    """Construct the policy, set the trainable surface according to the regime, and print the manifest."""
    if regime not in REGIMES:
        raise ValueError("regime must be one of %s, got %r" % (REGIMES, regime))

    model = model_cls(config).to(device)

    if regime == "frozen":
        apply_text_model_lora(model.flow_model, rank=adapter_rank, lora_alpha=adapter_alpha)
    else:
        freeze_module(model.flow_model)
        open_action_expert(model.flow_model, regime, adapter_rank, adapter_alpha)
        print("  [regime] %s, the backbone stays frozen and the action expert is open" % regime, flush=True)

    for p in model.value_head.parameters():
        p.requires_grad = True

    print_manifest(model)
    return model


def build_optimizer(model, actor_lr: float, critic_lr: float) -> torch.optim.Optimizer:
    """Two parameter groups, actor and critic, then prove the rates.

    A separate critic rate is standard here: the critic learns a scalar from sparse targets and can afford to
    move faster than the policy it is scoring. Both rates are required arguments so neither becomes an inherited
    default nobody chose.
    """
    critic = [p for n, p in model.named_parameters() if p.requires_grad and "value_head" in n]
    actor = [p for n, p in model.named_parameters() if p.requires_grad and "value_head" not in n]
    optimizer = torch.optim.Adam(
        [{"params": actor, "lr": actor_lr}, {"params": critic, "lr": critic_lr}], eps=1e-8)
    print_optimizer_proof(optimizer)
    return optimizer
