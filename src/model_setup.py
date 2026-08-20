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

Four regimes are offered. Which one suits a given run depends on the policy, the task and the available memory,
and this module takes no position on that; it exists to make the choice explicit and to report what the choice
actually produced.

- expert_full_vlm_lora: the action expert trains at full rank, low rank adapters on the language backbone train
  alongside it, and the flow action head trains at full rank. The vision tower and the connector stay frozen.
  This is the RECOMMENDED DEFAULT.
- frozen: low rank adapters on the language backbone train; the action expert does not.
- lora: low rank adapters on the action expert train instead; the backbone carries none.
- full: the action expert trains at full rank; the backbone carries no adapter.

The first regime is recommended because the other three each leave out one of the two things that have to move.
An adapter on the backbone cannot compensate for an action decoder that never changes, and an action decoder
opened without any adaptation of the language side has no way to use a changed instruction. Adapting the
backbone while training the decoder at full rank is the combination reported to work in the recent
fine-tuning literature; see the design section of the README for the references. Nothing in this repository
reports a result, and this recommendation is a statement about published practice, not about our measurements.

`frozen` and `lora` open different surfaces rather than nested ones, so which of them trains more parameters
depends on the relative size of the backbone and the expert at the chosen adapter rank, and cannot be read off
the order of the list. Measured for the default policy at the default rank, the flow model's trainable count is
1.64M under frozen, 1.11M under lora and 96.70M under full, so lora is the smallest of those three rather than
the middle one. `expert_full_vlm_lora` is the largest of the four, since it is `full` plus the backbone adapters
plus the action head. The manifest printed at startup is the authority for any particular run, which is the
whole reason it is printed.

In every regime the vision and language backbone stays frozen apart from adapters, and the critic head always
trains, since a frozen critic makes the advantage meaningless.

The pretrained action head projections train in the `frozen` and `lora` regimes: NOT AT ALL. That is
deliberate there, and it is what the behavioural parity rule in the protocol rests on, so the manifest's
action head group reads zero in THOSE TWO regimes. A zero there is the evidence; a nonzero one would mean
something reached the head that should not have.

The invariant is NARROWED to those two regimes on purpose, and the narrowing is the point rather than an
oversight. In `expert_full_vlm_lora` the head trains at full rank, so its manifest group is EXPECTED to be
nonzero and a zero there would be the fault. The parity rule cannot apply to a regime that deliberately
changes the head: parity with the pretrained policy is a statement about a surface that leaves the action
path alone, and this regime does not. `full` is left as it was, head frozen, so that it remains the
adapter-free control for exactly this comparison. Which regimes the invariant covers is written in the
protocol as well as here, because an invariant whose scope lives in only one file is an invariant that
quietly widens.
"""

from __future__ import annotations

import torch

try:
    from .lora_utils import apply_text_model_lora, freeze_module
except ImportError:
    from lora_utils import apply_text_model_lora, freeze_module

REGIMES = ("expert_full_vlm_lora", "frozen", "lora", "full")
# The regimes in which the pretrained action head must stay frozen. The behavioural parity rule in the
# protocol applies to exactly these and to no others; see the module docstring for why the scope is
# narrow rather than universal.
HEAD_FROZEN_REGIMES = ("frozen", "lora", "full")
# Regimes whose surface satisfies the rule that the action expert trains at full rank. A run in any other
# regime has to declare itself a control or ablation before it is allowed to start.
EXPERT_FULL_REGIMES = ("expert_full_vlm_lora", "full")

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
    if regime in ("full", "expert_full_vlm_lora"):
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


def open_action_head(flow_model) -> None:
    """Make the flow action head trainable at full rank.

    Used only by `expert_full_vlm_lora`. The head is five small projections; opening it is what lets the
    regime change the action path rather than only the representation feeding it. In every other regime it
    stays frozen and the manifest's action head group must read zero, which is what the parity rule checks.
    """
    opened = 0
    for name, p in flow_model.named_parameters():
        if any(k in name for k in GROUP_KEYS["action_head"]):
            p.requires_grad = True
            opened += 1
    if opened == 0:
        raise RuntimeError(
            "no action head parameters were found, so opening the head matched nothing. The keys %r did not "
            "appear in any parameter name, which means the naming moved. An empty match is an error here, not "
            "a silent no-op: the regime would then run with a frozen head while claiming an open one."
            % (GROUP_KEYS["action_head"],))


def assert_surface(model, regime: str, control_arm: bool = False, control_reason: str = "") -> bool:
    """Refuse to start unless the action expert trains at full rank, or the run declares itself a control.

    The manifest alone is not a gate. It prints what is trainable and a reader may or may not look. This
    raises. The question it asks is not "is the expert somewhere in the trainable set" but "is the expert's
    OWN body training", because an adapter attached to the expert satisfies the first and not the second, and
    the difference between those two is not visible in a parameter count read casually.

    Decided from the expert body's trainable FRACTION rather than a hardcoded parameter count, so it survives
    a change of model size, of layer count or of naming.
    """
    total = trainable = 0
    for name, p in model.named_parameters():
        if "lm_expert" not in name or "lora" in name.lower():
            continue
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    if total == 0:
        raise RuntimeError(
            "no action expert parameters were found at all, so the surface cannot be checked. Either this is "
            "not the expected architecture or the naming changed. An empty match set is an error, never a pass.")

    fraction = trainable / total
    print("  [surface] regime=%s action expert trainable fraction=%.4f (%d of %d own parameters)"
          % (regime, fraction, trainable, total), flush=True)

    if fraction >= 0.99:
        return True

    message = (
        "the action expert does not train at full rank in this run: %.2f%% of its own parameters carry "
        "gradients. A run whose action decoder cannot move is not a fine-tuning run of the action decoder, "
        "whatever else is trainable." % (100.0 * fraction))
    if control_arm:
        print("  [surface] DECLARED CONTROL OR ABLATION: %s" % message, flush=True)
        print("  [surface] reason given: %s" % (control_reason or "(none supplied)"), flush=True)
        print("  [surface] this run's results carry a surface label and are not comparable to a full-rank run.",
              flush=True)
        return False
    raise RuntimeError(
        message + " Pass control_arm=True, with a reason, if this is deliberately a control or an ablation.")


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
    all_params = sum(p.numel() for p in model.parameters())
    pct = (100.0 * total / all_params) if all_params else 0.0
    print("  [manifest] trainable=%.2fM of %.2fM (%.3f%% of the model): %s"
          % (total / 1e6, all_params / 1e6, pct, parts), flush=True)
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


def build_model(model_cls, config, device, regime: str, adapter_rank: int, adapter_alpha: int,
                control_arm: bool = False, control_reason: str = "", enforce_surface: bool = True):
    """Construct the policy, set the trainable surface according to the regime, print the manifest, and gate.

    The gate runs HERE rather than at the call site so that it cannot be forgotten by a new entry point. A
    check that each caller has to remember is a check that one caller eventually will not.
    """
    if regime not in REGIMES:
        raise ValueError("regime must be one of %s, got %r" % (REGIMES, regime))

    model = model_cls(config).to(device)

    if regime == "frozen":
        apply_text_model_lora(model.flow_model, rank=adapter_rank, lora_alpha=adapter_alpha)
    elif regime == "expert_full_vlm_lora":
        # Order matters. apply_text_model_lora freezes the policy before it injects, so it has to run FIRST;
        # opening the expert or the head before it would have that work silently undone.
        apply_text_model_lora(model.flow_model, rank=adapter_rank, lora_alpha=adapter_alpha)
        open_action_expert(model.flow_model, regime, adapter_rank, adapter_alpha)
        open_action_head(model.flow_model)
        print("  [regime] %s, the action expert trains at full rank, the language backbone carries adapters, "
              "the flow action head trains at full rank, and the vision tower and connector stay frozen"
              % regime, flush=True)
    else:
        freeze_module(model.flow_model)
        open_action_expert(model.flow_model, regime, adapter_rank, adapter_alpha)
        print("  [regime] %s, the backbone stays frozen and the action expert is open" % regime, flush=True)

    for p in model.value_head.parameters():
        p.requires_grad = True

    print_manifest(model)
    # The gate is about what a RUN TRAINS, so evaluation passes enforce_surface=False: scoring an existing
    # checkpoint trains nothing, and its surface was decided when it was produced. Gating there would have
    # made every checkpoint from the control regimes unscoreable, which is the opposite of what the gate is
    # for. It defaults to True so that a new TRAINING entry point has to opt out deliberately.
    if enforce_surface:
        assert_surface(model, regime, control_arm=control_arm, control_reason=control_reason)
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
