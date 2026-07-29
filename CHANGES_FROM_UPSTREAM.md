# Changes from upstream

Four files in this repository are derivative works of RLinf and are licensed accordingly. Two of them are
additionally derivative works of LeRobot: `src/action_model.py` and `src/flow_sde.py`.

No upstream source file was copied into this tree. That is a statement about how the files were produced, and
it should not be read as a claim that little upstream material is present, because a good deal of it is. Three of
the four follow a single upstream module, `rlinf/models/embodiment/openpi/openpi_action_model.py`, closely enough
that in places the mathematics is reproduced line for line. The fourth, `src/value_head.py`, draws on two
upstream files: `rlinf/models/embodiment/modules/value_head.py` for the head itself, and the suffix pooling in
`openpi_action_model.py` for the function that turns policy features into a value. Anyone deciding how much
upstream material they are receiving should assume it is substantial and read the upstream alongside these files.

The upstreams are pinned, because a bare file path is not a reference: upstream headers move, and a retained
copyright line that was accurate when it was written stops matching without anything reporting it. RLinf is
pinned at commit `832db3f5c7aa6c61d72f20aaf43b7ce0f4ad2b03`, dated 2026-05-27; LeRobot at version `0.4.4`. Read
the upstream at those revisions when comparing.

Each derivative file carries, in its header, the upstream it derives from, a notice of what was modified as
Section 4(b) of the Apache License requires, and the upstream copyright notice that Section 4(c) requires a
derivative work to retain. A reader working from the headers alone is not given a shorter story than one who
reads this file; where the two differ it is the header that carries the extra sentence, since it can say what a
file does not do without the list here growing a bullet for every absence. This document collects the same
information in one place and lists which files are original, so the boundary is unambiguous.

## Files adapted from upstream projects

**`src/flow_sde.py`** implements the stochastic sampler for a flow matching policy.

- The velocity field is supplied by the caller as a callback rather than read from a specific model class, so the
  sampler works with any policy whose denoising step can be evaluated, and so the pretrained action head is never
  modified in order to sample from it.
- The integration grid and the deterministic step follow LeRobot's sampling loop in
  `lerobot/policies/smolvla/modeling_smolvla.py` rather than RLinf's, which builds the grid from a linear space
  and subtracts consecutive entries. The two are equal in exact arithmetic and differ in the last places in
  floating point; the reason for the change is not the size of that difference but that using the policy
  library's own expression is what makes the sampler's arithmetic the same as the policy's rather than close
  to it.

**`src/value_head.py`** implements the critic. The `ValueHead` class follows
`rlinf/models/embodiment/modules/value_head.py`; `compute_value_from_suffix` follows the suffix pooling in
`openpi_action_model.py`.

- Adaptive target rescaling was added, so the head learns against roughly unit scale targets while callers keep
  reading real units. The readout is reparameterised whenever the statistics move, so a statistics update does
  not by itself shift any prediction.
- The rescaling statistics are registered buffers rather than attributes, so they are saved with the weights.
  As attributes they were absent from every checkpoint, and since the readout is reparameterised by the running
  scale, a resumed run reloaded a readout divided by a scale it no longer had.
- Optional normalisation between hidden layers was added, applied to the hidden layers only.
- The initialisation gain for the smooth rectifier activations is mapped to the rectifier entry. The upstream
  passes the activation name straight to the gain table, which does not accept the head's own default, so this
  is a behavioural change and not only a tidy up.
- The upstream crops the suffix to the action chunk before pooling it. That crop is not repeated here, because
  the policy library this runs against already performs it immediately before the projection whose input the
  critic reads, so a second crop would be a no op in the ordinary case and a silent change of meaning in any
  other. This was checked against the library's source rather than assumed.

**`src/ppo_step.py`** implements the policy gradient update.

- The update grades stored trajectories from collection rather than freshly drawn ones. Pairing a fresh sample
  with a stored advantage makes the two independent and the gradient zero mean.
- The likelihood is summed over the executed part of the action chunk and the real degrees of freedom only,
  rather than over the padded chunk. The same slice is applied to the current, stored and reference likelihoods.
- Adaptive target rescaling for the critic was added, with the loss computed on the normalised scale.
- A non finite update guard was added: a batch producing non finite loss or gradients is dropped and reported
  rather than applied, since applying it poisons every parameter it touches.
- The reference anchor is written so that descending the penalty raises the likelihood of the executed action
  under the current policy. It defaults to off.

**`src/action_model.py`** wraps the pretrained policy with the pieces a policy gradient needs. This file has two
upstreams.

From RLinf, `openpi_action_model.py`:

- Written for a different policy family than the upstream targets.
- The features the critic reads are captured through a forward pre hook, so the pretrained action head is never
  edited. Adding a branch to the head would have broken the protocol's behavioural parity rule the moment it was
  written.

From LeRobot, `lerobot/policies/smolvla/modeling_smolvla.py`:

- The prefix cache construction follows that file closely: the observation is embedded once per chunk and the
  cached prefix is reused across the integration steps, rather than re-embedding at each step.
- What changed is the surrounding contract. The construction is exposed so an external sampler can drive the
  integration, the output is cropped to the executed part of the chunk and the real degrees of freedom, and the
  policy's own postprocessing is applied to the action tensor rather than to the full output dictionary.

## Files original to this repository

`src/gae.py`, `src/env_libero.py`, `src/env_processors.py`, `src/lora_utils.py`, `src/checkpoint.py`,
`src/completeness_gate.py`, `src/model_setup.py`, `src/rollout.py`, `src/evaluate.py`, `src/eval_cli.py`,
`src/tripwire.py`, `src/train_flow_rl.py`, everything under `scripts/`, `tests/` and `examples/`, and all
documentation except `THIRD_PARTY_LICENSES/`, which holds upstream licence texts, and `LICENSE`, which is the
Apache text itself.

That accounts for all sixteen modules in `src/`: four adapted, twelve original.

`assets/demo.gif` is original too, in the sense that matters here: it is footage of the author's own runs, not
adapted from any upstream file. It is rendered output rather than source, so what it contains is discussed in
`NOTICE` alongside the simulator components whose renderer produced it.

Every source file and shell script in this group carries a copyright line and no notice of modification, because
there is nothing upstream for them to modify. The Markdown documents carry neither; `NOTICE` carries the project
copyright line, since that is what a notice file is for. All of them are covered by the repository licence like
everything else.

## Dependencies

LeRobot supplies the policy implementation, the observation preprocessing and the environment factory. LIBERO
supplies the task suites and simulators. Both are imported, and no source file from either is copied into this
tree. The one directory that does hold somebody else's text is `THIRD_PARTY_LICENSES/`, which carries upstream
licence files whole because that is what those licences ask for. Their content is unaltered; their line endings
are normalised to LF like every other text file here, which `NOTICE` records for the one where it applied.
LeRobot is also an upstream of `src/action_model.py`, as recorded above and in `NOTICE`; LIBERO is a dependency
only.
