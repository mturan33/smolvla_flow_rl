# Verification protocol

A protocol for reinforcement learning fine tuning runs whose numbers will be reported. It is written for the
failure mode this class of work is prone to: the run does not crash, the loss falls, and the result is wrong.

The rules below are written as engineering practice, not as a record of any particular project. Where a rule
describes a failure, it is described by its shape rather than by its circumstances, because the shape is what
generalises.

Each rule is marked with how far it is automated here:

- **[gate]** runs from `scripts/run_gates.sh` and fails the build.
- **[runtime]** is enforced by the code at run time, and prints its evidence into the log.
- **[manual]** is stated but not automated in this repository, usually because checking it needs a trained
  checkpoint, a second machine, or a judgement call. These are the rules a reader must apply themselves.

## The single principle

A check that passes tells you nothing unless you know it would have failed. Every **[gate]** rule below has been
given the defect it exists to catch and required to fail on it, and those negative tests ship alongside the
gates. A gate that has only ever passed has not been tested, it has been watched.

## Before a run starts

**Manifest.** [runtime] Print the trainable parameter count, broken down by module group, and the per group learning rates
read back from inside the optimiser rather than from the arguments. A configuration in which the intended
parameters are frozen, or in which a group was created but never attached to the optimiser, produces a run that
trains something and improves nothing. Neither condition raises an error.

**Effect proof.** [runtime] Any mechanism enabled by a flag must print numerical evidence that it did something, in the
first steps where it is active. Evidence means a quantity that would look different if the mechanism were off,
not a line saying it is enabled. A flag that is parsed, stored and never read looks identical to one that works,
in the arguments, in the configuration dump and in the log.

**Contact first.** [manual] Establish that the setup can learn a single task before reading anything into what a
longer run produces. A setup that cannot move at all on one task will still produce numbers, and those numbers
will describe the setup rather than the question being asked of it.

## Invariants

**First update ratio.** [runtime] On the first update, before any parameter has moved, the likelihood ratio between the
current and the stored policy must be one within tolerance. Any other value means the stored and recomputed
likelihoods are computed on different surfaces, in which case every update in the run is being driven by that
discrepancy rather than by the advantage.

**Preprocessing parity.** [manual] The same raw observation must produce the same tensors everywhere it is consumed, and
the preprocessing must not modify what it was given. A recording or logging path that mutates the observation it
inspects perturbs the very rollout it is recording.

**Behavioural parity with the pretrained policy.** [manual] Before any parameter has moved, a deterministic rollout through
this wrapper must produce the action the pretrained policy produces from the same observation, to within the
arithmetic. The wrapper drives the pretrained head through a callback and never edits it, so a difference here is
a difference the wrapper introduced, and any later statement about what training changed would be measuring the
wrapper instead. Perform it by scoring one observation through both paths at equal noise and comparing the
executed action; it is left manual because it needs the real weights and a device rather than a fixture.

**Cross process reproducibility.** [manual] The same checkpoint under the same seed must produce the same evaluation in an
independent process. A number that cannot be reproduced in a second process cannot be checked by anyone else. It
is cheap to test and is usually assumed instead.

**Resume fidelity.** [gate] A checkpoint must restore the random state as well as the parameters and the optimiser.
Without it, a resumed run draws a different sequence than an uninterrupted one would have drawn at the same
point, and is no longer reproducible from its seed.

## Reading results

**Completeness.** [gate] No number may be produced from an unfinished evaluation. Completeness means the total episode
count derived from the configuration, verified against what is on disk. Checking that every task has appeared at
least once is not a completeness criterion: in a design that cycles through the task list it is satisfied almost
immediately, and it will let a reader report a confident number from a small fraction of the intended episodes.

**Certificates must be backed.** [manual] A marker file recording that a stage finished may only be written when the
artefacts it certifies exist. Writing it on the exit code of a command means a stage that produced nothing looks
complete to everything downstream.

**Provenance claims.** [manual] A recording may be described as the episode a reported number came from only if it
reproduces that episode, checked against the stored per episode record including the step at which it ended. Two
different trajectories can reach the same outcome, so matching the outcome alone is not evidence of provenance.

## Keeping the checks alive

**Run them on a schedule, not on demand.** [manual] A gate that only runs when someone remembers its name is a file. The
whole suite runs from one entry point, and that entry point is what continuous integration or a scheduled job
should call. Nothing in this repository schedules it, which is why the rule is marked manual: wiring it to
something that runs without being asked is the reader's job, and until it is done the gates are on demand.

**What runs must be what is committed.** [manual] If runs execute from a different copy of the tree than the one under
version control, a fix can be written, tested and committed while having no effect on any run. Hash the shared
files and fail on any difference.

**The dependency set must be committed.** [gate] Walk the import closure of each entry point and fail if it reaches a
module that is not in the repository. A repository that cannot reproduce a number because part of the system
lives outside it is not an artefact, whatever its file listing suggests.

**Reproduce, do not merely resolve.** [manual] The strongest available check is to run an evaluation with the execution
tree removed from the module search path and compare it against the stored record, episode by episode. Structural
checks can all pass while this fails.
