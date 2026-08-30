#!/usr/bin/env python3
# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Single source for the training rollout's reset seeds.

WHY THIS EXISTS, and it is a bug that hides well. Two runs launched with the same seed can part at the very
first rollout step, on the OBSERVATION side, before the sampler is called even once. The cause is a second
reset: the trainer builds the environment with a seeded reset and then resets AGAIN at the top of the outer
loop with no seed, so the initial states come from whatever RNG state the worker happens to hold. A
manipulation simulator will amplify a one-ULP difference in initial state into a different next step, so
different initial states are enough to produce different truncation counts and, downstream, different
returns. Nothing in the training loop looks wrong while this happens.

REFERENCE PARITY, not a deviation. The reference implementation seeds its training env resets:

    rlinf/envs/libero/libero_env.py:86    self.seed = self.cfg.seed + seed_offset
    rlinf/envs/libero/libero_env.py:100   self._generator = np.random.default_rng(seed=self.seed)
    rlinf/envs/libero/libero_env.py:410   reset state ids drawn from that generator
    rlinf/workers/env/env_worker.py:249   seed_offset = rank * stage_num + stage_id

    (RLinf, pin 832db3f5c7aa6c61d72f20aaf43b7ce0f4ad2b03)

So the reference derives a deterministic per-worker stream and draws reset states from it. Seeding ours is a
parity fix rather than an invention.

SEMANTICS. The seed is derived from the run seed, the environment index and the outer counter. Every outer
and every environment gets its own stream, and the same run is bit-identical when repeated.

`SeedSequence` does the derivation rather than hand-rolled arithmetic. Hand-rolled offsets have to defend a
collision argument against whatever other seed namespace the evaluation path uses, and against large seeds
and long runs; `SeedSequence` is collision-free by construction over the whole (seed, outer, env) triple, so
there is no arithmetic left to get wrong later.
"""
import numpy as np

MAX = 2 ** 31 - 1


def train_reset_seeds(base_seed, outer, n_envs):
    """The per-env reset seeds for one outer. Deterministic in (base_seed, outer, env index)."""
    if n_envs <= 0:
        raise ValueError("n_envs must be positive, got %r" % (n_envs,))
    out = []
    for i in range(n_envs):
        ss = np.random.SeedSequence([int(base_seed), int(outer), int(i)])
        out.append(int(ss.generate_state(1)[0] % MAX))
    return out


def _selftest():
    """Asserted in both directions, because a seed that is ignored reads exactly like a seed that works."""
    a = train_reset_seeds(1337, 0, 7)
    assert a == train_reset_seeds(1337, 0, 7), "same inputs must give the same seeds"
    assert a != train_reset_seeds(2337, 0, 7), "a different run seed must give different seeds"
    assert a != train_reset_seeds(1337, 1, 7), "a different outer must give different seeds"
    assert len(set(a)) == len(a), "envs within one outer must not share a stream"
    assert all(0 <= s <= MAX for s in a)
    print("seeding selftest OK:", a[:3], "...")


if __name__ == "__main__":
    _selftest()
