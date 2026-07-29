# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. You may obtain a copy of the License in the LICENSE file at the
# root of this repository or at http://www.apache.org/licenses/LICENSE-2.0
"""Stop a run that has already failed, instead of letting it spend the night proving it.

Fine tuning a large policy on a small budget fails in a small number of recognisable ways, and none of them
raise an exception. The two watched here are the ones that waste the most time.

The likelihood ratio blows up. Once the current policy has moved far enough from the one that collected the
data, the update is off policy in a way importance sampling cannot fix, and the run is doing damage rather than
learning. A single large ratio is noise, so the decision is made on a window: the median over the window says the
drift is systematic, and the maximum catches a spike that a median would absorb.

Success collapses. If the evaluated success rate falls below a floor the run has lost the behaviour it started
from, and continuing only buries the checkpoint that still had it.

The exit status is distinct from an ordinary failure, so a caller can tell a run that was stopped on purpose from
one that crashed. A tripped run is an outcome, not an error.
"""

from __future__ import annotations

import os
import statistics
from typing import List, Optional

TRIPPED_EXIT_CODE = 42


class Tripwire:
    """Watches update statistics and evaluation results, and says when to stop.

    Disabled by default. An always on guard whose thresholds were chosen for one setting will eventually kill a
    healthy run in another, so switching it on is the caller's decision and the thresholds are arguments.
    """

    def __init__(self, enabled: bool, ratio_median_limit: float, ratio_max_limit: float,
                 success_floor: float, window: int, output_dir: Optional[str] = None):
        # Thresholds are only meaningful when the guard is on, and the trainer builds this object either way, so
        # absent values must be tolerated rather than coerced. Requiring them at construction made every run
        # without the guard fail before it started.
        self.enabled = bool(enabled)
        self.ratio_median_limit = float(ratio_median_limit) if ratio_median_limit is not None else float("inf")
        self.ratio_max_limit = float(ratio_max_limit) if ratio_max_limit is not None else float("inf")
        self.success_floor = float(success_floor) if success_floor is not None else float("-inf")
        self.window = int(window) if window is not None else 0
        self.output_dir = output_dir
        self._medians: List[float] = []
        self._maxima: List[float] = []
        self.reason: Optional[str] = None

    def arm(self, extra: str = "") -> None:
        if not self.enabled:
            return
        # A floor at or below zero cannot fire, since a pooled rate is never negative. Said here rather than left
        # for a reader to work out, because the armed line is the only evidence in a run that a guard exists and
        # it should not imply a limit that will never be tested.
        floor = ("%.3g" % self.success_floor if self.success_floor > 0
                 else "%.3g, which cannot fire since a rate is never negative" % self.success_floor)
        print("  [tripwire] armed, ratio median limit %.3g, ratio max limit %.3g, success floor %s, "
              "window %d%s" % (self.ratio_median_limit, self.ratio_max_limit, floor,
                               self.window, (" " + extra) if extra else ""), flush=True)

    def _trip(self, reason: str) -> bool:
        self.reason = reason
        print("  [tripwire] tripped: %s" % reason, flush=True)
        if self.output_dir:
            try:
                with open(os.path.join(self.output_dir, "TRIPWIRE_REASON.txt"), "w") as fh:
                    fh.write(reason + "\n")
            except OSError:
                pass
        return True

    def observe_update(self, ratio_median: float, ratio_max: float) -> bool:
        """Record one update's ratio statistics. Returns True when the run should stop.

        Only the first window is watched, counted in update cycles rather than in optimiser steps, since the
        trainer calls this once per cycle with that cycle's last update. Later drift is the policy having moved on
        purpose, which is what training is; the failure this catches happens early or not at all.
        """
        if not self.enabled or self.window <= 0 or len(self._medians) >= self.window:
            return False
        self._medians.append(float(ratio_median))
        self._maxima.append(float(ratio_max))
        if len(self._medians) < self.window:
            return False
        med = statistics.median(self._medians)
        mx = max(self._maxima)
        # The window's statistics are printed whether or not they cross a limit. A guard that speaks only when it
        # fires is indistinguishable, in a log, from one that was never wired to anything; this line is the
        # evidence that it observed the run.
        print("  [tripwire] window closed, median %.3g against limit %.3g, max %.3g against limit %.3g"
              % (med, self.ratio_median_limit, mx, self.ratio_max_limit), flush=True)
        if med > self.ratio_median_limit or mx > self.ratio_max_limit:
            return self._trip("ratio window median %.3g and max %.3g exceed the limits" % (med, mx))
        return False

    def observe_eval(self, pooled_success: float) -> bool:
        """Record one evaluation result. Returns True when the run should stop."""
        if not self.enabled:
            return False
        if pooled_success < self.success_floor:
            return self._trip("pooled success %.3g fell below the floor" % pooled_success)
        return False
