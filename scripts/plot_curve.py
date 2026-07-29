# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. See the LICENSE file at the root of this repository.
"""Plot a training run's diagnostics from its metrics file.

Four panels, chosen because each answers a question that is hard to answer from the log text.

The likelihood ratio, with the clip range drawn on it. A ratio pinned at one means the update is not grading
what it thinks it is grading; a ratio drifting far outside the band means the data is too old for the policy.

The clipped fraction. When most samples are clipped the gradient is throttled, and a run in that state can look
stable for a long time while learning very little.

The gradient norm, on a log scale, since the interesting behaviour spans orders of magnitude.

The reward, which is the only one of the four that says whether any of this is working.

Usage: python scripts/plot_curve.py <run_dir> [--out /path/to/curve.png]

The default output goes inside the run directory, which is where a figure about a run belongs. The example spells
out an absolute path rather than a bare filename because the documented way to run this is from the repository
root, and a bare filename lands a figure of measured quantities in the working tree.
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(args.run_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(
            "no metrics.jsonl in %s, so there is nothing to plot. Point this at a run's --output_dir; the file is "
            "written one line per update cycle as the run proceeds." % args.run_dir)
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    if not rows:
        print("no metrics in %s" % path)
        return 1

    x = [r.get("cycle", i) for i, r in enumerate(rows)]
    panels = [
        ("likelihood ratio", "ratio_mean", False),
        ("clipped fraction", "clip_frac", False),
        ("gradient norm", "grad_norm", True),
        ("reward", "reward_mean", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (title, key, logy) in zip(axes.ravel(), panels):
        y = [r.get(key) for r in rows]
        if all(v is None for v in y):
            ax.set_axis_off()
            continue
        ax.plot(x, y, lw=1.4)
        if key == "ratio_mean":
            ax.axhline(1.0, color="grey", lw=0.8)
            # The band is drawn only when the run recorded the range it actually used. Assuming a value would
            # draw a boundary the run never had, in a figure that gives the reader no way to tell.
            clip = rows[0].get("clip_range")
            if clip is None:
                title = title + " (clip range not recorded)"
            else:
                ax.axhspan(1 - clip, 1 + clip, color="grey", alpha=0.15)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("update cycle")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    out = args.out or os.path.join(args.run_dir, "curve.png")
    fig.savefig(out, dpi=130)
    print("wrote %s from %d cycles" % (out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
