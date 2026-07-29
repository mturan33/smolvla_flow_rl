# Copyright 2026 Mehmet Turan Yardimci
#
# Licensed under the Apache License, Version 2.0. See the LICENSE file at the root of this repository.
"""Fetch the pretrained policy, and the backbone it asks for, from the model hub.

No weights are distributed with this repository. This helper downloads them at your request.

The default policy is one conditioned on the task suite this harness drives. That is not a preference: the
harness supplies two camera images under specific keys, an eight dimensional state, and expects seven
dimensional actions, and a policy declaring anything else cannot be driven by it. The generic SmolVLA base
checkpoint declares three differently named cameras and different widths, so it could not be driven here at all.
The wrapper checks what a checkpoint declares and refuses at construction, before any decision is taken, with a
message naming the mismatch.

The backbone is read out of the downloaded policy's own configuration rather than written down here. Different
SmolVLA checkpoints ask for different backbones, and a hardcoded one is right for whichever policy it was written
against and silently wrong for the next: the policy then reaches the network on first use, and on an offline
machine it fails with an error that names neither this helper nor the repository it wanted.

The processor configuration files are as important as the weights. They carry the normalisation statistics, and a
policy loaded without them runs and acts subtly wrong, which is a hard failure to attribute later. The whole
repository directory is fetched for that reason rather than the weight file alone.

Usage: python scripts/download_base_policy.py --out /path/to/base [--repo <hub id>] [--revision <sha>]
"""

import argparse
import json
import os

DEFAULT_POLICY = "HuggingFaceVLA/smolvla_libero"
# The revision whose model card was read when the licence statement in NOTICE was written. Pinned for the same
# reason the code upstreams are pinned there: a licence quoted from a moving target stops being a quotation.
DEFAULT_POLICY_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
# The backbone revision, read on the canonical repository. The default policy's configuration names an alias that
# the Hub redirects there; both resolve to this revision. Pinned rather than left floating for the same reason as
# the policy above, and recorded in NOTICE with the alias spelled out so the redirect is not a surprise.
BACKBONE_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
# Backbone repositories commonly ship exported graphs alongside the weights, and the torch load path never opens
# them. On the default backbone they are 24 files and about 5.9 GB against 2.0 GB of weights and tokeniser, so
# fetching everything triples the download for a repository whose stated premise is one consumer GPU. Skipping
# them is safe for any backbone, since transformers loads the safetensors and never the exports.
BACKBONE_SKIP = ["onnx/*", "*.onnx", "*.onnx_data"]


def refuse_in_tree(out: str) -> None:
    """Refuse an output directory that lies inside this repository.

    Weights and their processor configurations are not distributed here, and NOTICE, the README and this file all
    say so. A download placed inside the working tree turns that into something a single `git add -A` can falsify,
    and the upstream base of the default policy grants no licence at all. Ignore rules cover the weight formats,
    but the same download also writes configuration files that carry the normalisation statistics, and those are
    not weight shaped. Refusing the location is the only version of this that cannot be got wrong later.
    """
    # Asked of the filesystem rather than of the strings. Comparing paths, however normalised, misses everything
    # the filesystem knows and the name does not: a case variant spelling on a case insensitive mount, which the
    # repository lives on here, a symlinked or bind mounted second route, a hardlinked directory. Walking the
    # target's existing ancestors and asking whether any of them is this directory answers all of those at once.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_stat = os.stat(root)
    target = os.path.abspath(out)

    inside = False
    probe = target
    while True:
        try:
            if os.path.exists(probe) and os.path.samestat(os.stat(probe), root_stat):
                inside = True
                break
        except OSError:
            pass
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent

    if inside:
        raise SystemExit(
            "--out %s is inside this repository, and downloaded weights and processor configurations must not be. "
            "This repository distributes none, which several of its own files state. Choose a directory outside "
            "%s." % (target, root))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="local directory to place the policy in, outside this repository")
    ap.add_argument("--repo", default=DEFAULT_POLICY, help="model hub repository id for the policy")
    ap.add_argument("--revision", default=None,
                    help="policy revision, defaults to the pinned one when --repo is left at its default")
    ap.add_argument("--backbone", default=None,
                    help="override the backbone the policy asks for, which is otherwise read from its config")
    ap.add_argument("--backbone_revision", default=None, help="backbone revision")
    ap.add_argument("--skip_backbone", action="store_true",
                    help="download the policy only, leaving the backbone to be resolved at first use")
    args = ap.parse_args()
    refuse_in_tree(args.out)

    from huggingface_hub import snapshot_download

    # Whether this invocation is the one whose model cards were actually read. Repository identity is not enough:
    # a card is editable, so a different revision of the same repository may state different terms, and stating
    # them anyway would be the moving target NOTICE argues against in this repository's own words.
    at_defaults = (args.repo == DEFAULT_POLICY and args.revision is None
                   and args.backbone is None and args.backbone_revision is None
                   and not args.skip_backbone)
    revision = args.revision or (DEFAULT_POLICY_REVISION if args.repo == DEFAULT_POLICY else None)
    path = snapshot_download(repo_id=args.repo, revision=revision, local_dir=args.out)
    print("downloaded %s%s to %s" % (args.repo, (" at %s" % revision) if revision else "", path))

    fetched = [(args.repo, revision)]
    backbone = args.backbone
    if backbone is None and not args.skip_backbone:
        cfg_path = os.path.join(path, "config.json")
        try:
            backbone = json.load(open(cfg_path)).get("vlm_model_name")
        except OSError:
            backbone = None
        if backbone is None:
            print("could not read a backbone id from %s; pass --backbone to name one" % cfg_path)

    if backbone and not args.skip_backbone:
        # The policy pulls its language and vision backbone from a second repository when it is constructed.
        # Fetching only the policy leaves a setup that appears complete and then reaches the network on first use,
        # which fails on an offline machine at the least convenient moment.
        # The pinned revision belongs to the backbone the default policy asks for. Applying it to a backbone some
        # other policy named would send one repository's commit to a different repository, which fails with a
        # revision error naming a hash the user never typed.
        pin_backbone = (args.repo == DEFAULT_POLICY and args.revision is None and args.backbone is None)
        brev = args.backbone_revision or (BACKBONE_REVISION if pin_backbone else None)
        bpath = snapshot_download(repo_id=backbone, revision=brev, ignore_patterns=BACKBONE_SKIP)
        print("downloaded the backbone %s%s to the shared cache at %s"
              % (backbone, (" at %s" % brev) if brev else "", bpath))
        fetched.append((backbone, brev))

        # Fetching by commit hash caches the files but writes no branch pointer, because the hub only records one
        # when the requested revision is a name. The policy then asks for the backbone by name and with no
        # revision at all, so on a machine running offline, which is what launch.sh honours once a user sets the
        # flag for a warm cache, the load fails with an error naming the backbone rather than this helper. It also
        # fails that way on any machine with no network at all. Resolving the default branch once more
        # costs a metadata call and leaves that pointer behind.
        if brev:
            snapshot_download(repo_id=backbone, ignore_patterns=BACKBONE_SKIP)

    print()
    if at_defaults:
        print("Each repository fetched above states the Apache License, Version 2.0 on its model card, read at "
              "the pinned revisions. Those terms are theirs and not this repository's; see NOTICE.")
    else:
        print("This helper makes no statement about the terms covering what it just fetched, because it is not "
              "the combination whose model cards were read. Check the model card of each of:")
        for repo, rev in fetched:
            print("  %s%s" % (repo, (" at %s" % rev) if rev else " at whatever the repository head is"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
