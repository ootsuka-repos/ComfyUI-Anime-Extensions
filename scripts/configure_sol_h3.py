#!/usr/bin/env python3
"""Connect existing, prepared Sol task manifests to this ComfyUI installation.

This command imports no GPU libraries, downloads nothing and installs no Python
packages. Prepare sources, interpreters, weights and the generic cache with the
pinned upstream tools first; --allow-pending records an explicitly incomplete setup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REVISION = "8e0db4fa562d727ea28b8d63015c196db7d97cae"
TASKS = ("t2va", "fl2va", "ref2va")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--sana-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True,
                        help="Shared checkpoint root visible to the Qwen container, e.g. the HF hub cache")
    parser.add_argument("--qwen-image", default="sol-h3-spark-qwen")
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime_root.expanduser().resolve(strict=True)
    package = args.sana_root.expanduser().resolve(strict=True) / "models/minimax_h3/Sol-H3-Spark"
    revision = subprocess.check_output(["git", "-C", str(package), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error(f"Expected Sana {REVISION}; found {revision}")
    args.comfy_root.expanduser().resolve(strict=True)
    weights = args.weights_root.expanduser().resolve(strict=True)
    paths = {task: str(runtime / f"paths-{task}.json") for task in TASKS}
    if not args.allow_pending:
        sys.path.insert(0, str(package))
        from runtime.config import load_paths
        for task, filename in paths.items():
            load_paths(filename, task=task)
    config = {"schema": 1, "package": str(package), "runtime_root": str(runtime),
              "paths": paths, "qwen_image": args.qwen_image, "weights_root": str(weights),
              "comfy_root": str(runtime / "sources/ComfyUI")}
    output = runtime / "config.json"
    text = json.dumps(config, indent=2) + "\n"
    if output.exists() and output.read_text() != text:
        parser.error(f"Existing host configuration differs; preserved: {output}")
    if not output.exists():
        with output.open("x") as stream:
            stream.write(text)
    print(output)
    print("Recorded pending setup; inference remains unavailable until preparation finishes."
          if args.allow_pending else "Task manifests resolve; model semantics and GPU inference still require validation.")


if __name__ == "__main__":
    main()
