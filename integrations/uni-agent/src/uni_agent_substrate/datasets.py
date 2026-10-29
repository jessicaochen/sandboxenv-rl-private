# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a uni-agent task parquet from one of several SWE datasets, with digest-pinned images for Substrate.

Every source yields the same row shape, so swapping datasets is a flag change:

    {"data_source": ..., "prompt": [{"role": "user", "content": problem_statement}],
     "extra_info": {"tools_kwargs": {"task": {"name": <task>, "sandbox": {"image": <ref>},
                                              "metadata": {...}}}}}

Sources:
  r2e_gym_easy  hfilaretov/Benchmark-R2E-Gym-Easy (JSONL; R2E-Gym rows under metadata.instance_dict)
  r2e_gym       R2E-Gym/R2E-Gym-Subset (or any dataset with the same columns)
  swe_rebench   uni-agent's built-in builder (nebius/SWE-rebench filtered)
  swe_bench     uni-agent's built-in builder (princeton-nlp/SWE-bench_Verified)

Images: Substrate needs digest-pinned rootfs references. Run once with --images-out to
list the slice's tag references, resolve them with scripts/resolve_images.sh, then run again
with --image-digests to write pinned references into the parquet. The slice is
deterministic (sorted by instance id, first --max-instances), so both runs agree.

Example:
  python3 -m uni_agent_substrate.datasets --source r2e_gym_easy --max-instances 64 --images-out images.txt --out train.parquet
  scripts/resolve_images.sh images.txt REGION-docker.pkg.dev/PROJECT/<dockerhub-remote-repo> images.json
  python3 -m uni_agent_substrate.datasets --source swe_bench --max-instances 20 --exclude-file offline-exclusions.txt \
      --image-digests images.json --out val.parquet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

R2E_TASK = "r2e_gym"
R2E_EASY_DATASET = "hfilaretov/Benchmark-R2E-Gym-Easy"
R2E_EASY_FILES = {"train": "benchmark_r2e_gym_easy_train.jsonl", "val": "benchmark_r2e_gym_easy_val.jsonl"}
R2E_SUBSET_DATASET = "R2E-Gym/R2E-Gym-Subset"
R2E_METADATA_KEYS = (
    "instance_id",
    "repo_name",
    "base_commit",
    "commit_hash",
    "problem_statement",
    "expected_output_json",
    "parsed_commit_content",
)


def r2e_row(example: dict[str, Any], data_source: str) -> dict[str, Any]:
    """R2E-Gym columns -> uni-agent task row."""
    metadata = {k: example.get(k, "") for k in R2E_METADATA_KEYS}
    if not metadata["instance_id"]:
        metadata["instance_id"] = f"{example.get('repo_name', 'repo')}-{example.get('commit_hash', '')}"
    if not metadata["expected_output_json"]:
        raise ValueError(f"{metadata['instance_id']}: missing expected_output_json")
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": example["problem_statement"]}],
        "extra_info": {
            "tools_kwargs": {
                "task": {"name": R2E_TASK, "sandbox": {"image": example["docker_image"]}, "metadata": metadata}
            }
        },
    }


def load_r2e_gym_easy(split: str) -> Iterable[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(R2E_EASY_DATASET, R2E_EASY_FILES[split], repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            inst = json.loads(row["responses_create_params"]["metadata"]["instance_dict"])
            yield r2e_row(inst, R2E_EASY_DATASET)


def load_r2e_gym(dataset: str, split: str) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    for example in load_dataset(dataset, split=split):
        yield r2e_row(example, dataset)


def load_uni_agent_builtin(source: str) -> Iterable[dict[str, Any]]:
    """Delegate to uni-agent's own preprocessors (same row shape)."""
    if source == "swe_bench":
        from uni_agent.tasks.swe_bench.preprocess import (
            build_swe_bench_verified as build,
        )
    else:
        from uni_agent.tasks.swe_rebench.preprocess import build_swe_rebench as build
    # uni-agent caps before sorting; take everything and slice deterministically below.
    yield from build(None)


def instance_id(row: dict[str, Any]) -> str:
    return str(row["extra_info"]["tools_kwargs"]["task"]["metadata"].get("instance_id", ""))


def image_of(row: dict[str, Any]) -> str:
    return row["extra_info"]["tools_kwargs"]["task"]["sandbox"]["image"]


def set_image(row: dict[str, Any], ref: str) -> None:
    row["extra_info"]["tools_kwargs"]["task"]["sandbox"]["image"] = ref


def select_rows(rows: list[dict[str, Any]], max_instances: int | None, shuffle_seed: int | None) -> list[dict[str, Any]]:
    """Deterministic slice: sort by instance id, optionally shuffle with a seed, then take the head."""
    rows = sorted(rows, key=instance_id)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
    return rows if max_instances is None else rows[:max_instances]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=["r2e_gym_easy", "r2e_gym", "swe_rebench", "swe_bench"])
    ap.add_argument("--dataset", help="HF dataset id for --source r2e_gym (default R2E-Gym/R2E-Gym-Subset; R2E-Gym/R2E-Gym-V1 is the full set)")
    ap.add_argument("--split", default="train", help="split (r2e_gym_easy: train|val; r2e_gym: HF split)")
    ap.add_argument("--max-instances", type=int, help="keep the first N instances (by sorted instance id, or after --shuffle-seed)")
    ap.add_argument("--shuffle-seed", type=int, help="shuffle deterministically before slicing (datasets sorted by repo need this)")
    ap.add_argument("--exclude-file", help="file of instance ids to drop (first token per line, # comments)")
    ap.add_argument("--images-out", help="write the slice's image references (one per line)")
    ap.add_argument("--image-digests", help="JSON {tag ref: digest ref}; rewrite images in the output")
    ap.add_argument("--out", required=True, help="output parquet path")
    args = ap.parse_args()

    if args.source == "r2e_gym_easy":
        rows = list(load_r2e_gym_easy(args.split))
    elif args.source == "r2e_gym":
        rows = list(load_r2e_gym(args.dataset or R2E_SUBSET_DATASET, args.split))
    else:
        rows = list(load_uni_agent_builtin(args.source))
    total = len(rows)

    excluded = set()
    if args.exclude_file:
        for line in Path(args.exclude_file).read_text(encoding="utf-8").splitlines():
            tok = line.split("#", 1)[0].strip().split()
            if tok:
                excluded.add(tok[0])
        rows = [r for r in rows if instance_id(r) not in excluded]

    rows = select_rows(rows, args.max_instances, args.shuffle_seed)

    images = sorted({image_of(r) for r in rows})
    if args.images_out:
        Path(args.images_out).write_text("\n".join(images) + "\n", encoding="utf-8")

    unpinned = 0
    if args.image_digests:
        digests = json.loads(Path(args.image_digests).read_text(encoding="utf-8"))
        kept = []
        for r in rows:
            ref = image_of(r)
            pinned = ref if "@sha256:" in ref else digests.get(ref) or digests.get("docker.io/" + ref)
            if pinned is None:
                unpinned += 1
                continue
            set_image(r, pinned)
            kept.append(r)
        rows = kept

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out)
    print(
        f"{args.source}: loaded {total}, excluded {len(excluded)}, wrote {len(rows)} rows to {out} "
        f"({len(images)} distinct images{', ' + str(unpinned) + ' dropped as unpinned' if unpinned else ''})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
