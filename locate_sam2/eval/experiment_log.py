from __future__ import annotations

import hashlib
import json
import pickle
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from locate_sam2.config import ROOT, load_config, resolve_path


def git_revision(root: Path | None = None) -> str | None:
    root = root or ROOT
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def refs_pickle_for_dataset(dataset: str) -> str:
    """RefCOCO-g standard eval uses the Google partition; others use UNC."""
    if dataset == "refcocog":
        return "refs(google).p"
    return "refs(unc).p"


def subset_ref_ids(
    data_dir: Path,
    dataset: str = "refcoco",
    split: str = "val",
    subset_size: int = 200,
    seed: int = 42,
) -> list[int]:
    ref_file = data_dir / dataset / refs_pickle_for_dataset(dataset)
    with ref_file.open("rb") as f:
        refs = pickle.load(f)
    refs = [r for r in refs if r.get("split") == split]
    rng = random.Random(seed)
    if subset_size > 0 and subset_size < len(refs):
        refs = rng.sample(refs, subset_size)
    return [int(r["ref_id"]) for r in refs]


def subset_hash(ref_ids: list[int]) -> str:
    payload = ",".join(str(x) for x in sorted(ref_ids))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_run_manifest(
    *,
    run_name: str,
    dataset: str,
    split: str,
    seed: int,
    subset_size: int,
    ref_ids: list[int],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "dataset": dataset,
        "split": split,
        "seed": seed,
        "requested_subset_size": subset_size,
        "evaluated_n": len(ref_ids),
        "subset_ref_ids": ref_ids,
        "subset_hash": subset_hash(ref_ids),
        "config_snapshot": config,
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_run_manifest(manifest: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{manifest['run_name']}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    append_experiment_log(manifest, output_dir)
    return path


def append_experiment_log(manifest: dict[str, Any], output_dir: Path) -> None:
    log_root = output_dir.parent / "experiments"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / "RUN_LOG.jsonl"
    entry = {
        "timestamp_utc": manifest.get("timestamp_utc"),
        "run_name": manifest.get("run_name"),
        "git_revision": manifest.get("git_revision"),
        "dataset": manifest.get("dataset"),
        "split": manifest.get("split"),
        "seed": manifest.get("seed"),
        "subset_hash": manifest.get("subset_hash"),
        "evaluated_n": manifest.get("evaluated_n"),
        "metrics": manifest.get("metrics"),
        "notes": manifest.get("notes"),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def validate_eval_setup(
    data_dir: Path,
    dataset: str = "refcoco",
    split: str = "val",
    subset_size: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    from pycocotools.coco import COCO

    ref_file = data_dir / dataset / refs_pickle_for_dataset(dataset)
    with ref_file.open("rb") as f:
        all_refs = pickle.load(f)
    split_refs = [r for r in all_refs if r.get("split") == split]

    instances_path = data_dir / dataset / "instances.json"
    if not instances_path.exists():
        instances_path = data_dir / "refcoco" / "instances.json"

    coco = COCO(str(instances_path))
    ref_ids = subset_ref_ids(data_dir, dataset, split, subset_size, seed)

    rng = random.Random(seed)
    sampled = rng.sample(split_refs, subset_size) if subset_size < len(split_refs) else split_refs
    sampled_ids = [int(r["ref_id"]) for r in sampled]

    missing_images = 0
    missing_anns = 0
    for ref in sampled:
        image_path = data_dir / "train2014" / f"COCO_train2014_{ref['image_id']:012d}.jpg"
        if not image_path.exists():
            missing_images += 1
        if ref["ann_id"] not in coco.anns:
            missing_anns += 1

    return {
        "dataset": dataset,
        "split": split,
        "full_split_size": len(split_refs),
        "subset_size": subset_size,
        "seed": seed,
        "subset_hash": subset_hash(sampled_ids),
        "subset_ref_ids_match": sampled_ids == ref_ids,
        "instances_json": str(instances_path),
        "missing_images_in_subset": missing_images,
        "missing_anns_in_subset": missing_anns,
        "evaluable_in_subset": subset_size - max(missing_images, missing_anns),
        "checks_passed": missing_images == 0 and missing_anns == 0,
    }


def load_config_snapshot() -> dict[str, Any]:
    cfg_path = ROOT / "configs" / "default.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)
