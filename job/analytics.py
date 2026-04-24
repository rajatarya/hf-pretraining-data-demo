#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars>=1.0",
#   "pyarrow>=15",
#   "rich>=13",
# ]
# ///
"""Analytics payload run inside an HF Job. Mounts bucket rw at /workspace."""

from __future__ import annotations

import json
import os
from fnmatch import fnmatch
from pathlib import Path
import time

import polars as pl


def walk_file_sizes(root: str, pattern: str) -> pl.DataFrame:
    """Recursively collect (relative_path, size_bytes) for files matching `pattern`."""
    root_path = Path(root)
    rows_path: list[str] = []
    rows_size: list[int] = []
    stack = [root_path]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False) and fnmatch(entry.name, pattern):
                        rows_path.append(str(Path(entry.path).relative_to(root_path)))
                        rows_size.append(entry.stat().st_size)
        except (PermissionError, FileNotFoundError):
            continue
    return pl.DataFrame({"relative_path": rows_path, "size_bytes": rows_size})


def aggregate_by_label(meta: pl.DataFrame) -> pl.DataFrame:
    """Top labels by count, descending."""
    return (
        meta.group_by("label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )


def aggregate_by_classification(meta: pl.DataFrame) -> pl.DataFrame:
    """Counts by asset classification, descending."""
    return (
        meta.group_by("classification")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )


def curate_grasp_ready(meta: pl.DataFrame, max_mass_kg: float = 2.0) -> pl.DataFrame:
    """Hand-manipulation Props with mass ≤ `max_mass_kg` (or mass unknown).

    The dataset's `"Prop general hand manipulation"` class is the graspable
    subset by design; mass is filtered only when known, since many entries
    have null mass but are still graspable props.
    """
    mass = pl.col("mass")
    return meta.filter(
        (pl.col("classification") == "Prop general hand manipulation")
        & (mass.is_null() | (mass <= max_mass_kg))
    )


def main(workspace: str) -> int:
    workspace_path = Path(workspace)
    dataset = workspace_path / "dataset"
    csv = dataset / "physical_ai_simready_warehouse_01.csv"
    if not csv.exists():
        print(f"ERROR: dataset CSV not found at {csv} — run ingest phase first.", flush=True)
        return 1

    print(f"reading {csv}", flush=True)
    meta = pl.read_csv(str(csv))
    # Real-world CSV has some non-numeric mass rows; coerce to Float64 with
    # unparseable values as null.
    meta = meta.with_columns(pl.col("mass").cast(pl.Float64, strict=False))
    print(f"  {meta.height} rows, {len(meta.columns)} columns", flush=True)

    print(f"walking {dataset / 'Props'} for *.usd via hf-mount", flush=True)
    usd = walk_file_sizes(str(dataset / "Props"), "*.usd")
    print(f"  found {usd.height} USD files", flush=True)

    print(f"walking {dataset / 'computex_handmanip_renders'} for *.png via hf-mount", flush=True)
    png = walk_file_sizes(str(dataset / "computex_handmanip_renders"), "*.png")
    print(f"  found {png.height} thumbnails", flush=True)

    print("aggregating by label + classification, computing mass stats", flush=True)
    by_label = aggregate_by_label(meta).head(10)
    by_class = aggregate_by_classification(meta)
    mass_stats = meta["mass"].drop_nulls().describe()

    print("curating grasp-ready subset (mass <= 2kg, Prop general hand manipulation)", flush=True)
    curated = curate_grasp_ready(meta)
    print(f"  {curated.height} grasp-ready assets", flush=True)

    summary = {
        "n_assets": int(meta.height),
        "n_usd_files": int(usd.height),
        "n_thumbnails": int(png.height),
        "total_usd_bytes": int(usd["size_bytes"].sum()) if usd.height else 0,
        "total_thumbnail_bytes": int(png["size_bytes"].sum()) if png.height else 0,
        "top_labels": {r["label"]: r["count"] for r in by_label.iter_rows(named=True)},
        "by_classification": {r["classification"]: r["count"] for r in by_class.iter_rows(named=True)},
        "mass_stats": mass_stats.to_dicts(),
        "n_grasp_ready": int(curated.height),
        "xet_mount_note": f"dataset bytes were read via the bucket mount at {dataset}",
    }

    out = workspace_path / "analytics"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    joined = (
        meta.join(usd.rename({"size_bytes": "usd_size_bytes"}),
                  on="relative_path", how="left")
            .join(png.rename({"size_bytes": "thumbnail_size_bytes",
                              "relative_path": "thumbnail_path"}),
                  on="thumbnail_path", how="left")
    )
    joined.write_parquet(str(out / "assets_by_category.parquet"))
    curated.write_parquet(str(out / "training_manifest.parquet"))

    print(f"wrote {out}/summary.json")
    print(f"wrote {out}/assets_by_category.parquet ({joined.height} rows)")
    print(f"wrote {out}/training_manifest.parquet ({curated.height} rows)")

    # give mount a couple seconds to sync written files.
    time.sleep(2)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1]))
