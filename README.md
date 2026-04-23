# hf-pretraining-data-demo

A reproducible pre-training data workflow on Hugging Face:

1. **Ingest** — `hf buckets cp` any HF dataset into a Storage Bucket. Near-zero byte transfer thanks to Xet chunk-level dedupe.
2. **Analyze** — a CPU HF Job mounts the bucket read-write, walks the dataset via `hf-mount`, runs polars analytics, and writes a curated training manifest back.
3. **Mutate + re-sync** — add a derived column to the dataset's CSV, upload, and measure the delta. Xet only transfers the new bytes.

One uv script. No SDK. No download step.

## Run

```bash
uv run demo.py
```

Defaults ingest [`nvidia/PhysicalAI-SimReady-Warehouse-01`](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01) into `<your-username>/nvidia-simready`. Swap either with flags:

```bash
uv run demo.py --dataset org/other-dataset --bucket my-org/my-bucket
```

## Prereqs

```bash
curl -LsSf https://hf.co/cli/install.sh | bash    # hf CLI >= 1.11
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv
hf auth login                                      # or export HF_TOKEN=hf_...
```

## Flags

```bash
uv run demo.py --namespace <org>      # HF namespace for the Job + default bucket (default: your sole org, or username)
uv run demo.py --bucket <ns>/<name>   # override bucket (default: <namespace>/nvidia-simready)
uv run demo.py --dataset <org>/<ds>   # override dataset (default: nvidia/PhysicalAI-SimReady-Warehouse-01)
uv run demo.py --skip-ingest          # dataset already in bucket
uv run demo.py --skip-job             # skip the analytics Job
uv run demo.py --skip-mutate          # skip the CSV-mutation beat
```

If you belong to multiple orgs, you'll need to pass `--namespace <name>` to choose which namespace runs the Job.

## What the Job does

`job/analytics.py` mounts the bucket at `/workspace`, reads `dataset/*.csv` with polars, walks the mounted filesystem to stat every USD + PNG file, joins + aggregates, then writes:

- `analytics/summary.json` — per-label counts, classification breakdown, mass distribution, counts
- `analytics/assets_by_category.parquet` — full per-asset join
- `analytics/training_manifest.parquet` — curated grasp-ready subset

Change the filter in `curate_grasp_ready()` or replace the whole job payload to fit your use case.

## Tests

```bash
uv run --with pytest --with polars --with pyarrow --with rich --with "huggingface_hub>=1.11" pytest tests/ -v
```

Thirty unit tests, no network, <1s.

## Example output (NVIDIA SimReady Warehouse)

```
Phase 3 — ingest dataset → bucket
  nominal      :  13777.1 MB
  transferred  :      1.3 MB
  dedup saved  :    100.0 %
  elapsed      :     40.9 s

Phase 5 — poll (every 3s)
  job finished: status=succeeded elapsed=72.8s

Phase 7 — mutate CSV + re-sync
  nominal      :      0.3 MB
  transferred  :      0.0 MB
  dedup saved  :    100.0 %
  elapsed      :      1.4 s
```

`transferred` is what actually hit the network. 13.7 GB → 1.3 MB because Xet recognized the dataset chunks were already in CAS.

## Cleanup

```bash
./cleanup.sh                             # uses <user>/nvidia-simready
./cleanup.sh my-org/my-bucket            # or pass an explicit bucket
```

## Further reading

- [Hugging Face Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)
- [`hf-mount`](https://github.com/huggingface/hf-mount)
- [HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/cli#jobs) (+ the `hf jobs uv run` subcommand)
- [Xet storage backend](https://huggingface.co/blog/xet)

## License

Apache 2.0 — see `LICENSE`.
