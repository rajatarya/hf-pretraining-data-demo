#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface_hub>=1.11",
#   "polars>=1.0",
#   "pyarrow>=15",
#   "rich>=13",
# ]
# ///
"""HF Buckets + hf-mount + HF Jobs demo for NVIDIA Cosmos pre-training data.

Run: uv run demo.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATASET_ID = "nvidia/PhysicalAI-SimReady-Warehouse-01"
DATASET_CSV = "physical_ai_simready_warehouse_01.csv"


# ─── Types ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HFIdentity:
    """HF user + org membership."""
    user: str
    orgs: tuple[str, ...]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    """Run the 7-phase NVIDIA Buckets + Jobs demo."""
    p = argparse.ArgumentParser(description="HF Buckets + hf-mount + HF Jobs demo")
    p.add_argument("--namespace", default=None,
                   help="HF namespace for the Job + default bucket (default: your sole org if "
                        "you have exactly one, else your username; required when you have 2+ orgs)")
    p.add_argument("--bucket", default=None,
                   help="bucket name (default: <namespace>/nvidia-simready)")
    p.add_argument("--dataset", default=DEFAULT_DATASET_ID,
                   help=f"dataset repo id, e.g. 'org/name' (default: {DEFAULT_DATASET_ID})")
    p.add_argument("--poll-interval", type=float, default=3.0)
    p.add_argument("--job-timeout", type=float, default=15 * 60)
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--skip-job", action="store_true")
    p.add_argument("--skip-mutate", action="store_true")
    p.add_argument("--flavor", default="cpu-basic")
    args = p.parse_args()

    console = Console()

    def _print_bytes_panel(title: str, nominal: int, transferred: int, elapsed_ms: int) -> None:
        if transferred < 0:
            body = "(new-bytes not reported by this CLI version)"
        else:
            pct = (1 - transferred / nominal) * 100 if nominal > 0 else 0
            body = (
                f"nominal      : {nominal / 1024**2:>8.1f} MB\n"
                f"transferred  : {transferred / 1024**2:>8.1f} MB\n"
                f"dedup saved  : {pct:>8.1f} %\n"
                f"elapsed      : {elapsed_ms / 1000:>8.1f} s"
            )
        console.print(Panel(body, title=title))

    # Phase 1: preflight
    console.rule("[bold]Phase 1 — preflight")
    identity = hf_whoami_info()
    namespace = resolve_namespace(identity, args.namespace)
    bucket = args.bucket or f"{namespace}/nvidia-simready"
    console.print(
        f"user:      {identity.user}\n"
        f"orgs:      {', '.join(identity.orgs) if identity.orgs else '(none)'}\n"
        f"namespace: {namespace}\n"
        f"bucket:    {bucket}"
    )

    # Phase 2: ensure bucket
    console.rule("[bold]Phase 2 — ensure bucket")
    ensure_bucket(bucket)
    console.print(f"[green]bucket ready[/green]: hf://buckets/{bucket}")

    # Phase 3: ingest
    if not args.skip_ingest:
        console.rule("[bold]Phase 3 — ingest dataset → bucket")
        console.print(f"querying total size of dataset {args.dataset} from the Hub...")
        nominal_bytes = get_dataset_total_bytes(args.dataset)
        elapsed_ms, new_bytes, _ = run_hf_cp_capture(
            f"hf://datasets/{args.dataset}", f"hf://buckets/{bucket}/dataset/"
        )
        _print_bytes_panel("Pre-training ingest", nominal_bytes, new_bytes, elapsed_ms)
    else:
        console.print("[yellow]--skip-ingest: phase 3 skipped[/yellow]")

    # Phases 4-6: Job lifecycle
    if not args.skip_job:
        console.rule("[bold]Phase 4 — submit Job")
        script_path = str(Path(__file__).parent / "job" / "analytics.py")
        job_id = submit_job(script_path, bucket, flavor=args.flavor, namespace=namespace)
        job_url = build_job_url(namespace, job_id)
        console.print(Panel(
            f"job_id: {job_id}\nurl:    {job_url}",
            title="Job submitted — open in browser if you like",
            border_style="cyan",
        ))

        console.rule(f"[bold]Phase 5 — poll (every {args.poll_interval:.0f}s)")
        result = poll_job(job_id, args.poll_interval, args.job_timeout, inspect_job_status)
        console.print(f"job finished: status={result.status} elapsed={result.elapsed_s:.1f}s")
        if result.status != "succeeded":
            console.print(f"[red]Job did not succeed. URL: {job_url}[/red]")
            if result.status == "failed":
                subprocess.run(["hf", "jobs", "logs", job_id])
                return 2
            if result.status == "timeout":
                return 3
            if result.status == "interrupted":
                console.print(f"[yellow]Job still running at {job_url}[/yellow]")
                return 0
            return 1

        # Phase 6: download the summary (with upfront wait for bucket propagation)
        console.rule("[bold]Phase 6 — fetch summary")
        console.print("waiting ~90s for Job outputs to propagate, then downloading...")
        summary_path = "/tmp/nvidia-simready-summary.json"
        ok = download_bucket_file(
            f"hf://buckets/{bucket}/analytics/summary.json", summary_path
        )
        if not ok:
            console.print(
                f"[yellow]Summary not yet available. The Job succeeded, but the\n"
                f"bucket index may still be catching up. Try manually:\n"
                f"  hf buckets cp hf://buckets/{bucket}/analytics/summary.json -[/yellow]"
            )
        else:
            with open(summary_path) as f:
                summary = json.load(f)
            table = Table(title="analytics/summary.json")
            for k, v in summary.items():
                table.add_row(str(k), json.dumps(v, default=str)[:200])
            console.print(table)
            subprocess.run(["hf", "buckets", "list", f"{bucket}/analytics", "-h", "-R"])
    else:
        console.print("[yellow]--skip-job: phases 4-6 skipped[/yellow]")

    # Phase 7: mutate + resync
    if not args.skip_mutate:
        console.rule("[bold]Phase 7 — mutate CSV + re-sync")
        original = Path("/tmp/original.csv")
        annotated = Path("/tmp/annotated.csv")
        subprocess.run(
            ["hf", "buckets", "cp",
             f"hf://buckets/{bucket}/dataset/{DATASET_CSV}", str(original)],
            check=True,
        )
        mutate_csv_add_grasp_score(str(original), str(annotated))
        full_bytes = annotated.stat().st_size
        elapsed_ms, new_bytes, _ = run_hf_cp_capture(
            str(annotated), f"hf://buckets/{bucket}/dataset/{DATASET_CSV}"
        )
        _print_bytes_panel("Mutate + re-sync", full_bytes, new_bytes, elapsed_ms)
    else:
        console.print("[yellow]--skip-mutate: phase 7 skipped[/yellow]")

    console.rule("[bold green]Done")
    console.print(
        f"Outputs:\n"
        f"  hf://buckets/{bucket}/dataset/         # ingested + annotated dataset\n"
        f"  hf://buckets/{bucket}/analytics/       # Job outputs\n"
        "To clean up: ./cleanup.sh"
    )
    return 0


# ─── Helpers ──────────────────────────────────────────────────────────────────


# ─── CLI wrappers ─────────────────────────────────────────────────────────────

class HFCliError(RuntimeError):
    """Raised when an `hf` CLI subprocess exits non-zero."""


def hf_whoami_info() -> HFIdentity:
    """Return the authenticated HF user + orgs. Raises HFCliError if not logged in."""
    r = subprocess.run(
        ["hf", "auth", "whoami"], capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        raise HFCliError(f"hf auth whoami failed: {r.stderr.strip()}")
    # Output varies by TTY:
    #   non-TTY: "user=<name> orgs=<csv>"
    #   TTY:     "✓ Logged in\n  user: <name>\n  orgs: <csv>"
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    clean = ansi_re.sub("", r.stdout)
    user: str | None = None
    orgs_csv: str = ""
    for line in clean.splitlines():
        line = line.strip()
        for sep in ("=", ":"):
            if line.startswith(f"user{sep}"):
                user = line.split(sep, 1)[1].strip().split()[0]
            if line.startswith(f"orgs{sep}"):
                orgs_csv = line.split(sep, 1)[1].strip()
        # non-TTY form: "user=foo orgs=bar,baz" — parse both tokens on one line
        for token in line.split():
            if token.startswith("user=") and user is None:
                user = token.split("=", 1)[1]
            if token.startswith("orgs="):
                orgs_csv = token.split("=", 1)[1]
    if user is None:
        raise HFCliError(f"hf auth whoami output not understood: {r.stdout!r}")
    orgs = tuple(o.strip() for o in orgs_csv.split(",") if o.strip()) if orgs_csv else ()
    return HFIdentity(user=user, orgs=orgs)


def hf_whoami() -> str:
    """Return the authenticated HF username (no orgs)."""
    return hf_whoami_info().user


def resolve_namespace(identity: HFIdentity, explicit: str | None) -> str:
    """Pick the HF namespace for Jobs + bucket. Explicit flag wins. Otherwise
    use the sole org if the user has exactly one; fall back to the username
    with zero orgs; error with 2+ orgs to force the user to disambiguate."""
    if explicit:
        return explicit
    if len(identity.orgs) == 0:
        return identity.user
    if len(identity.orgs) == 1:
        return identity.orgs[0]
    orgs_list = ", ".join(identity.orgs)
    raise HFCliError(
        f"User '{identity.user}' belongs to {len(identity.orgs)} orgs ({orgs_list}). "
        f"Pass --namespace <name> to pick one (or --namespace {identity.user} for your personal namespace)."
    )


def get_dataset_total_bytes(dataset_id: str) -> int:
    """Sum of all file sizes in the dataset repo. Requires internet; cheap call."""
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(dataset_id, files_metadata=True)
    return sum((s.size or 0) for s in info.siblings)


def ensure_bucket(bucket: str) -> None:
    """Create the bucket private; silently no-op if it already exists."""
    subprocess.run(
        ["hf", "buckets", "create", bucket, "--private"],
        capture_output=True, text=True, check=False,
    )


def run_hf_cp_capture(src: str, dst: str) -> tuple[int, int, str]:
    """Run `hf buckets cp SRC DST`, stream stderr to terminal, return
    (elapsed_ms, new_bytes, full_stderr). new_bytes is -1 if unparsed."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as log:
        log_path = log.name
    cmd = f"hf buckets cp {src} {dst} 2> >(tee {log_path} >&2)"
    t0 = time.monotonic()
    proc = subprocess.run(cmd, shell=True, executable="/bin/bash")
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    stderr_text = Path(log_path).read_text()
    Path(log_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise HFCliError(f"hf buckets cp failed (exit {proc.returncode}): {src} -> {dst}")
    return elapsed_ms, parse_new_data_bytes(stderr_text), stderr_text


def submit_job(
    script_path: str,
    bucket: str,
    flavor: str = "cpu-basic",
    namespace: str | None = None,
) -> str:
    """Submit `script_path` to HF Jobs via `hf jobs uv run --detach`. Returns job_id."""
    cmd = ["hf", "jobs", "uv", "run", "--detach", "--flavor", flavor]
    if namespace:
        cmd.extend(["--namespace", namespace])
    cmd.extend([
        "-v", f"hf://buckets/{bucket}:/workspace",
        script_path, "/workspace",
    ])
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise HFCliError(f"hf jobs uv run failed: {r.stderr.strip()}")
    # hf jobs uv run --detach prints:
    #   Job started with ID: <id>
    #   View at: https://huggingface.co/jobs/<ns>/<id>
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("Job started with ID:"):
            return line.split(":", 1)[1].strip()
    raise HFCliError(f"hf jobs uv run did not print a job_id. stdout={r.stdout!r}")


def inspect_job_status(job_id: str) -> str:
    """Return the Job's status in canonical form ('pending'|'running'|'succeeded'|'failed'|'cancelled').
    Maps HF CLI's stage vocabulary to poll_job's expected vocabulary."""
    r = subprocess.run(
        ["hf", "jobs", "inspect", job_id],
        capture_output=True, text=True, check=False,
    )
    if r.returncode != 0:
        raise HFCliError(f"hf jobs inspect failed: {r.stderr.strip()}")
    data = json.loads(r.stdout)
    obj = data[0] if isinstance(data, list) else data
    status = obj.get("status", {})
    stage = status.get("stage") if isinstance(status, dict) else status
    stage_lower = str(stage).lower() if stage else "unknown"
    # Map HF CLI stage to canonical status for poll_job
    return _HF_STAGE_TO_STATUS.get(stage_lower, stage_lower)


def download_bucket_file(uri: str, dst: str, initial_wait_s: float = 90.0,
                        max_retries: int = 10, retry_interval_s: float = 3.0) -> bool:
    """Download a bucket file to `dst`, tolerating the post-Job propagation
    lag. Sleeps `initial_wait_s` first (bucket index catches up), then retries
    `hf buckets cp uri dst` until it succeeds. Returns True on success,
    False if the file still can't be fetched after all retries."""
    import os
    time.sleep(initial_wait_s)
    for _ in range(max_retries):
        # Ensure no stale dst from a prior attempt.
        try:
            os.unlink(dst)
        except FileNotFoundError:
            pass
        r = subprocess.run(
            ["hf", "buckets", "cp", uri, dst],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        time.sleep(retry_interval_s)
    return False


# HF CLI uses stage strings like "COMPLETED" / "ERROR"; poll_job expects
# "succeeded" / "running" / "failed" / etc. Map between them.
_HF_STAGE_TO_STATUS = {
    "completed": "succeeded",
    "running": "running",
    "pending": "pending",
    "error": "failed",
    "failed": "failed",
    "cancelled": "cancelled",
    "canceled": "cancelled",   # alternate spelling
    "deleted": "cancelled",
}


# ─── Progress parsing ─────────────────────────────────────────────────────────

_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
# hf CLI uses both upper- ("MB", "GB") and lowercase-prefix forms ("kB"), so
# allow both in the regex and normalize the captured unit before lookup.
_NEW_DATA_RE = re.compile(r'([\d.]+)([KkMmGgTt]?B)\s*/\s*([\d.]+)([KkMmGgTt]?B)')


def parse_new_data_bytes(stderr_text: str) -> int:
    """Parse `hf buckets cp` stderr for the last "New Data Upload" line and
    return its total-new bytes. Returns -1 if no match."""
    last_match = None
    for line in stderr_text.splitlines():
        if "New Data Upload" not in line:
            continue
        m = _NEW_DATA_RE.search(line)
        if m:
            last_match = m
    if last_match is None:
        return -1
    unit = last_match.group(4).upper()
    return int(float(last_match.group(3)) * _UNITS.get(unit, 1))


# ─── Job state ────────────────────────────────────────────────────────────────

JobStatus = Literal["pending", "running", "succeeded", "failed", "cancelled", "timeout", "interrupted"]


@dataclass
class JobResult:
    """Terminal result returned by poll_job."""
    status: JobStatus
    elapsed_s: float


def poll_job(
    job_id: str,
    poll_interval: float,
    timeout: float,
    inspector: Callable[[str], str],
) -> JobResult:
    """Block until the Job reaches a terminal state, times out, or is
    interrupted. `inspector(job_id)` must return 'pending' | 'running' |
    'succeeded' | 'failed' | 'cancelled'. Ctrl-C yields 'interrupted' and
    leaves the remote Job running."""
    TERMINAL_OK = {"succeeded"}
    TERMINAL_BAD = {"failed", "cancelled"}
    start = time.monotonic()
    while True:
        try:
            status = inspector(job_id)
        except KeyboardInterrupt:
            return JobResult(status="interrupted", elapsed_s=time.monotonic() - start)

        if status in TERMINAL_OK or status in TERMINAL_BAD:
            return JobResult(status=status, elapsed_s=time.monotonic() - start)

        now = time.monotonic()
        if (now - start) >= timeout:
            return JobResult(status="timeout", elapsed_s=now - start)

        time.sleep(poll_interval)


def build_job_url(namespace: str, job_id: str) -> str:
    """Return the Hub URL for a Job."""
    return f"https://huggingface.co/jobs/{namespace}/{job_id}"


# ─── Data work ────────────────────────────────────────────────────────────────

def mutate_csv_add_grasp_score(src_csv: str, dst_csv: str) -> None:
    """Read `src_csv`, add `grasp_score = 1/(1+mass)` (nulls→1.0), write to `dst_csv`."""
    import polars as pl
    df = pl.read_csv(src_csv)
    df = df.with_columns(pl.col("mass").cast(pl.Float64, strict=False))
    df = df.with_columns(
        (1.0 / (1.0 + pl.col("mass").fill_null(1.0))).alias("grasp_score")
    )
    df.write_csv(dst_csv)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HFCliError as e:
        Console(stderr=True).print(f"[red]error:[/red] {e}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        Console(stderr=True).print("[yellow]interrupted[/yellow]")
        raise SystemExit(130)
