import demo


def test_parse_new_data_bytes_megabytes():
    stderr = (
        "Computing xorbs: 100%|#####| 2/2 [00:00<00:00]\n"
        "New Data Upload  : 100%|#####|  1.05MB / 1.05MB\n"
    )
    assert demo.parse_new_data_bytes(stderr) == int(1.05 * 1024 * 1024)


def test_parse_new_data_bytes_zero_bytes():
    stderr = "New Data Upload  : |#####|  0.00B / 0.00B\n"
    assert demo.parse_new_data_bytes(stderr) == 0


def test_parse_new_data_bytes_missing_returns_negative_one():
    assert demo.parse_new_data_bytes("nothing relevant here") == -1


def test_parse_new_data_bytes_uses_last_match():
    stderr = (
        "New Data Upload  : 50%|##|  0.10MB / 1.05MB\n"
        "New Data Upload  : 100%|#####|  1.05MB / 1.05MB\n"
    )
    assert demo.parse_new_data_bytes(stderr) == int(1.05 * 1024 * 1024)


def test_build_job_url_with_namespace():
    assert (
        demo.build_job_url("rajatarya", "abc123def")
        == "https://huggingface.co/jobs/rajatarya/abc123def"
    )


import pytest


def _inspector_from_sequence(statuses):
    """Returns an inspector callable that yields one status per call."""
    it = iter(statuses)

    def inspect(_job_id):
        return next(it)
    return inspect


def test_poll_job_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    inspector = _inspector_from_sequence(["pending", "running", "succeeded"])
    result = demo.poll_job(
        job_id="abc", poll_interval=0.01, timeout=30, inspector=inspector
    )
    assert result.status == "succeeded"
    assert result.elapsed_s >= 0


def test_poll_job_fails(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    inspector = _inspector_from_sequence(["running", "failed"])
    result = demo.poll_job(
        job_id="abc", poll_interval=0.01, timeout=30, inspector=inspector
    )
    assert result.status == "failed"


def test_poll_job_cancelled(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    inspector = _inspector_from_sequence(["cancelled"])
    result = demo.poll_job(
        job_id="abc", poll_interval=0.01, timeout=30, inspector=inspector
    )
    assert result.status == "cancelled"


def test_poll_job_times_out(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monotonic_vals = iter([0.0, 0.1, 0.2, 99.0])
    monkeypatch.setattr("time.monotonic", lambda: next(monotonic_vals))
    inspector = lambda _j: "running"
    result = demo.poll_job(
        job_id="abc", poll_interval=0.01, timeout=1.0, inspector=inspector
    )
    assert result.status == "timeout"


def test_poll_job_keyboard_interrupt_returns_interrupted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)

    def inspect(_j):
        raise KeyboardInterrupt

    result = demo.poll_job(
        job_id="abc", poll_interval=0.01, timeout=30, inspector=inspect
    )
    assert result.status == "interrupted"


def test_mutate_csv_add_grasp_score(tmp_path):
    import polars as pl
    src = tmp_path / "in.csv"
    src.write_text(
        "asset_name,mass\n"
        "a,0.5\n"
        "b,\n"
        "c,4.0\n"
    )
    dst = tmp_path / "out.csv"

    demo.mutate_csv_add_grasp_score(str(src), str(dst))

    df = pl.read_csv(str(dst))
    assert df.columns == ["asset_name", "mass", "grasp_score"]
    scores = df["grasp_score"].to_list()
    assert abs(scores[0] - (1.0 / 1.5)) < 1e-6   # mass=0.5
    assert abs(scores[1] - (1.0 / 2.0)) < 1e-6   # null -> 1.0
    assert abs(scores[2] - (1.0 / 5.0)) < 1e-6   # mass=4.0


def test_hf_stage_mapping_maps_completed_to_succeeded():
    assert demo._HF_STAGE_TO_STATUS["completed"] == "succeeded"


def test_hf_stage_mapping_covers_running_and_failed():
    assert demo._HF_STAGE_TO_STATUS["running"] == "running"
    assert demo._HF_STAGE_TO_STATUS["error"] == "failed"
    assert demo._HF_STAGE_TO_STATUS["cancelled"] == "cancelled"


def test_hf_whoami_parses_user_token(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "user=rajatarya orgs=huggingface,xet-team\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    assert demo.hf_whoami() == "rajatarya"


def test_hf_whoami_raises_on_unparseable(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "no user here\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    with pytest.raises(demo.HFCliError):
        demo.hf_whoami()


def test_submit_job_parses_job_id_from_started_line(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = (
            "UserWarning: 'HfApi.run_uv_job' is experimental...\n"
            "Job started with ID: 69e95d4bd2fd2eb837d76e8d\n"
            "View at: https://huggingface.co/jobs/rajatarya/69e95d4bd2fd2eb837d76e8d\n"
        )
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    assert demo.submit_job("/tmp/x.py", "ns/bkt") == "69e95d4bd2fd2eb837d76e8d"


def test_submit_job_raises_when_no_job_id(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "Nothing useful here\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    with pytest.raises(demo.HFCliError):
        demo.submit_job("/tmp/x.py", "ns/bkt")


def test_mutate_csv_add_grasp_score_handles_string_mass(tmp_path):
    """Regression: real CSV has mixed numeric/non-numeric mass values."""
    import polars as pl
    src = tmp_path / "in.csv"
    src.write_text(
        "asset_name,mass\n"
        "a,0.5\n"
        "b,\n"
        "c,abc\n"     # unparseable
        "d,4.0\n"
    )
    dst = tmp_path / "out.csv"
    demo.mutate_csv_add_grasp_score(str(src), str(dst))
    df = pl.read_csv(str(dst))
    scores = df["grasp_score"].to_list()
    # a: mass=0.5 → 1/1.5
    # b,c: null-or-unparseable → 1.0 → score 1/2
    # d: mass=4.0 → 1/5
    assert abs(scores[0] - (1.0 / 1.5)) < 1e-6
    assert abs(scores[1] - 0.5) < 1e-6
    assert abs(scores[2] - 0.5) < 1e-6
    assert abs(scores[3] - (1.0 / 5.0)) < 1e-6


def test_parse_new_data_bytes_lowercase_kilobytes():
    """hf CLI uses lowercase kB for small files; must match."""
    stderr = (
        "New Data Upload  : |          |  0.00B /  0.00B            \n"
        "New Data Upload  : 100%|######|  332kB / 332kB,  831kB/s    \n"
    )
    assert demo.parse_new_data_bytes(stderr) == int(332 * 1024)


def test_parse_new_data_bytes_mixed_case_units():
    """Mixed MB / kB / GB output — last match wins, unit case-insensitive."""
    stderr = (
        "New Data Upload  : 10%|#|   50MB / 500MB\n"
        "New Data Upload  : 100%|####| 500MB / 500MB\n"
    )
    assert demo.parse_new_data_bytes(stderr) == int(500 * 1024**2)


def test_hf_whoami_parses_tty_format(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "\x1b[32m✓ Logged in\x1b[0m\n  user: rajatarya\n  orgs: huggingface,xet-team\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    assert demo.hf_whoami() == "rajatarya"


def test_hf_whoami_parses_tty_without_ansi(monkeypatch):
    """Same TTY shape but with no ANSI codes (e.g. NO_COLOR env set)."""
    class FakeProc:
        returncode = 0
        stdout = "Logged in\n  user: rajatarya\n  orgs: huggingface,xet-team\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    assert demo.hf_whoami() == "rajatarya"


def test_get_dataset_total_bytes_sums_siblings(monkeypatch):
    class FakeSibling:
        def __init__(self, size):
            self.size = size

    class FakeInfo:
        siblings = [FakeSibling(100), FakeSibling(200), FakeSibling(None), FakeSibling(50)]

    class FakeApi:
        def dataset_info(self, repo_id, files_metadata):
            assert repo_id == "some/dataset"
            assert files_metadata is True
            return FakeInfo()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: FakeApi())
    assert demo.get_dataset_total_bytes("some/dataset") == 350  # None → 0


def test_hf_whoami_info_parses_user_and_orgs_nontty(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "user=rajatarya orgs=huggingface,xet-team\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    info = demo.hf_whoami_info()
    assert info.user == "rajatarya"
    assert info.orgs == ("huggingface", "xet-team")


def test_hf_whoami_info_parses_user_and_orgs_tty(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "\x1b[32m✓ Logged in\x1b[0m\n  user: rajatarya\n  orgs: huggingface,xet-team\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    info = demo.hf_whoami_info()
    assert info.user == "rajatarya"
    assert info.orgs == ("huggingface", "xet-team")


def test_hf_whoami_info_no_orgs(monkeypatch):
    class FakeProc:
        returncode = 0
        stdout = "user=solo_user\n"
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeProc())
    info = demo.hf_whoami_info()
    assert info.user == "solo_user"
    assert info.orgs == ()


def test_resolve_namespace_explicit_wins():
    ident = demo.HFIdentity(user="u", orgs=("a", "b", "c"))
    assert demo.resolve_namespace(ident, "myorg") == "myorg"


def test_resolve_namespace_single_org_used_as_default():
    ident = demo.HFIdentity(user="u", orgs=("myorg",))
    assert demo.resolve_namespace(ident, None) == "myorg"


def test_resolve_namespace_no_orgs_falls_back_to_user():
    ident = demo.HFIdentity(user="u", orgs=())
    assert demo.resolve_namespace(ident, None) == "u"


def test_resolve_namespace_multi_orgs_raises(monkeypatch):
    import pytest
    ident = demo.HFIdentity(user="u", orgs=("a", "b"))
    with pytest.raises(demo.HFCliError) as exc:
        demo.resolve_namespace(ident, None)
    msg = str(exc.value)
    assert "2 orgs" in msg
    assert "a" in msg and "b" in msg
    assert "--namespace" in msg
