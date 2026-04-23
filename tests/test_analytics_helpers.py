from pathlib import Path

import polars as pl
import analytics

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_metadata.csv")


def test_aggregate_by_label_returns_per_label_count():
    meta = pl.read_csv(FIXTURE)
    agg = analytics.aggregate_by_label(meta)
    labels = dict(zip(agg["label"].to_list(), agg["count"].to_list()))
    assert labels == {"book": 1, "mug": 1, "crate": 1, "paintbrush": 1, "warehouse": 1}


def test_aggregate_by_classification():
    meta = pl.read_csv(FIXTURE)
    agg = analytics.aggregate_by_classification(meta)
    result = dict(zip(agg["classification"].to_list(), agg["count"].to_list()))
    assert result["Prop general hand manipulation"] == 3
    assert result["Assembly"] == 1
    assert result["Scenario"] == 1


def test_curate_grasp_ready_filters_by_mass_and_classification():
    meta = pl.read_csv(FIXTURE)
    curated = analytics.curate_grasp_ready(meta)
    names = curated["asset_name"].to_list()
    assert "sm_book_a" in names
    assert "sm_mug_a" in names
    assert "sm_brush_a" in names
    assert "sm_crate_a" not in names
    assert "sm_warehouse_a" not in names
    assert len(curated) == 3


def test_walk_file_sizes_empty_dir_returns_empty_df(tmp_path):
    df = analytics.walk_file_sizes(str(tmp_path), "*.usd")
    assert df.height == 0
    assert set(df.columns) == {"relative_path", "size_bytes"}


def test_walk_file_sizes_finds_files(tmp_path):
    (tmp_path / "a.usd").write_bytes(b"x" * 100)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.usd").write_bytes(b"y" * 200)
    (tmp_path / "c.png").write_bytes(b"z" * 50)  # wrong extension
    df = analytics.walk_file_sizes(str(tmp_path), "*.usd").sort("relative_path")
    assert df["relative_path"].to_list() == ["a.usd", "sub/b.usd"]
    assert df["size_bytes"].to_list() == [100, 200]


def test_curate_grasp_ready_handles_string_mass_column(tmp_path):
    """Regression: real CSV can have mass as String (mixed empty/text rows)."""
    csv = tmp_path / "mixed.csv"
    csv.write_text(
        "asset_name,classification,label,mass\n"
        "a,Prop general hand manipulation,book,0.5\n"
        "b,Prop general hand manipulation,mug,\n"
        "c,Prop general hand manipulation,crate,abc\n"
        "d,Assembly,pallet,3.0\n"
    )
    meta = pl.read_csv(str(csv))
    # Simulate analytics.main() cast
    meta = meta.with_columns(pl.col("mass").cast(pl.Float64, strict=False))
    curated = analytics.curate_grasp_ready(meta)
    # 'a' qualifies: Prop class, mass 0.5 ≤ 2.0
    # 'b' qualifies: Prop class, mass is null (empty string -> null)
    # 'c' qualifies: Prop class, mass "abc" -> null (unparseable)
    # 'd' excluded: Assembly class (not hand-manipulation)
    names = set(curated["asset_name"].to_list())
    assert names == {"a", "b", "c"}, (
        "expect hand-manipulation items with null-or-light mass"
    )


def test_curate_grasp_ready_includes_null_mass_hand_manipulation(tmp_path):
    """Regression: real dataset has many hand-manipulation items with null mass."""
    csv = tmp_path / "null_mass.csv"
    csv.write_text(
        "asset_name,classification,label,mass\n"
        "book,Prop general hand manipulation,book,0.4\n"
        "mug_no_mass,Prop general hand manipulation,mug,\n"
        "heavy_crate,Assembly,crate,15.0\n"
        "light_prop_general,Prop general,sticker,0.1\n"
        "huge_prop_hand,Prop general hand manipulation,rock,50.0\n"
    )
    meta = pl.read_csv(str(csv))
    meta = meta.with_columns(pl.col("mass").cast(pl.Float64, strict=False))
    curated = analytics.curate_grasp_ready(meta)
    names = set(curated["asset_name"].to_list())
    assert names == {"book", "mug_no_mass"}, (
        "expect hand-manipulation items with null-or-light mass; "
        "Assembly and Prop-general-non-hand excluded; heavy hand-manipulation excluded"
    )
