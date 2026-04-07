"""Tests for PTM record validation, position augmentation, prediction output,
and inference error handling.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from conftest import FakeStream

import ptmdiscoverer.inference as inf
from ptmdiscoverer.utils import augment_protein_positions, validate_ptm_records


def _make_delta_mass_df(
    peptides: list[str],
    best_positions: list[str],
    positions_in_protein: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Peptide": peptides,
            "Best.Positions": best_positions,
            "Positions.In.Protein": positions_in_protein,
        }
    )


# --- augment_protein_positions ---


@pytest.mark.unit
def test_augment_exact_match() -> None:
    """Exact match: one unique (peptide, local_idx) maps to one absolute position."""
    df = _make_delta_mass_df(["PEPTIDEK"], ["3"], [42.0])
    records = [
        {
            "PEPTIDE": "PEPTIDEK",
            "PTM.Best.Positions": ["3"],
            "PTM.Protein.Position": -1,
        }
    ]
    augment_protein_positions(records, df)
    assert records[0]["PTM.Protein.Position"] == 42


@pytest.mark.unit
def test_augment_ambiguous_multiple_positions() -> None:
    """Same (peptide, site) mapping to two different absolute positions is ambiguous."""
    df = _make_delta_mass_df(
        ["PEPTIDEK", "PEPTIDEK"],
        ["3", "3"],
        [42.0, 99.0],
    )
    records = [
        {
            "PEPTIDE": "PEPTIDEK",
            "PTM.Best.Positions": ["3"],
            "PTM.Protein.Position": -1,
        }
    ]
    augment_protein_positions(records, df)
    assert records[0]["PTM.Protein.Position"] == -1


@pytest.mark.unit
def test_augment_no_match_position_not_found() -> None:
    """Peptide not in delta-mass table leaves position unchanged."""
    df = _make_delta_mass_df(["OTHERPEP"], ["5"], [100.0])
    records = [
        {
            "PEPTIDE": "NOPEPTIDE",
            "PTM.Best.Positions": ["3"],
            "PTM.Protein.Position": -1,
        }
    ]
    augment_protein_positions(records, df)
    assert records[0]["PTM.Protein.Position"] == -1


@pytest.mark.unit
def test_augment_skips_already_valid_position() -> None:
    """Records with a valid (positive) position are not overwritten."""
    df = _make_delta_mass_df(["PEPTIDEK"], ["3"], [42.0])
    records = [
        {
            "PEPTIDE": "PEPTIDEK",
            "PTM.Best.Positions": ["3"],
            "PTM.Protein.Position": 99,
        }
    ]
    augment_protein_positions(records, df)
    assert records[0]["PTM.Protein.Position"] == 99


@pytest.mark.unit
def test_augment_peptide_only_fallback() -> None:
    """Peptide maps to exactly one absolute position across all sites (fallback)."""
    df = _make_delta_mass_df(["PEPTIDEK"], ["7"], [55.0])
    records = [
        {
            "PEPTIDE": "PEPTIDEK",
            "PTM.Best.Positions": ["3"],
            "PTM.Protein.Position": -1,
        }
    ]
    augment_protein_positions(records, df)
    assert records[0]["PTM.Protein.Position"] == 55


@pytest.mark.unit
def test_augment_empty_df_is_noop() -> None:
    """Empty delta-mass DataFrame produces no changes."""
    df = pd.DataFrame()
    records = [{"PEPTIDE": "ABC", "PTM.Best.Positions": ["1"], "PTM.Protein.Position": -1}]
    augment_protein_positions(records, df)
    assert records[0]["PTM.Protein.Position"] == -1


# --- validate_ptm_records ---


@pytest.mark.unit
def test_validate_ptm_records_rejects_missing_fields() -> None:
    """Records missing required fields are dropped."""
    records = [
        {
            "PEPTIDE": "ABC",
            "PTM.DeltaMass": 15.2,
            "PTM.Best.Positions": ["1"],
            "PTM.NAME": "Phosphorylation",
            "PTM.Impact.In.Enzyme.Function": "Unknown",
            "PTM.Protein.Position": 42,
        },
        {
            "PTM.DeltaMass": 15.2,
            "PTM.Best.Positions": ["1"],
            "PTM.NAME": "Mod",
            "PTM.Impact.In.Enzyme.Function": "?",
            "PTM.Protein.Position": 10,
        },
        {
            "PEPTIDE": "DEF",
            "PTM.DeltaMass": 15.2,
            "PTM.Best.Positions": ["1"],
            "PTM.Impact.In.Enzyme.Function": "?",
            "PTM.Protein.Position": 5,
        },
    ]
    valid = validate_ptm_records(records)
    assert len(valid) == 1
    assert valid[0]["PEPTIDE"] == "ABC"


@pytest.mark.unit
def test_validate_ptm_records_rejects_wrong_types() -> None:
    """Records with wrong field types are dropped."""
    records = [
        {
            "PEPTIDE": 123,
            "PTM.DeltaMass": 15.2,
            "PTM.Best.Positions": ["1"],
            "PTM.NAME": "Mod",
            "PTM.Impact.In.Enzyme.Function": "?",
            "PTM.Protein.Position": 42,
        },
        {
            "PEPTIDE": "ABC",
            "PTM.DeltaMass": "not_a_number",
            "PTM.Best.Positions": ["1"],
            "PTM.NAME": "Mod",
            "PTM.Impact.In.Enzyme.Function": "?",
            "PTM.Protein.Position": 42,
        },
    ]
    valid = validate_ptm_records(records)
    assert len(valid) == 0


@pytest.mark.unit
def test_validate_ptm_records_all_valid() -> None:
    """All valid records pass through unchanged."""
    records = [
        {
            "PEPTIDE": "ABC",
            "PTM.DeltaMass": 15.2,
            "PTM.Best.Positions": ["1"],
            "PTM.NAME": "Phosphorylation",
            "PTM.Impact.In.Enzyme.Function": "Activating",
            "PTM.Protein.Position": 42,
        },
        {
            "PEPTIDE": "DEF",
            "PTM.DeltaMass": 79,
            "PTM.Best.Positions": ["3", "5"],
            "PTM.NAME": "Methylation",
            "PTM.Impact.In.Enzyme.Function": "Unknown",
            "PTM.Protein.Position": 100,
        },
    ]
    valid = validate_ptm_records(records)
    assert len(valid) == 2


# --- _write_prediction_files ---


@pytest.mark.unit
def test_write_prediction_files_drops_invalid_positions(tmp_path: Path) -> None:
    """Records with non-positive PTM.Protein.Position are dropped before writing."""
    records: list[dict[str, Any]] = [
        {"PEPTIDE": "ABC", "PTM.Protein.Position": 42},
        {"PEPTIDE": "DEF", "PTM.Protein.Position": -1},
        {"PEPTIDE": "GHI", "PTM.Protein.Position": "bad"},
        {"PEPTIDE": "JKL", "PTM.Protein.Position": 0},
    ]
    pred_base = str(tmp_path / "pred")
    kept = inf._write_prediction_files(records, pred_base)
    assert len(kept) == 1
    assert kept[0]["PEPTIDE"] == "ABC"
    assert (tmp_path / "pred.json").exists()
    assert (tmp_path / "pred.csv").exists()


@pytest.mark.unit
def test_write_prediction_files_all_invalid_no_files(tmp_path: Path) -> None:
    """When all records are invalid, no files are written."""
    records: list[dict[str, Any]] = [
        {"PEPTIDE": "ABC", "PTM.Protein.Position": -1},
        {"PEPTIDE": "DEF", "PTM.Protein.Position": -1},
    ]
    pred_base = str(tmp_path / "pred")
    kept = inf._write_prediction_files(records, pred_base)
    assert kept == []
    assert not (tmp_path / "pred.json").exists()
    assert not (tmp_path / "pred.csv").exists()


@pytest.mark.unit
def test_write_prediction_files_mixed_invalid(tmp_path: Path) -> None:
    """Mixed invalid positions (-1 and non-numeric) are all removed."""
    records: list[dict[str, Any]] = [
        {"PEPTIDE": "A", "PTM.Protein.Position": -1},
        {"PEPTIDE": "B", "PTM.Protein.Position": "bad"},
        {"PEPTIDE": "C", "PTM.Protein.Position": 5},
    ]
    pred_base = str(tmp_path / "pred")
    kept = inf._write_prediction_files(records, pred_base)
    assert len(kept) == 1
    assert kept[0]["PEPTIDE"] == "C"


# --- Fake study helper for integration tests ---


def _make_fake_study(tmp_path: Path) -> tuple[Path, object]:
    """Create a minimal study folder and fake LLM client for inference tests."""
    base = tmp_path / "study"
    base.mkdir()
    (base / "output").mkdir()
    (base / "dataset_labels.csv").write_text("MSRUNID\n1\n", encoding="utf-8")
    psm_dir = base / "Results_MSFragger" / "1"
    psm_dir.mkdir(parents=True)
    psm_content = (
        "Spectrum\tPeptide\tCharge\tCalculated Peptide Mass\tCalculated M/Z\t"
        "Delta Mass\tBest Positions\tNumber Best Positions\t"
        "Ions Best Position\tIons Second Best Position\t"
        "Score Best Position\tScore Second Best Position\t"
        "Probability\tNumber of Missed Cleavages\tProtein ID\tRetention\tIs Unique\n"
        "s1\tPEPTIDEK\t2\t1000\t500\t15.2\t1\t1\t10\t5\t60\t10\t0.95\t0\tP12345\t600\tTrue\n"
    )
    (psm_dir / "psm.tsv").write_text(psm_content, encoding="utf-8")
    (base / "base_prompt.txt").write_text("Base prompt", encoding="utf-8")
    metadata = {
        "version": 1,
        "proteins": [{"uniprot_id": "P12345", "name": "Prot", "delta_mass_file": ""}],
    }
    (base / "protein_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    fenced_json = (
        '```json\n[{"PEPTIDE":"PEPTIDEK","PTM.DeltaMass":15.2,'
        '"PTM.Best.Positions":["1"],"PTM.NAME":"Mod",'
        '"PTM.Impact.In.Enzyme.Function":"?","PTM.Protein.Position":5}]\n```'
    )

    class FakeChoices:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream(fenced_json)))

    return base, FakeClient()


# --- run_inference ---


@pytest.mark.integration
def test_run_inference_returns_result_dict(tmp_path: Path) -> None:
    """run_inference() returns a dict with run metadata."""
    base, client = _make_fake_study(tmp_path)
    result = inf.run_inference(
        base_data_folder=str(base),
        client=client,
        timestamp="2024-01-01-00-00-00-000",
    )
    assert isinstance(result, dict)
    assert "run_dir" in result
    assert "timestamp" in result
    assert result["proteins_processed"] == 1
    assert result["total_ptm_records"] >= 1


@pytest.mark.integration
def test_inference_uses_logging_not_print(tmp_path: Path, capsys) -> None:
    """inference.py uses logging instead of print()."""
    base, client = _make_fake_study(tmp_path)
    inf.run_inference(
        base_data_folder=str(base),
        client=client,
        timestamp="2024-01-01-00-00-00-000",
    )
    captured = capsys.readouterr()
    assert captured.out == "", f"inference.py should not print to stdout, but got:\n{captured.out}"


@pytest.mark.unit
def test_run_inference_missing_api_key(tmp_path: Path) -> None:
    """run_inference raises SystemExit when no client and no OPENAI_API_KEY."""
    base = tmp_path / "study"
    base.mkdir()
    (base / "base_prompt.txt").write_text("p", encoding="utf-8")
    (base / "protein_metadata.json").write_text('{"version":1,"proteins":[]}', encoding="utf-8")

    import os as _os

    old = _os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(SystemExit):
            inf.run_inference(base_data_folder=str(base))
    finally:
        if old is not None:
            _os.environ["OPENAI_API_KEY"] = old


@pytest.mark.unit
def test_run_inference_missing_dataset_labels(tmp_path: Path) -> None:
    """run_inference handles missing dataset_labels.csv gracefully."""
    base, client = _make_fake_study(tmp_path)
    (base / "dataset_labels.csv").unlink()
    result = inf.run_inference(
        base_data_folder=str(base),
        client=client,
        timestamp="2024-01-01-00-00-00-000",
    )
    assert isinstance(result, dict)


# --- process_protein error paths ---


@pytest.mark.unit
def test_process_protein_failed_csv_read(tmp_path: Path) -> None:
    """process_protein handles unreadable delta-mass file gracefully."""
    bad_csv = tmp_path / "bad.csv"
    bad_csv.mkdir()
    protein = {"uniprot_id": "P999", "name": "Prot", "delta_mass_file": str(bad_csv)}

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream()))

    ctx = inf.InferenceContext(
        client=DummyClient(),
        base_prompt="BASE",
        model="m",
        timestamp="T",
        base_data_folder=str(tmp_path),
        run_dir=str(tmp_path),
        log_handle=io.StringIO(),
        msrun_table=pd.DataFrame(),
    )
    changed, records = inf.process_protein(ctx, protein)
    assert changed is False
    assert records == []


@pytest.mark.unit
def test_process_protein_no_json_block(tmp_path: Path) -> None:
    """process_protein handles LLM returning no JSON block."""
    delta_path = tmp_path / "P999_delta-masses.csv"
    pd.DataFrame({"Peptide": ["ABC"], "Delta.Mass": [12.3]}).to_csv(delta_path, index=False)
    protein = {"uniprot_id": "P999", "name": "Prot", "delta_mass_file": str(delta_path)}

    class FakeChoices:
        def __init__(self, content):
            self.message = SimpleNamespace(content=content)

    class DummyClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream("No JSON here")))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ctx = inf.InferenceContext(
        client=DummyClient(),
        base_prompt="BASE",
        model="m",
        timestamp="T",
        base_data_folder=str(tmp_path),
        run_dir=str(run_dir),
        log_handle=io.StringIO(),
        msrun_table=pd.DataFrame(),
    )
    changed, records = inf.process_protein(ctx, protein)
    assert records == []
    assert list(run_dir.glob("output_P999_*.html"))
