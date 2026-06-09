"""Targeted unit tests that cover specific branches missed by the existing suite.

Each test is focused on a single uncovered code path identified from the
coverage report.  No LLM calls are made.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# reporting — _json_default, resolve_analysis_input, load_run_predictions,
#              load_aggregated_predictions
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_json_default_path_returns_string() -> None:
    from ptmdiscoverer.reporting import _json_default

    assert _json_default(Path("/some/path")) == "/some/path"


@pytest.mark.unit
def test_json_default_raises_for_unserializable() -> None:
    from ptmdiscoverer.reporting import _json_default

    with pytest.raises(TypeError, match="not JSON serializable"):
        _json_default(object())


@pytest.mark.unit
def test_json_default_item_method(monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers the hasattr(value, 'item') branch via a lightweight mock."""
    from ptmdiscoverer.reporting import _json_default

    obj = SimpleNamespace(item=lambda: 42)
    assert _json_default(obj) == 42


@pytest.mark.unit
def test_resolve_analysis_input_run_dir_not_found(tmp_path: Path) -> None:
    from ptmdiscoverer.reporting import resolve_analysis_input

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_analysis_input(run_dir=str(tmp_path / "no_such_dir"))


@pytest.mark.unit
def test_resolve_analysis_input_aggregated_csv_not_found(tmp_path: Path) -> None:
    from ptmdiscoverer.reporting import resolve_analysis_input

    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_analysis_input(aggregated_csv=str(tmp_path / "no_such.csv"))


@pytest.mark.unit
def test_load_run_predictions_with_existing_provenance_columns(tmp_path: Path) -> None:
    """Prediction CSV already carries Protein.ID / Run.Index / Source.File —
    exercises the 'else: fillna()' branches that are skipped when those
    columns are absent."""
    from ptmdiscoverer.reporting import load_run_predictions

    run_dir = tmp_path / "output_ts"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "PEPTIDE": "AAAK",
                "PTM.NAME": "Oxidation",
                "PTM.DeltaMass": 15.99,
                "PTM.Protein.Position": 5,
                "PTM.Best.Positions": "[1]",
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "Protein.ID": "P12345",
                "Run.Index": 2,
                "Source.File": "original_source.csv",
            }
        ]
    ).to_csv(run_dir / "ptm_predictions_P12345_ts.csv", index=False)

    df = load_run_predictions(run_dir)
    assert df["Protein.ID"].iloc[0] == "P12345"
    assert df["Run.Index"].iloc[0] == 2
    assert df["Source.File"].iloc[0] == "original_source.csv"


@pytest.mark.unit
def test_load_run_predictions_all_empty_csvs_raises(tmp_path: Path) -> None:
    from ptmdiscoverer.reporting import load_run_predictions

    run_dir = tmp_path / "output_ts"
    run_dir.mkdir()
    # Write a CSV that exists but has zero data rows
    pd.DataFrame(columns=["PEPTIDE", "PTM.NAME", "PTM.DeltaMass", "PTM.Protein.Position"]).to_csv(
        run_dir / "ptm_predictions_P12345_ts.csv", index=False
    )

    with pytest.raises(ValueError, match="contain no rows"):
        load_run_predictions(run_dir)


@pytest.mark.unit
def test_load_aggregated_predictions_empty_raises(tmp_path: Path) -> None:
    from ptmdiscoverer.reporting import load_aggregated_predictions

    csv_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["PEPTIDE", "PTM.NAME"]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="contains no rows"):
        load_aggregated_predictions(csv_path)


# ---------------------------------------------------------------------------
# cli — extract (unknown protein, PSM-extraction path), results (--run-dir)
# ---------------------------------------------------------------------------


import io as _io
import json as _json
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, cast

import ptmdiscoverer.cli as _cli
from conftest import make_test_study


def _run_cli(args: list[str]) -> tuple[int, dict[str, Any], str]:
    stdout_buf = _io.StringIO()
    stderr_buf = _io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exit_code = _cli.main(args)
    return exit_code, cast(dict[str, Any], _json.loads(stdout_buf.getvalue())), stderr_buf.getvalue()


@pytest.mark.unit
def test_cli_extract_unknown_protein(tmp_path: Path) -> None:
    """extract with a protein ID that is not in metadata → error JSON."""
    base = make_test_study(tmp_path)
    exit_code, payload, _ = _run_cli(["extract", str(base), "UNKNOWN_XYZ"])
    assert exit_code == 1
    assert "not found" in payload["error"].lower()


@pytest.mark.unit
def test_cli_extract_triggers_psm_extraction(tmp_path: Path) -> None:
    """Q99999 has no pre-existing delta-mass CSV → extraction runs from PSM data."""
    base = make_test_study(tmp_path)
    exit_code, payload, _ = _run_cli(["extract", str(base), "Q99999"])
    assert exit_code == 0
    assert payload["source"] == "extraction"
    assert payload["rows"] >= 1


@pytest.mark.unit
def test_cli_results_study_folder_no_output_dir(tmp_path: Path) -> None:
    """results --study-folder when no output/ dir exists → error JSON."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "base_prompt.txt").write_text("p", encoding="utf-8")
    (study / "protein_metadata.json").write_text(
        json.dumps({"version": 1, "proteins": [{"uniprot_id": "P1", "name": "X"}]}),
        encoding="utf-8",
    )
    exit_code, payload, _ = _run_cli(["results", "--study-folder", str(study)])
    assert exit_code == 1
    assert "error" in payload


@pytest.mark.unit
def test_cli_results_run_dir_with_json_files(tmp_path: Path) -> None:
    """results --run-dir pointing at a dir with prediction JSON files."""
    run_dir = tmp_path / "output_20260101"
    run_dir.mkdir()
    prediction_json = run_dir / "ptm_predictions_P12345_20260101.json"
    prediction_json.write_text(
        json.dumps([{"PTM.NAME": "Oxidation", "PTM.Protein.Position": 5}]),
        encoding="utf-8",
    )

    exit_code, payload, _ = _run_cli(["results", "--run-dir", str(run_dir)])
    assert exit_code == 0
    assert payload["run_dir"] == str(run_dir)
    assert len(payload["predictions"]) == 1
    assert payload["predictions"][0]["protein_id"] == "P12345"
    assert payload["predictions"][0]["total_records"] == 1


@pytest.mark.unit
def test_cli_results_run_dir_missing(tmp_path: Path) -> None:
    """results --run-dir pointing at a non-existent directory → error JSON."""
    exit_code, payload, _ = _run_cli(["results", "--run-dir", str(tmp_path / "no_such")])
    assert exit_code == 1
    assert "error" in payload


# ---------------------------------------------------------------------------
# utils — augment_protein_positions edge branches
# ---------------------------------------------------------------------------

from ptmdiscoverer.utils import augment_protein_positions


@pytest.mark.unit
def test_augment_empty_peptide_key_is_skipped() -> None:
    """Record with no PEPTIDE key → pep_key == '' → skipped."""
    delta_df = pd.DataFrame([{"Peptide": "PEPTIDEK", "Best.Positions": "M3", "Positions.In.Protein": 103.0}])
    records: list[dict] = [{}]  # no PEPTIDE
    augment_protein_positions(records, delta_df)
    assert "PTM.Protein.Position" not in records[0]


@pytest.mark.unit
def test_augment_already_valid_position_is_skipped() -> None:
    """Record with PTM.Protein.Position already > 0 is left unchanged."""
    delta_df = pd.DataFrame([{"Peptide": "PEPTIDEK", "Best.Positions": "M3", "Positions.In.Protein": 103.0}])
    records = [{"PEPTIDE": "PEPTIDEK", "PTM.Protein.Position": 55, "PTM.Best.Positions": ["M3"]}]
    augment_protein_positions(records, delta_df)
    assert records[0]["PTM.Protein.Position"] == 55  # unchanged


@pytest.mark.unit
def test_augment_no_best_positions_site_is_skipped() -> None:
    """Record with no PTM.Best.Positions → skipped."""
    delta_df = pd.DataFrame([{"Peptide": "PEPTIDEK", "Best.Positions": "M3", "Positions.In.Protein": 103.0}])
    records = [{"PEPTIDE": "PEPTIDEK", "PTM.Protein.Position": -1}]  # no PTM.Best.Positions
    augment_protein_positions(records, delta_df)
    assert records[0]["PTM.Protein.Position"] == -1  # unchanged


@pytest.mark.unit
def test_augment_resolves_via_direct_mapping() -> None:
    """(pep, local_idx) maps to exactly one position → field is updated."""
    delta_df = pd.DataFrame([{"Peptide": "PEPTIDEK", "Best.Positions": "M3", "Positions.In.Protein": 103.0}])
    records = [{"PEPTIDE": "PEPTIDEK", "PTM.Protein.Position": -1, "PTM.Best.Positions": ["M3"]}]
    augment_protein_positions(records, delta_df)
    assert records[0]["PTM.Protein.Position"] == 103


@pytest.mark.unit
def test_augment_resolves_via_peptide_only_fallback() -> None:
    """(pep, local_idx) not in mapping but peptide_only has exactly one abs pos."""
    delta_df = pd.DataFrame([{"Peptide": "PEPT", "Best.Positions": "M3", "Positions.In.Protein": 103.0}])
    # Record uses site M7 (not in mapping) but PEPT → single abs pos 103
    records = [{"PEPTIDE": "PEPT", "PTM.Protein.Position": -1, "PTM.Best.Positions": ["M7"]}]
    augment_protein_positions(records, delta_df)
    assert records[0]["PTM.Protein.Position"] == 103


@pytest.mark.unit
def test_augment_unresolved_when_multiple_candidates() -> None:
    """Ambiguous mapping (multiple abs positions for same peptide) → position unchanged."""
    delta_df = pd.DataFrame(
        [
            {"Peptide": "AMBIG", "Best.Positions": "M3", "Positions.In.Protein": 103.0},
            {"Peptide": "AMBIG", "Best.Positions": "M3", "Positions.In.Protein": 200.0},
        ]
    )
    records = [{"PEPTIDE": "AMBIG", "PTM.Protein.Position": -1, "PTM.Best.Positions": ["M3"]}]
    augment_protein_positions(records, delta_df)
    assert records[0]["PTM.Protein.Position"] == -1  # ambiguous → left untouched


@pytest.mark.unit
def test_augment_non_numeric_positions_in_protein_does_not_raise() -> None:
    """Non-numeric Positions.In.Protein entry is skipped gracefully (except branch)."""
    delta_df = pd.DataFrame([{"Peptide": "PEPTIDEK", "Best.Positions": "M3", "Positions.In.Protein": "N/A"}])
    records = [{"PEPTIDE": "PEPTIDEK", "PTM.Protein.Position": -1, "PTM.Best.Positions": ["M3"]}]
    augment_protein_positions(records, delta_df)  # must not raise
    # mapping was not built → position stays -1
    assert records[0]["PTM.Protein.Position"] == -1


# ---------------------------------------------------------------------------
# inference — assess_study_folder error branches, run_inference API key check
# ---------------------------------------------------------------------------

from ptmdiscoverer.inference import InferenceContext, assess_study_folder, process_protein


@pytest.mark.unit
def test_assess_study_folder_bad_metadata_json(tmp_path: Path) -> None:
    """Metadata file exists but is not valid JSON → metadata_format_ok=False."""
    (tmp_path / "base_prompt.txt").write_text("p", encoding="utf-8")
    (tmp_path / "protein_metadata.json").write_text("NOT_VALID_JSON{{", encoding="utf-8")

    report = assess_study_folder(str(tmp_path))
    assert report["checks"]["protein_metadata_file"]["parse_ok"] is False
    assert any("Failed to parse" in e for e in report["errors"])


@pytest.mark.unit
def test_assess_study_folder_zero_proteins_warns(tmp_path: Path) -> None:
    """Metadata is valid JSON but proteins list is empty → warning appended."""
    (tmp_path / "base_prompt.txt").write_text("p", encoding="utf-8")
    (tmp_path / "protein_metadata.json").write_text(json.dumps({"version": 1, "proteins": []}), encoding="utf-8")

    report = assess_study_folder(str(tmp_path))
    assert report["valid"] is True
    assert any("zero valid protein" in w for w in report["warnings"])


@pytest.mark.unit
def test_run_inference_raises_system_exit_when_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_inference with no client and no OPENAI_API_KEY raises SystemExit.
    Requires a valid study folder since validate_study_folder runs first.
    """
    from ptmdiscoverer.inference import run_inference

    # Minimal valid study so validate_study_folder passes
    (tmp_path / "base_prompt.txt").write_text("p", encoding="utf-8")
    (tmp_path / "protein_metadata.json").write_text(
        json.dumps({"version": 1, "proteins": [{"uniprot_id": "P1", "name": "X"}]}),
        encoding="utf-8",
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        run_inference(base_data_folder=str(tmp_path))


@pytest.mark.unit
def test_process_protein_skips_when_msrun_table_empty(tmp_path: Path) -> None:
    """Protein without a delta_mass_file and empty msrun_table → (False, [])."""
    ctx = InferenceContext(
        client=None,
        base_prompt="",
        model="",
        timestamp="ts",
        base_data_folder=str(tmp_path),
        run_dir=str(tmp_path),
        log_handle=io.StringIO(),
        msrun_table=pd.DataFrame(),
    )
    changed, records = process_protein(ctx, {"uniprot_id": "PTEST"})
    assert changed is False
    assert records == []
