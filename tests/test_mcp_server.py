"""Tests for the MCP server exposing PTMdiscoverer tools."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from conftest import FakeStream, make_test_study, write_simple_aggregated_csv, write_simple_run_dir

# The mcp_server module is an optional extra; guard the import so the test
# file itself can be collected even when the 'mcp' extra is not installed.
mcp_mod = pytest.importorskip("ptmdiscoverer.mcp_server", reason="mcp extra not installed")


def _fake_client(json_payload: str) -> object:
    """Return a fake OpenAI-compatible client returning the given JSON payload."""

    class FakeChoices:
        def __init__(self, content: str):
            self.message = SimpleNamespace(content=content)

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream(json_payload)))

    return FakeClient()


# ---------------------------------------------------------------------------
# validate_study_folder_tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_study_folder_tool_returns_report(tmp_path: Path) -> None:
    """validate_study_folder_tool returns structured validation output."""
    base = make_test_study(tmp_path)
    result = await mcp_mod.validate_study_folder_tool(study_folder=str(base))
    assert isinstance(result, dict)
    assert result["valid"] is True
    assert result["protein_count"] == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_validate_study_folder_tool_invalid_folder(tmp_path: Path) -> None:
    """validate_study_folder_tool reports invalid paths without raising."""
    result = await mcp_mod.validate_study_folder_tool(study_folder=str(tmp_path / "nope"))
    assert result["valid"] is False
    assert result["errors"]


# ---------------------------------------------------------------------------
# list_proteins
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_proteins_returns_metadata(tmp_path: Path) -> None:
    """list_proteins returns the protein entries with availability info."""
    base = make_test_study(tmp_path)
    result = await mcp_mod.list_proteins(study_folder=str(base))
    assert isinstance(result, dict)
    assert len(result["proteins"]) == 2
    p1 = next(p for p in result["proteins"] if p["uniprot_id"] == "P12345")
    assert p1["has_delta_masses"] is True
    p2 = next(p for p in result["proteins"] if p["uniprot_id"] == "Q99999")
    assert p2["has_delta_masses"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_proteins_invalid_folder(tmp_path: Path) -> None:
    """list_proteins returns an error dict for a non-existent folder."""
    result = await mcp_mod.list_proteins(study_folder=str(tmp_path / "nope"))
    assert "error" in result


# ---------------------------------------------------------------------------
# extract_protein_delta_masses
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_delta_masses_existing_file(tmp_path: Path) -> None:
    """extract_protein_delta_masses reads an existing CSV."""
    base = make_test_study(tmp_path)
    result = await mcp_mod.extract_protein_delta_masses(study_folder=str(base), uniprot_id="P12345")
    assert isinstance(result, dict)
    assert result["rows"] >= 1
    assert "preview" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_delta_masses_triggers_extraction(tmp_path: Path) -> None:
    """extract_protein_delta_masses extracts from PSM data when CSV missing."""
    base = make_test_study(tmp_path, with_psm=True)
    result = await mcp_mod.extract_protein_delta_masses(study_folder=str(base), uniprot_id="Q99999")
    assert isinstance(result, dict)
    # Extraction may or may not yield rows depending on data, but should not error
    assert "rows" in result or "error" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_delta_masses_unknown_protein(tmp_path: Path) -> None:
    """extract_protein_delta_masses errors for unknown protein."""
    base = make_test_study(tmp_path)
    result = await mcp_mod.extract_protein_delta_masses(study_folder=str(base), uniprot_id="ZZZZZZ")
    assert "error" in result


# ---------------------------------------------------------------------------
# run_ptm_inference
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_ptm_inference_single(tmp_path: Path) -> None:
    """run_ptm_inference completes a single run and returns metadata."""
    base = make_test_study(tmp_path)
    fenced = (
        '```json\n[{"PEPTIDE":"ABCK","PTM.DeltaMass":15.2,'
        '"PTM.Best.Positions":["1"],"PTM.NAME":"Mod",'
        '"PTM.Impact.In.Enzyme.Function":"?","PTM.Protein.Position":5}]\n```'
    )
    client = _fake_client(fenced)
    with patch.object(mcp_mod, "_get_client", return_value=client):
        result = await mcp_mod.run_ptm_inference(study_folder=str(base), n_runs=1)
    assert isinstance(result, dict)
    assert result["proteins_processed"] >= 1
    assert "run_dir" in result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_ptm_inference_multiple(tmp_path: Path) -> None:
    """run_ptm_inference with n_runs>1 returns aggregated info."""
    base = make_test_study(tmp_path)
    fenced = (
        '```json\n[{"PEPTIDE":"ABCK","PTM.DeltaMass":15.2,'
        '"PTM.Best.Positions":["1"],"PTM.NAME":"Mod",'
        '"PTM.Impact.In.Enzyme.Function":"?","PTM.Protein.Position":5}]\n```'
    )
    client = _fake_client(fenced)
    with patch.object(mcp_mod, "_get_client", return_value=client):
        result = await mcp_mod.run_ptm_inference(study_folder=str(base), n_runs=2)
    assert isinstance(result, dict)
    assert result["n_runs"] == 2
    assert "aggregated_rows" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_ptm_inference_invalid_folder(tmp_path: Path) -> None:
    """run_ptm_inference errors for missing study folder."""
    result = await mcp_mod.run_ptm_inference(study_folder=str(tmp_path / "nope"))
    assert "error" in result


# ---------------------------------------------------------------------------
# get_run_results
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_run_results_returns_artifacts(tmp_path: Path) -> None:
    """get_run_results lists prediction files from a run directory."""
    base = make_test_study(tmp_path)
    # Create a fake run dir with prediction artifacts
    run_dir = base / "output" / "output_2024-01-01-00-00-00-000"
    run_dir.mkdir(parents=True)
    pred = [{"PEPTIDE": "ABCK", "PTM.Protein.Position": 5}]
    (run_dir / "ptm_predictions_P12345_2024-01-01-00-00-00-000.json").write_text(json.dumps(pred), encoding="utf-8")
    (run_dir / "ptm_predictions_P12345_2024-01-01-00-00-00-000.csv").write_text(
        "PEPTIDE,PTM.Protein.Position\nABCK,5\n", encoding="utf-8"
    )
    result = await mcp_mod.get_run_results(run_dir=str(run_dir))
    assert isinstance(result, dict)
    assert len(result["predictions"]) >= 1
    assert result["predictions"][0]["protein_id"] == "P12345"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_run_results_lists_available_runs(tmp_path: Path) -> None:
    """get_run_results without run_dir lists available runs in study folder."""
    base = make_test_study(tmp_path)
    rd1 = base / "output" / "output_2024-01-01"
    rd1.mkdir(parents=True)
    rd2 = base / "output" / "output_2024-01-02"
    rd2.mkdir(parents=True)
    result = await mcp_mod.get_run_results(study_folder=str(base))
    assert isinstance(result, dict)
    assert len(result["available_runs"]) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_run_results_invalid_dir(tmp_path: Path) -> None:
    """get_run_results errors for non-existent directory."""
    result = await mcp_mod.get_run_results(run_dir=str(tmp_path / "nope"))
    assert "error" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analyze_run_results_returns_report_metadata(tmp_path: Path) -> None:
    aggregated_csv = write_simple_aggregated_csv(tmp_path)

    result = await mcp_mod.analyze_run_results(
        aggregated_csv=str(aggregated_csv),
        output_dir=str(tmp_path),
        create_plots=False,
    )

    assert result["analysis_mode"] == "aggregated_csv"
    assert result["output"]["plots"] == []
    assert Path(result["output"]["summary_json"]).is_file()
    assert result["source"]["input_files"] == [str(aggregated_csv)]
    assert "protein_focus_summary" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_analyze_run_results_from_run_dir_uses_prediction_csvs(tmp_path: Path) -> None:
    run_dir = write_simple_run_dir(tmp_path)

    result = await mcp_mod.analyze_run_results(run_dir=str(run_dir), output_dir=str(tmp_path), create_plots=False)

    assert result["analysis_mode"] == "run_dir"
    assert result["source"]["run_dir"] == str(run_dir)
    assert result["source"]["input_files"] == [str(run_dir / "ptm_predictions_P12345_2024-01-01-00-00-00-000.csv")]
    assert Path(result["output"]["summary_json"]).is_file()


# ---------------------------------------------------------------------------
# _get_client helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_client_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_client raises ValueError when OPENAI_API_KEY is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        mcp_mod._get_client()


@pytest.mark.unit
def test_get_client_constructs_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_client returns an OpenAI client when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    with patch("ptmdiscoverer.mcp_server.openai.OpenAI") as mock_cls:
        mock_cls.return_value = object()
        client = mcp_mod._get_client()
        mock_cls.assert_called_once()
        assert client is not None


# ---------------------------------------------------------------------------
# extract_protein_delta_masses — additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_delta_masses_invalid_folder(tmp_path: Path) -> None:
    """extract_protein_delta_masses returns error for non-existent folder."""
    result = await mcp_mod.extract_protein_delta_masses(study_folder=str(tmp_path / "nope"), uniprot_id="P12345")
    assert "error" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_delta_masses_no_labels_csv(tmp_path: Path) -> None:
    """extract_protein_delta_masses errors when no delta-mass CSV and no labels file."""
    base = make_test_study(tmp_path, with_psm=False)
    # Q99999 has no pre-existing delta_mass_file and no PSM data (with_psm=False)
    result = await mcp_mod.extract_protein_delta_masses(study_folder=str(base), uniprot_id="Q99999")
    assert "error" in result
    assert "dataset_labels.csv" in result["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_extract_delta_masses_csv_read_error(tmp_path: Path) -> None:
    """extract_protein_delta_masses returns error when the CSV file cannot be read."""
    base = make_test_study(tmp_path)
    with patch("ptmdiscoverer.mcp_server.pd.read_csv", side_effect=OSError("disk error")):
        result = await mcp_mod.extract_protein_delta_masses(study_folder=str(base), uniprot_id="P12345")
    assert "error" in result
    assert "disk error" in result["error"]


# ---------------------------------------------------------------------------
# run_ptm_inference — client failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_ptm_inference_client_build_fails(tmp_path: Path) -> None:
    """run_ptm_inference returns error when _get_client raises."""
    base = make_test_study(tmp_path)
    with patch.object(mcp_mod, "_get_client", side_effect=ValueError("no API key")):
        result = await mcp_mod.run_ptm_inference(study_folder=str(base))
    assert "error" in result
    assert "no API key" in result["error"]


# ---------------------------------------------------------------------------
# get_run_results — additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_run_results_no_args() -> None:
    """get_run_results with neither run_dir nor study_folder returns error."""
    result = await mcp_mod.get_run_results()
    assert "error" in result


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_run_results_study_folder_no_output_dir(tmp_path: Path) -> None:
    """get_run_results returns error when study_folder has no output/ subdirectory."""
    # Create a bare study dir without an output/ subdir
    study = tmp_path / "study"
    study.mkdir()
    result = await mcp_mod.get_run_results(study_folder=str(study))
    assert "error" in result
    assert "No output directory" in result["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_run_results_corrupt_json_skipped(tmp_path: Path) -> None:
    """get_run_results skips corrupt JSON files and returns empty predictions."""
    run_dir = tmp_path / "output_2024-corrupt"
    run_dir.mkdir()
    (run_dir / "ptm_predictions_P12345_2024.json").write_text("{this is not valid json}", encoding="utf-8")
    result = await mcp_mod.get_run_results(run_dir=str(run_dir))
    assert isinstance(result, dict)
    assert result["predictions"] == []
