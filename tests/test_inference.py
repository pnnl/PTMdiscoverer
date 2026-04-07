"""Tests for inference workflow (metadata/prompt and extraction path)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from conftest import FakeStream

import ptmdiscoverer.inference as inf


@pytest.mark.unit
def test_load_protein_metadata_success(tmp_path: Path) -> None:
    meta = {
        "version": 1,
        "proteins": [{"uniprot_id": "P123", "name": "Prot", "delta_mass_file": ""}],
    }
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    proteins = inf.load_protein_metadata(str(p))
    assert proteins[0]["uniprot_id"] == "P123"


@pytest.mark.unit
def test_load_protein_metadata_invalid(tmp_path: Path) -> None:
    meta = {"version": 1, "proteins": {"oops": 1}}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError):
        inf.load_protein_metadata(str(p))


@pytest.mark.unit
def test_build_prompt_strips_spectrum() -> None:
    df = pd.DataFrame({"Spectrum": ["s1"], "Peptide": ["AAA"], "Delta.Mass": [15.0]})
    prompt = inf.build_prompt("BASE", "ProtName", df)
    assert "Spectrum" not in prompt
    assert "Peptide" in prompt and "Delta.Mass" in prompt


@pytest.mark.integration
def test_inference_end_to_end_creates_predictions(tmp_path: Path) -> None:
    base = tmp_path / "case"
    base.mkdir()
    (base / "output").mkdir()
    (base / "dataset_labels.csv").write_text("MSRUNID\n1\n", encoding="utf-8")
    # Create psm.tsv
    psm_dir = base / "Results_MSFragger" / "1"
    psm_dir.mkdir(parents=True)
    psm_content = (
        "Spectrum\tPeptide\tCharge\tCalculated Peptide Mass\tCalculated M/Z\tDelta Mass\tBest Positions\tNumber Best Positions\tIons Best Position\tIons Second Best Position\tScore Best Position\tScore Second Best Position\tProbability\tNumber of Missed Cleavages\tProtein ID\tRetention\tIs Unique\n"
        "s1\tPEPTIDEK\t2\t1000\t500\t15.2\t1\t1\t10\t5\t60\t10\t0.95\t0\tP12345\t600\tTrue\n"
    )
    (psm_dir / "psm.tsv").write_text(psm_content, encoding="utf-8")
    (base / "base_prompt.txt").write_text("Base prompt header", encoding="utf-8")
    metadata = {
        "version": 1,
        "proteins": [
            {"uniprot_id": "P12345", "name": "ExampleProt", "delta_mass_file": ""},
            {"uniprot_id": "NOHIT", "name": "NoHitProt", "delta_mass_file": ""},
        ],
    }
    (base / "protein_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    class FakeChoices:
        def __init__(self, content: str):  # noqa: D401
            self.message = SimpleNamespace(content=content)

    fenced_json = (
        "Here is the answer:\n```json\n"
        + json.dumps(
            [
                {
                    "PEPTIDE": "PEPTIDEK",
                    "PTM.DeltaMass": 15.2,
                    "PTM.Best.Positions": ["1"],
                    "PTM.NAME": "ExampleMod",
                    "PTM.Impact.In.Enzyme.Function": "Unknown",
                    "PTM.Protein.Position": 42,
                }
            ]
        )
        + "\n```"
    )

    class FakeClient:
        def __init__(self):  # noqa: D401
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_kw: FakeStream(fenced_json)))

    inf.run_inference(base_data_folder=str(base), client=FakeClient(), timestamp="2023-01-01-00-00")

    # Assert extraction created file for P12345
    generated = base / "P12345_delta-masses.csv"
    assert generated.exists()
    # Metadata should now include path for P12345 but still empty for NOHIT (no PSM match)
    updated = json.loads((base / "protein_metadata.json").read_text(encoding="utf-8"))
    entries = {p["uniprot_id"]: p for p in updated["proteins"]}
    assert entries["P12345"]["delta_mass_file"].endswith("P12345_delta-masses.csv")
    assert entries["NOHIT"]["delta_mass_file"] == ""
    # Expect a run directory: output/output_*
    run_dir = base / "output" / "output_2023-01-01-00-00"
    assert run_dir.exists(), "Expected timestamped run directory"
    # Per-protein prediction files (json + csv) and a log inside run_dir
    pred_jsons = list(run_dir.glob("ptm_predictions_P12345_*.json"))
    assert pred_jsons, "Expected per-protein JSON prediction file"
    pred_csvs = list(run_dir.glob("ptm_predictions_P12345_*.csv"))
    assert pred_csvs, "Expected per-protein CSV prediction file"
    # JSON and CSV should contain matching record
    jdata = json.loads(pred_jsons[0].read_text(encoding="utf-8"))
    assert jdata[0]["PEPTIDE"] == "PEPTIDEK"
    # Required field must be present (schema enforced) and value propagated from model output
    assert jdata[0]["PTM.Protein.Position"] == 42
    # Log file present
    logs = list(run_dir.glob("output_log_*.txt"))
    assert logs, "Expected a log file with prompt/response"
    log_text = logs[0].read_text(encoding="utf-8")
    assert "P12345" in log_text and "PEPTIDEK" in log_text


@pytest.mark.integration
def test_run_multiple_inferences_with_aggregation(tmp_path: Path) -> None:
    """Run multiple inferences with in-function aggregation and validate output."""
    base = tmp_path / "case"
    base.mkdir()
    (base / "output").mkdir()
    (base / "dataset_labels.csv").write_text("MSRUNID\n1\n", encoding="utf-8")
    # Create psm.tsv (single qualifying row)
    psm_dir = base / "Results_MSFragger" / "1"
    psm_dir.mkdir(parents=True)
    psm_content = (
        "Spectrum\tPeptide\tCharge\tCalculated Peptide Mass\tCalculated M/Z\tDelta Mass\tBest Positions\tNumber Best Positions\tIons Best Position\tIons Second Best Position\tScore Best Position\tScore Second Best Position\tProbability\tNumber of Missed Cleavages\tProtein ID\tRetention\tIs Unique\n"
        "s1\tPEPTIDEK\t2\t1000\t500\t15.2\t1\t1\t10\t5\t60\t10\t0.95\t0\tP12345\t600\tTrue\n"
    )
    (psm_dir / "psm.tsv").write_text(psm_content, encoding="utf-8")
    (base / "base_prompt.txt").write_text("Base prompt header", encoding="utf-8")
    metadata = {"version": 1, "proteins": [{"uniprot_id": "P12345", "name": "Prot", "delta_mass_file": ""}]}
    (base / "protein_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    class DummyChoices:
        def __init__(self, content: str):  # noqa: D401
            self.message = SimpleNamespace(content=content)

    fenced_json = (
        "```json\n["
        + '{"PEPTIDE":"PEPTIDEK","PTM.DeltaMass":15.2,"PTM.Best.Positions":["1"],"PTM.NAME":"Mod","PTM.Impact.In.Enzyme.Function":"?","PTM.Protein.Position":5}'
        + "]\n```"
    )

    class DummyClient:
        def __init__(self):  # noqa: D401
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream(fenced_json)))

    aggregated_csv = base / "aggregated.csv"
    run_dirs, agg_df = inf.run_and_aggregate_multiple_inferences(
        2,
        base_data_folder=str(base),
        client=DummyClient(),
        aggregated_csv_path=str(aggregated_csv),
    )
    assert len(run_dirs) == 2
    assert not agg_df.empty
    assert aggregated_csv.exists()
    # Expect two aggregated rows (one prediction per run)
    assert len(agg_df) == 2
    assert {"Protein.ID", "Run.Index"}.issubset(agg_df.columns)
    assert agg_df["Run.Index"].nunique() == 2


@pytest.mark.integration
def test_run_multiple_inferences_default_aggregated_path(tmp_path: Path) -> None:
    """Ensure default aggregated CSV path (timestamped) is created when not provided."""
    base = tmp_path / "case"
    base.mkdir()
    (base / "output").mkdir()
    (base / "dataset_labels.csv").write_text("MSRUNID\n1\n", encoding="utf-8")
    psm_dir = base / "Results_MSFragger" / "1"
    psm_dir.mkdir(parents=True)
    psm_content = (
        "Spectrum\tPeptide\tCharge\tCalculated Peptide Mass\tCalculated M/Z\tDelta Mass\tBest Positions\tNumber Best Positions\tIons Best Position\tIons Second Best Position\tScore Best Position\tScore Second Best Position\tProbability\tNumber of Missed Cleavages\tProtein ID\tRetention\tIs Unique\n"
        "s1\tPEPTIDEK\t2\t1000\t500\t15.2\t1\t1\t10\t5\t60\t10\t0.95\t0\tP12345\t600\tTrue\n"
    )
    (psm_dir / "psm.tsv").write_text(psm_content, encoding="utf-8")
    metadata = {"version": 1, "proteins": [{"uniprot_id": "P12345", "name": "Prot", "delta_mass_file": ""}]}
    (base / "protein_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    # Create required base prompt file (was missing and caused FileNotFoundError)
    (base / "base_prompt.txt").write_text("Base prompt header", encoding="utf-8")

    class DummyChoices:
        def __init__(self, content: str):  # noqa: D401
            self.message = SimpleNamespace(content=content)

    fenced_json = (
        "```json\n["
        + '{"PEPTIDE":"PEPTIDEK","PTM.DeltaMass":15.2,"PTM.Best.Positions":["1"],"PTM.NAME":"Mod","PTM.Impact.In.Enzyme.Function":"?","PTM.Protein.Position":7}'
        + "]\n```"
    )

    class DummyClient:
        def __init__(self):  # noqa: D401
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream(fenced_json)))

    run_dirs, agg_df = inf.run_and_aggregate_multiple_inferences(
        2,
        base_data_folder=str(base),
        client=DummyClient(),
    )
    assert len(run_dirs) == 2
    assert not agg_df.empty
    # Find aggregated file(s)
    agg_files = list((base / "output").glob("aggregated_predictions_*.csv"))
    assert agg_files, "Expected a default aggregated CSV file"
    # Basic sanity: single aggregated file and row count equals runs (1 prediction per run)
    assert len(agg_files) == 1
    import pandas as _pd

    written_df = _pd.read_csv(agg_files[0])
    assert len(written_df) == 2
