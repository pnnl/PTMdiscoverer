"""Tests for deterministic PTM run-results reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ptmdiscoverer.reporting import analyze_run_results


def _write_aggregated_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "PEPTIDE": "AAAK",
                "PTM.Best.Positions": "[1]",
                "PTM.DeltaMass": 15.99,
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "PTM.NAME": "Oxidation",
                "PTM.Protein.Position": 5,
                "Source.File": f"run{run_index}.csv",
                "Run.Index": run_index,
                "Protein.ID": "P12345",
            }
            for run_index in range(6)
        ]
        + [
            {
                "PEPTIDE": "BBBK",
                "PTM.Best.Positions": "[2]",
                "PTM.DeltaMass": 79.96,
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "PTM.NAME": "Phosphorylation",
                "PTM.Protein.Position": 10,
                "Source.File": f"run{run_index}.csv",
                "Run.Index": run_index,
                "Protein.ID": "Q99999",
            }
            for run_index in [0, 1]
        ]
        + [
            {
                "PEPTIDE": "BBBK",
                "PTM.Best.Positions": "[2]",
                "PTM.DeltaMass": 42.01,
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "PTM.NAME": "Acetylation",
                "PTM.Protein.Position": 10,
                "Source.File": "run2.csv",
                "Run.Index": 2,
                "Protein.ID": "Q99999",
            }
        ]
    )
    path = tmp_path / "aggregated_predictions_fixture.csv"
    frame.to_csv(path, index=False)
    return path


def _write_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "output_2026-03-20-12-00-00-000"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "PEPTIDE": "AAAK",
                "PTM.Best.Positions": "[1]",
                "PTM.DeltaMass": 15.99,
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "PTM.NAME": "Oxidation",
                "PTM.Protein.Position": 5,
            }
        ]
    ).to_csv(run_dir / "ptm_predictions_Q99999_2026-03-20-12-00-00-000.csv", index=False)
    pd.DataFrame(
        [
            {
                "PEPTIDE": "BBBK",
                "PTM.Best.Positions": "[2]",
                "PTM.DeltaMass": 79.96,
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "PTM.NAME": "Phosphorylation",
                "PTM.Protein.Position": 10,
            }
        ]
    ).to_csv(run_dir / "ptm_predictions_P12345_2026-03-20-12-00-00-000.csv", index=False)
    return run_dir


@pytest.mark.unit
def test_analyze_run_results_from_aggregated_csv_writes_artifacts(tmp_path: Path) -> None:
    aggregated_csv = _write_aggregated_csv(tmp_path)

    payload = analyze_run_results(aggregated_csv=str(aggregated_csv), output_dir=str(tmp_path), top_k=5)

    assert payload["analysis_mode"] == "aggregated_csv"
    assert payload["num_runs"] == 6
    assert payload["metrics"]["total_predictions"] == 9
    assert payload["metrics"]["unique_proteins"] == 2
    assert len(payload["output"]["tables"]) == 7
    assert len(payload["output"]["plots"]) >= 6
    assert payload["high_consensus_sites"]
    assert payload["disagreement_sites"]
    assert payload["protein_focus_summary"]
    assert payload["delta_mass_hotspots"]
    assert Path(payload["output"]["report_dir"]).name.startswith("analysis_")
    assert Path(payload["output"]["summary_json"]).name.startswith("summary_")
    assert Path(payload["output"]["summary_markdown"]).name.startswith("summary_")
    assert Path(payload["output"]["summary_json"]).is_file()
    assert Path(payload["output"]["summary_markdown"]).is_file()
    table_names = {Path(table_path).name.split("_")[0] for table_path in payload["output"]["tables"]}
    assert {"ptm", "top", "peptide", "site", "protein", "run", "delta"} <= table_names
    for table_path in payload["output"]["tables"]:
        assert Path(table_path).is_file()
    for plot_path in payload["output"]["plots"]:
        assert Path(plot_path).is_file()


@pytest.mark.unit
def test_analyze_run_results_from_run_dir_normalizes_csv_predictions(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path)

    payload = analyze_run_results(run_dir=str(run_dir), output_dir=str(tmp_path), create_plots=False)

    assert payload["analysis_mode"] == "run_dir"
    assert payload["num_runs"] == 1
    assert payload["metrics"]["total_predictions"] == 2
    assert payload["metrics"]["unique_proteins"] == 2
    assert payload["output"]["plots"] == []
    assert payload["source"]["run_dir"] == str(run_dir)
    assert Path(payload["output"]["summary_json"]).is_file()


@pytest.mark.unit
def test_analyze_run_results_from_run_dir_preserves_provenance_columns(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path)

    payload = analyze_run_results(run_dir=str(run_dir), output_dir=str(tmp_path), create_plots=False)

    site_table = pd.read_csv(payload["output"]["tables"][0])
    assert set(site_table["Protein.ID"]) == {"P12345", "Q99999"}
    assert len(payload["source"]["input_files"]) == 2
    assert all(path.endswith(".csv") for path in payload["source"]["input_files"])
    assert Path(payload["output"]["summary_json"]).is_file()


@pytest.mark.unit
def test_analyze_run_results_builds_protein_rollup_and_disagreement_summary(tmp_path: Path) -> None:
    aggregated_csv = _write_aggregated_csv(tmp_path)

    payload = analyze_run_results(aggregated_csv=str(aggregated_csv), output_dir=str(tmp_path), create_plots=False)

    table_map = {Path(path).name.rsplit("_", 1)[0]: path for path in payload["output"]["tables"]}
    protein_rollup = pd.read_csv(table_map["protein_rollup"])
    disagreement = pd.read_csv(table_map["run_disagreement_summary"])
    delta_mass_summary = pd.read_csv(table_map["delta_mass_summary"])

    p12345 = protein_rollup.loc[protein_rollup["Protein.ID"] == "P12345"].iloc[0]
    assert p12345["TotalPredictions"] == 6
    assert p12345["TopSitePredictionFraction"] == 100.0
    assert p12345["DominantPTMName"] == "Oxidation"

    q99999_disagreement = disagreement.loc[disagreement["Protein.ID"] == "Q99999"].iloc[0]
    assert bool(q99999_disagreement["HasCompetingPTMs"]) is True
    assert q99999_disagreement["PTMNameCount"] == 2

    dominant_mass = delta_mass_summary.iloc[0]
    assert dominant_mass["RoundedDeltaMass"] == 15.99
    assert dominant_mass["ObservationCount"] == 6


@pytest.mark.unit
def test_analyze_run_results_requires_exactly_one_input(tmp_path: Path) -> None:
    aggregated_csv = _write_aggregated_csv(tmp_path)
    run_dir = _write_run_dir(tmp_path)

    with pytest.raises(ValueError, match="exactly one"):
        analyze_run_results(run_dir=str(run_dir), aggregated_csv=str(aggregated_csv))


@pytest.mark.unit
def test_analyze_run_results_rejects_missing_required_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"PEPTIDE": "AAAK", "PTM.NAME": "Oxidation"}]).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        analyze_run_results(aggregated_csv=str(bad_csv), create_plots=False)
