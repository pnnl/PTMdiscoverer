"""Optional end-to-end CLI integration test using the example study fixture.

Covers every subcommand and exercises all CLI flags that are absent from the
quickstart documentation but present in build_parser():

    extract --max-preview-rows
    run --base-url, --aggregated-csv-path
    results --run-dir, --max-preview-records
    analyze --top-k, --output-dir, --no-plots

Run with::

    uv run pytest -m integration -v

The test monkeypatches the LLM layer so no real API call is made, but every
other part of the pipeline (file I/O, extraction, reporting, analysis) runs
against real code and the committed fixture.

Fixture layout
--------------
tests/fixtures/example_study/
    protein_metadata.json          2 proteins: P99001 (pre-extracted), P99002 (PSM)
    base_prompt.txt                generic study prompt
    dataset_labels.csv             2 runs: control (1) and treatment (2)
    P99001_delta-masses.csv        4 pre-extracted rows (2 peptides × 2 runs)
    Results_MSFragger/
        1/psm.tsv                  6 rows: 3 pass all filters, 3 intentionally fail
        2/psm.tsv                  4 rows: 3 pass all filters, 1 intentionally fails

Intentional filter failures in the PSM files
---------------------------------------------
Run 1, row 4  AVMGDLTPK  Delta.Mass=5.003     → fails Delta.Mass > 10.0
Run 1, row 5  ARFMTPVEK  Probability=0.80     → fails Probability > 0.9
Run 1, row 6  FLQTEVMRK  MissedCleavages=1    → fails Number.of.Missed.Cleavages == 0
Run 2, row 4  SVTMEPQAK  NumberBestPositions=2 → fails localization uniqueness == 1
"""

from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from dotenv import load_dotenv

import ptmdiscoverer.cli as cli
from ptmdiscoverer.inference import run_and_aggregate_multiple_inferences, run_inference

# ---------------------------------------------------------------------------
# Location of the committed example fixture
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "example_study"

# ---------------------------------------------------------------------------
# Opt-in via:  pytest -m integration
# Excluded from the default run by addopts in pyproject.toml.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli(args: list[str]) -> tuple[int, dict[str, Any], str]:
    """Invoke the CLI in-process, return (exit_code, parsed_json, stderr)."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exit_code = cli.main(args)
    return exit_code, cast(dict[str, Any], json.loads(stdout_buf.getvalue())), stderr_buf.getvalue()


def _fake_run_inference(
    *,
    base_data_folder: str,
    model: str,
    base_url: str | None,
) -> dict[str, Any]:
    """Stub that writes minimal prediction artefacts without calling an LLM."""
    from pathlib import Path as _Path
    import datetime

    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-000")
    run_dir = _Path(base_data_folder) / "output" / f"output_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for protein_id in ("P99001", "P99002"):
        predictions = [
            {
                "PEPTIDE": "AAMTGLTK" if protein_id == "P99001" else "AVMGDLTPK",
                "PTM.DeltaMass": 15.995,
                "PTM.Best.Positions": ["M3"],
                "PTM.NAME": "Oxidation",
                "PTM.Impact.In.Enzyme.Function": "Stub prediction",
                "PTM.Protein.Position": 103 if protein_id == "P99001" else 53,
            }
        ]
        json_path = run_dir / f"ptm_predictions_{protein_id}_{ts}.json"
        json_path.write_text(json.dumps(predictions), encoding="utf-8")

        csv_path = run_dir / f"ptm_predictions_{protein_id}_{ts}.csv"
        pd.DataFrame(
            [
                {
                    "PEPTIDE": p["PEPTIDE"],
                    "PTM.DeltaMass": p["PTM.DeltaMass"],
                    "PTM.Best.Positions": str(p["PTM.Best.Positions"]),
                    "PTM.NAME": p["PTM.NAME"],
                    "PTM.Impact.In.Enzyme.Function": p["PTM.Impact.In.Enzyme.Function"],
                    "PTM.Protein.Position": p["PTM.Protein.Position"],
                }
                for p in predictions
            ]
        ).to_csv(csv_path, index=False)

    return {
        "run_dir": str(run_dir),
        "proteins_processed": 2,
        "total_ptm_records": 2,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFixtureIntegrity:
    """Verify the fixture files exist and have the expected shape before any CLI test."""

    def test_fixture_directory_exists(self) -> None:
        assert FIXTURE_DIR.is_dir(), f"Fixture not found: {FIXTURE_DIR}"

    def test_metadata_has_two_proteins(self) -> None:
        meta = json.loads((FIXTURE_DIR / "protein_metadata.json").read_text())
        assert meta["version"] == 1
        assert len(meta["proteins"]) == 2
        ids = {p["uniprot_id"] for p in meta["proteins"]}
        assert ids == {"P99001", "P99002"}

    def test_p99001_has_preextracted_csv(self) -> None:
        csv = FIXTURE_DIR / "P99001_delta-masses.csv"
        assert csv.is_file()
        frame = pd.read_csv(csv)
        assert {"Peptide", "Delta.Mass", "Probability"}.issubset(frame.columns)
        assert len(frame) == 4

    def test_psm_run1_has_six_rows_three_passing(self) -> None:
        tsv = FIXTURE_DIR / "Results_MSFragger" / "1" / "psm.tsv"
        assert tsv.is_file()
        frame = pd.read_csv(tsv, sep="\t")
        frame.columns = frame.columns.str.replace(" ", ".", regex=False)
        assert len(frame) == 6, "Run 1 should have 6 rows total (3 pass + 3 intentional failures)"
        passing = frame[
            (frame["Number.of.Missed.Cleavages"] == 0)
            & (frame["Probability"] > 0.9)
            & (frame["Delta.Mass"] > 10.0)
            & (frame["Number.Best.Positions"] == 1)
            & (frame["Ions.Best.Position"] >= frame["Ions.Second.Best.Position"])
            & (frame["Score.Best.Position"] > frame["Score.Second.Best.Position"])
        ]
        assert len(passing) == 3, f"Expected 3 passing rows, got {len(passing)}"

    def test_psm_run2_has_four_rows_three_passing(self) -> None:
        tsv = FIXTURE_DIR / "Results_MSFragger" / "2" / "psm.tsv"
        frame = pd.read_csv(tsv, sep="\t")
        frame.columns = frame.columns.str.replace(" ", ".", regex=False)
        assert len(frame) == 4, "Run 2 should have 4 rows total (3 pass + 1 intentional failure)"
        passing = frame[
            (frame["Number.of.Missed.Cleavages"] == 0)
            & (frame["Probability"] > 0.9)
            & (frame["Delta.Mass"] > 10.0)
            & (frame["Number.Best.Positions"] == 1)
            & (frame["Ions.Best.Position"] >= frame["Ions.Second.Best.Position"])
            & (frame["Score.Best.Position"] > frame["Score.Second.Best.Position"])
        ]
        assert len(passing) == 3, f"Expected 3 passing rows, got {len(passing)}"


class TestCLIValidateAndList:
    def test_validate_returns_valid(self) -> None:
        exit_code, payload, _ = _run_cli(["validate", str(FIXTURE_DIR)])
        assert exit_code == 0
        assert payload["valid"] is True
        assert payload["protein_count"] == 2

    def test_list_returns_both_proteins(self) -> None:
        exit_code, payload, _ = _run_cli(["list", str(FIXTURE_DIR)])
        assert exit_code == 0
        assert payload["count"] == 2
        ids = {p["uniprot_id"] for p in payload["proteins"]}
        assert ids == {"P99001", "P99002"}

    def test_list_p99001_has_delta_masses(self) -> None:
        _, payload, _ = _run_cli(["list", str(FIXTURE_DIR)])
        p99001 = next(p for p in payload["proteins"] if p["uniprot_id"] == "P99001")
        assert p99001["has_delta_masses"] is True

    def test_list_p99002_has_no_preextracted_csv(self) -> None:
        _, payload, _ = _run_cli(["list", str(FIXTURE_DIR)])
        p99002 = next(p for p in payload["proteins"] if p["uniprot_id"] == "P99002")
        assert p99002["has_delta_masses"] is False


class TestCLIExtract:
    def test_extract_p99001_reads_existing_csv(self) -> None:
        """P99001 has a pre-extracted CSV — extract should return source=existing_csv."""
        exit_code, payload, _ = _run_cli(["extract", str(FIXTURE_DIR), "P99001"])
        assert exit_code == 0
        assert payload["source"] == "existing_csv"
        assert payload["rows"] == 4

    def test_extract_p99001_max_preview_rows_flag(self) -> None:
        """--max-preview-rows limits the preview list length (undocumented flag)."""
        _, payload, _ = _run_cli(["extract", str(FIXTURE_DIR), "P99001", "--max-preview-rows", "2"])
        assert len(payload["preview"]) == 2

    def test_extract_p99001_full_preview_by_default(self) -> None:
        """Default --max-preview-rows=10 returns all 4 rows since 4 < 10."""
        _, payload, _ = _run_cli(["extract", str(FIXTURE_DIR), "P99001"])
        assert len(payload["preview"]) == 4

    def test_extract_p99002_runs_extraction_from_psm(self, tmp_path: Path) -> None:
        """P99002 has no pre-extracted CSV — extract should run extraction pipeline."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)

        exit_code, payload, _ = _run_cli(["extract", str(study_copy), "P99002"])
        assert exit_code == 0
        assert payload["source"] == "extraction"
        # 3 rows from run 1 + 3 rows from run 2 → deduplication may reduce count,
        # but at minimum we expect the unique peptide/modification combinations:
        # AVMGDLTPK/M3/~15.995 (both runs), STLQEIPVR/S1/~79.966 (both runs),
        # GAQVIELDIK/K10/~42.011 (run 1 only), IPLVNGTASAK/T8/~14.016 (run 2 only)
        assert payload["rows"] >= 3

    def test_extract_p99002_filters_out_failing_psm_rows(self, tmp_path: Path) -> None:
        """Failing PSM rows (low delta mass, low probability, missed cleavage) must be excluded."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)

        _, payload, _ = _run_cli(["extract", str(study_copy), "P99002"])
        peptides = {row["Peptide"] for row in payload.get("preview", [])}
        # These peptides come from intentionally failing rows and must not appear
        assert "ARFMTPVEK" not in peptides, "Low-probability row leaked through extraction"
        assert "FLQTEVMRK" not in peptides, "Missed-cleavage row leaked through extraction"
        assert "SVTMEPQAK" not in peptides, "Multi-site row leaked through extraction"


class TestCLIRun:
    def test_run_single_pass(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)

        exit_code, payload, stderr_text = _run_cli(["run", str(study_copy), "--model", "stub-model"])
        assert exit_code == 0
        assert payload["model"] == "stub-model"
        assert payload["n_runs"] == 1
        assert "Starting PTM inference" in stderr_text

    def test_run_base_url_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--base-url is passed through to run_inference (undocumented flag)."""
        import shutil

        captured: dict[str, Any] = {}

        def capturing_run(*, base_data_folder: str, model: str, base_url: str | None) -> dict[str, Any]:
            captured["base_url"] = base_url
            return _fake_run_inference(base_data_folder=base_data_folder, model=model, base_url=base_url)

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", capturing_run)

        _run_cli(["run", str(study_copy), "--model", "stub-model", "--base-url", "http://localhost:9999/v1"])
        assert captured["base_url"] == "http://localhost:9999/v1"

    def test_run_multi_pass_with_aggregated_csv_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--n-runs >1 triggers aggregation; --aggregated-csv-path controls output location."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        custom_agg_path = str(tmp_path / "my_aggregated.csv")

        def fake_multi(
            n: int,
            *,
            base_data_folder: str,
            model: str,
            base_url: str | None,
            aggregated_csv_path: str | None,
        ) -> tuple[list[str], pd.DataFrame]:
            assert n == 2
            assert aggregated_csv_path == custom_agg_path
            frame = pd.DataFrame([{"Protein.ID": "P99001", "PTM.NAME": "Oxidation", "Run.Index": 0}])
            return ([str(study_copy / "output" / "run1")], frame)

        monkeypatch.setattr(cli, "run_and_aggregate_multiple_inferences", fake_multi)

        exit_code, payload, _ = _run_cli(
            [
                "run",
                str(study_copy),
                "--n-runs",
                "2",
                "--model",
                "stub-model",
                "--aggregated-csv-path",
                custom_agg_path,
            ]
        )
        assert exit_code == 0
        assert payload["n_runs"] == 2
        assert payload["aggregated_rows"] == 1


class TestCLIResults:
    def test_results_study_folder_lists_runs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)

        # Create a run to list
        _run_cli(["run", str(study_copy), "--model", "stub-model"])

        exit_code, payload, _ = _run_cli(["results", "--study-folder", str(study_copy)])
        assert exit_code == 0
        assert len(payload["available_runs"]) >= 1
        assert all(r.startswith("output_") for r in payload["available_runs"])

    def test_results_run_dir_previews_predictions(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--run-dir points to a specific run directory (undocumented flag in quickstart)."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)

        _, run_payload, _ = _run_cli(["run", str(study_copy), "--model", "stub-model"])
        run_dir = run_payload["run_dir"]

        exit_code, payload, _ = _run_cli(["results", "--run-dir", run_dir])
        assert exit_code == 0
        assert payload["run_dir"] == run_dir
        assert len(payload["predictions"]) == 2  # P99001 + P99002

    def test_results_max_preview_records_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--max-preview-records limits prediction preview length (undocumented flag)."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)

        _, run_payload, _ = _run_cli(["run", str(study_copy), "--model", "stub-model"])
        run_dir = run_payload["run_dir"]

        _, payload, _ = _run_cli(["results", "--run-dir", run_dir, "--max-preview-records", "0"])
        for protein_entry in payload["predictions"]:
            assert protein_entry["preview"] == []


class TestCLIAnalyze:
    def test_analyze_run_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)

        _, run_payload, _ = _run_cli(["run", str(study_copy), "--model", "stub-model"])
        run_dir = run_payload["run_dir"]

        exit_code, payload, _ = _run_cli(["analyze", "--run-dir", run_dir, "--no-plots"])
        assert exit_code == 0
        assert payload["analysis_mode"] == "run_dir"
        assert Path(payload["output"]["summary_json"]).is_file()

    def test_analyze_no_plots_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--no-plots suppresses PNG generation (mentioned in docs but not shown in examples)."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)

        _, run_payload, _ = _run_cli(["run", str(study_copy), "--model", "stub-model"])

        _, payload, _ = _run_cli(["analyze", "--run-dir", run_payload["run_dir"], "--no-plots"])
        assert payload["output"]["plots"] == []

    def test_analyze_top_k_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--top-k controls how many top sites appear in summaries (undocumented flag)."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)

        _, run_payload, _ = _run_cli(["run", str(study_copy), "--model", "stub-model"])

        exit_code, payload, _ = _run_cli(["analyze", "--run-dir", run_payload["run_dir"], "--no-plots", "--top-k", "5"])
        assert exit_code == 0

    def test_analyze_output_dir_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """--output-dir overrides where analysis_<timestamp>/ is written (undocumented flag)."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)
        monkeypatch.setattr(cli, "run_inference", _fake_run_inference)
        custom_output = tmp_path / "my_analysis_output"

        _, run_payload, _ = _run_cli(["run", str(study_copy), "--model", "stub-model"])

        exit_code, payload, _ = _run_cli(
            [
                "analyze",
                "--run-dir",
                run_payload["run_dir"],
                "--no-plots",
                "--output-dir",
                str(custom_output),
            ]
        )
        assert exit_code == 0
        assert Path(payload["output"]["summary_json"]).is_file()
        assert str(custom_output) in payload["output"]["summary_json"]

    def test_analyze_aggregated_csv(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)

        # Build a small aggregated CSV using the same shape as the real pipeline produces
        agg_csv = tmp_path / "aggregated_predictions_example.csv"
        pd.DataFrame(
            [
                {
                    "PEPTIDE": "AAMTGLTK",
                    "PTM.Best.Positions": '["M3"]',
                    "PTM.DeltaMass": 15.995,
                    "PTM.Impact.In.Enzyme.Function": "Stub",
                    "PTM.NAME": "Oxidation",
                    "PTM.Protein.Position": 103,
                    "Source.File": f"run{i}.csv",
                    "Run.Index": i,
                    "Protein.ID": "P99001",
                }
                for i in range(3)
            ]
        ).to_csv(agg_csv, index=False)

        exit_code, payload, _ = _run_cli(["analyze", "--aggregated-csv", str(agg_csv), "--no-plots"])
        assert exit_code == 0
        assert payload["analysis_mode"] == "aggregated_csv"
        assert Path(payload["output"]["summary_json"]).is_file()


# ---------------------------------------------------------------------------
# Live LLM tier — requires OPENAI_API_KEY
# Run with:  uv run pytest -m llm -v
# ---------------------------------------------------------------------------

# Load .env once at import time so os.getenv picks it up for the skipif check.
load_dotenv(Path(__file__).parent.parent / ".env", override=False)

_REQUIRED_PREDICTION_COLS = {
    "PEPTIDE",
    "PTM.NAME",
    "PTM.DeltaMass",
    "PTM.Protein.Position",
}


@pytest.mark.llm
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live LLM test",
)
class TestLLMInference:
    """Live LLM call against the example fixture.

    No monkeypatching of the LLM layer.  The test only validates structure
    (file existence, required schema columns).

    Run with::

        uv run pytest -m llm -v
    """

    def test_run_produces_valid_prediction_schema(self, tmp_path: Path) -> None:
        """Call ``ptmdiscoverer run`` with a real LLM and verify output schema."""
        import shutil

        study_copy = tmp_path / "study"
        shutil.copytree(FIXTURE_DIR, study_copy)

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        # base_url is optional; pass only when explicitly configured
        extra_args: list[str] = []
        if os.getenv("OPENAI_BASE_URL"):
            extra_args += ["--base-url", os.environ["OPENAI_BASE_URL"]]

        exit_code, run_payload, stderr_str = _run_cli(["run", str(study_copy), "--model", model_name, *extra_args])
        assert exit_code == 0, f"ptmdiscoverer run failed:\n{stderr_str}"

        run_dir = Path(run_payload["run_dir"])
        assert run_dir.is_dir(), "run_dir was not created"

        prediction_csvs = list(run_dir.glob("*.csv"))
        assert prediction_csvs, "No prediction CSV files produced under run_dir"

        non_empty_found = False
        for csv_path in prediction_csvs:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            non_empty_found = True
            missing = _REQUIRED_PREDICTION_COLS - set(df.columns)
            assert not missing, f"{csv_path.name} is missing required columns: {missing}"
            assert df["PTM.NAME"].notna().all(), f"{csv_path.name}: PTM.NAME contains null values"
            assert (df["PTM.DeltaMass"] != 0).any(), f"{csv_path.name}: all PTM.DeltaMass values are zero"

        assert non_empty_found, "All prediction CSVs are empty — real LLM call may have failed silently"

        # Validate that the analyze subcommand can consume the run output.
        exit_code_analyze, analyze_payload, analyze_stderr = _run_cli(
            ["analyze", "--run-dir", str(run_dir), "--no-plots"]
        )
        assert exit_code_analyze == 0, f"ptmdiscoverer analyze failed:\n{analyze_stderr}"
        assert Path(analyze_payload["output"]["summary_json"]).is_file()
