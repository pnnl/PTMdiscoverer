"""Tests for the standalone PTMdiscoverer CLI."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from conftest import make_test_study, write_simple_aggregated_csv

import ptmdiscoverer.cli as cli


def _run_cli(args: list[str]) -> tuple[int, dict[str, Any], str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        exit_code = cli.main(args)
    return exit_code, cast(dict[str, Any], json.loads(stdout_buf.getvalue())), stderr_buf.getvalue()


@pytest.mark.unit
def test_cli_list_outputs_proteins(tmp_path: Path) -> None:
    base = make_test_study(tmp_path)
    exit_code, payload, stderr_text = _run_cli(["list", str(base)])

    assert exit_code == 0
    assert payload["count"] == 2
    assert payload["proteins"][0]["uniprot_id"] == "P12345"
    assert stderr_text == ""


@pytest.mark.unit
def test_cli_validate_returns_structured_report(tmp_path: Path) -> None:
    base = make_test_study(tmp_path)
    exit_code, payload, _ = _run_cli(["validate", str(base)])

    assert exit_code == 0
    assert payload["valid"] is True
    assert payload["protein_count"] == 2
    assert payload["checks"]["study_folder_exists"] is True


@pytest.mark.unit
def test_cli_extract_existing_csv(tmp_path: Path) -> None:
    base = make_test_study(tmp_path)
    exit_code, payload, _ = _run_cli(["extract", str(base), "P12345"])

    assert exit_code == 0
    assert payload["source"] == "existing_csv"
    assert payload["rows"] == 1


@pytest.mark.unit
def test_cli_run_multiple_uses_aggregation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = make_test_study(tmp_path)

    def fake_multi_run(
        n: int,
        *,
        base_data_folder: str,
        model: str,
        base_url: str | None,
        aggregated_csv_path: str | None,
    ) -> tuple[list[str], pd.DataFrame]:
        assert n == 3
        assert base_data_folder == str(base)
        assert model == "o4-mini-project"
        frame = pd.DataFrame([{"Protein.ID": "P12345", "PTM.NAME": "Mod", "Run.Index": 0}])
        return ([str(base / "output" / "output_1")], frame)

    monkeypatch.setattr(cli, "run_and_aggregate_multiple_inferences", fake_multi_run)
    exit_code, payload, stderr_text = _run_cli(
        [
            "run",
            str(base),
            "--n-runs",
            "3",
            "--model",
            "o4-mini-project",
        ]
    )

    assert exit_code == 0
    assert payload["n_runs"] == 3
    assert payload["aggregated_rows"] == 1
    assert "Starting PTM inference" in stderr_text
    assert str(base) in stderr_text
    assert "o4-mini-project" in stderr_text
    assert "3 runs" in stderr_text
    assert "Runtime depends on the model, prompt size, and number of runs." in stderr_text


@pytest.mark.unit
def test_cli_results_lists_runs(tmp_path: Path) -> None:
    base = make_test_study(tmp_path)
    (base / "output" / "output_2024-01-01").mkdir(parents=True)
    (base / "output" / "output_2024-01-02").mkdir(parents=True)

    exit_code, payload, _ = _run_cli(["results", "--study-folder", str(base)])

    assert exit_code == 0
    assert payload["available_runs"] == ["output_2024-01-01", "output_2024-01-02"]


@pytest.mark.unit
def test_cli_rejects_zero_runs(tmp_path: Path) -> None:
    base = make_test_study(tmp_path)
    exit_code, payload, stderr_text = _run_cli(["run", str(base), "--n-runs", "0"])

    assert exit_code == 1
    assert "error" in payload
    assert stderr_text == ""


@pytest.mark.unit
def test_cli_run_uses_openai_model_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = make_test_study(tmp_path)
    monkeypatch.setenv("OPENAI_MODEL", "o4-mini-project")
    assert cli._default_model() == "o4-mini-project"

    def fake_run_inference(*, base_data_folder: str, model: str, base_url: str | None) -> dict[str, object]:
        assert base_data_folder == str(base)
        assert model == "o4-mini-project"
        assert base_url is None
        return {"run_dir": str(base / "output" / "output_fake"), "proteins_processed": 1, "total_ptm_records": 0}

    monkeypatch.setattr(cli, "run_inference", fake_run_inference)
    exit_code, payload, stderr_text = _run_cli(["run", str(base)])

    assert exit_code == 0
    assert payload["model"] == "o4-mini-project"
    assert str(base) in stderr_text
    assert "o4-mini-project" in stderr_text
    assert "1 run" in stderr_text


@pytest.mark.unit
def test_cli_analyze_writes_report_from_aggregated_csv(tmp_path: Path) -> None:
    aggregated_csv = write_simple_aggregated_csv(tmp_path)

    exit_code, payload, stderr_text = _run_cli(
        [
            "analyze",
            "--aggregated-csv",
            str(aggregated_csv),
            "--output-dir",
            str(tmp_path),
            "--no-plots",
        ]
    )

    assert exit_code == 0
    assert payload["analysis_mode"] == "aggregated_csv"
    assert payload["output"]["plots"] == []
    assert Path(payload["output"]["summary_json"]).is_file()
    assert stderr_text == ""
