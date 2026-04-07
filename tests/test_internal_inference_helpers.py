"""Internal inference helper tests (timestamp + JSON parsing + process_protein)."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from conftest import FakeStream

import ptmdiscoverer.inference as inf
from ptmdiscoverer.utils import parse_first_json_block


@pytest.mark.unit
def test_format_timestamp_pattern() -> None:
    """Generate millisecond timestamp in expected pattern."""
    ts = inf.format_timestamp()
    assert re.match(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}$", ts)


@pytest.mark.unit
def test_parse_llm_json_block_variants() -> None:
    """Parse fenced JSON blocks returning only dict list entries or empty."""
    block_list = """```json\n[{"a":1},{"b":2},3,"x"]\n```"""
    assert len(parse_first_json_block(block_list)) == 2  # only dict entries kept
    block_obj = """```json\n{"a":1}\n```"""
    assert parse_first_json_block(block_obj)[0]["a"] == 1
    missing = "No fences here"
    assert parse_first_json_block(missing) == []
    invalid = """```json\n{invalid}\n```"""
    assert parse_first_json_block(invalid) == []


@pytest.mark.unit
def test_process_protein_skip_due_to_missing_labels(tmp_path: Path) -> None:
    """Skip processing when msrun table empty and no delta mass file."""
    protein = {"uniprot_id": "PXXX", "name": "Example", "delta_mass_file": ""}
    msrun_table = pd.DataFrame()
    log_buf = io.StringIO()
    run_dir = tmp_path / "output"
    run_dir.mkdir()

    class DummyClient:
        def __init__(self):  # noqa: D401
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream()))

    changed, records = inf.process_protein(
        inf.InferenceContext(
            client=DummyClient(),
            base_prompt="BASE",
            model="m",
            timestamp="T",
            base_data_folder=str(tmp_path),
            run_dir=str(run_dir),
            log_handle=log_buf,
            msrun_table=msrun_table,
        ),
        protein,
    )
    assert changed is False and records == []


@pytest.mark.unit
def test_process_protein_with_existing_delta_file(tmp_path: Path) -> None:
    """Use existing delta file and produce prediction artifacts."""
    base = tmp_path
    delta_path = base / "P999_delta-masses.csv"
    pd.DataFrame({"Peptide": ["ABC"], "Delta.Mass": [12.3]}).to_csv(delta_path, index=False)
    protein = {"uniprot_id": "P999", "name": "Prot", "delta_mass_file": str(delta_path)}
    msrun_table = pd.DataFrame()
    log_buf = io.StringIO()
    run_dir = base / "run"
    run_dir.mkdir()

    class FakeChoices:
        def __init__(self, content: str):  # noqa: D401
            self.message = SimpleNamespace(content=content)

    fenced_json = """Here\n```json\n[{"PEPTIDE":"ABC","PTM.DeltaMass":12.3,"PTM.Best.Positions":["1"],"PTM.NAME":"X","PTM.Impact.In.Enzyme.Function":"?","PTM.Protein.Position":9}]\n```"""

    class DummyClient:
        def __init__(self):  # noqa: D401
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_: FakeStream(fenced_json)))

    changed, records = inf.process_protein(
        inf.InferenceContext(
            client=DummyClient(),
            base_prompt="BASE",
            model="m",
            timestamp="2020-01-01-00-00-00-000",
            base_data_folder=str(base),
            run_dir=str(run_dir),
            log_handle=log_buf,
            msrun_table=msrun_table,
        ),
        protein,
    )
    assert changed is False
    assert len(records) == 1 and records[0]["PEPTIDE"] == "ABC" and records[0]["PTM.Protein.Position"] == 9
    pred_jsons = list(run_dir.glob("ptm_predictions_P999_*.json"))
    assert pred_jsons
    assert json.loads(pred_jsons[0].read_text())[0]["PTM.DeltaMass"] == 12.3
