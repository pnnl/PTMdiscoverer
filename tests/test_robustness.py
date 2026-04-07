"""Tests for study folder validation, metadata entry validation,
prediction file return semantics, and timestamp uniqueness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ptmdiscoverer.inference as inf
from ptmdiscoverer.utils import load_protein_metadata

# --- A: validate_study_folder ---


@pytest.mark.unit
def test_validate_study_folder_valid(tmp_path: Path) -> None:
    """Valid study folder passes without error."""
    (tmp_path / "base_prompt.txt").write_text("prompt", encoding="utf-8")
    (tmp_path / "protein_metadata.json").write_text('{"version":1,"proteins":[]}', encoding="utf-8")
    inf.validate_study_folder(str(tmp_path))


@pytest.mark.unit
def test_validate_study_folder_missing_prompt(tmp_path: Path) -> None:
    """Missing base_prompt.txt raises FileNotFoundError."""
    (tmp_path / "protein_metadata.json").write_text('{"version":1,"proteins":[]}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="base_prompt.txt"):
        inf.validate_study_folder(str(tmp_path))


@pytest.mark.unit
def test_validate_study_folder_missing_metadata(tmp_path: Path) -> None:
    """Missing protein_metadata.json raises FileNotFoundError."""
    (tmp_path / "base_prompt.txt").write_text("prompt", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="protein_metadata.json"):
        inf.validate_study_folder(str(tmp_path))


@pytest.mark.unit
def test_validate_study_folder_not_a_directory(tmp_path: Path) -> None:
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        inf.validate_study_folder(str(tmp_path / "nope"))


@pytest.mark.unit
def test_validate_study_folder_checks_msfragger_when_labels_present(tmp_path: Path) -> None:
    """When dataset_labels.csv exists, Results_MSFragger/ must be a directory."""
    (tmp_path / "base_prompt.txt").write_text("prompt", encoding="utf-8")
    (tmp_path / "protein_metadata.json").write_text('{"version":1,"proteins":[]}', encoding="utf-8")
    (tmp_path / "dataset_labels.csv").write_text("MSRUNID\n1\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Results_MSFragger"):
        inf.validate_study_folder(str(tmp_path))


@pytest.mark.unit
def test_validate_study_folder_skips_msfragger_without_labels(tmp_path: Path) -> None:
    """Without dataset_labels.csv, Results_MSFragger/ is not required."""
    (tmp_path / "base_prompt.txt").write_text("prompt", encoding="utf-8")
    (tmp_path / "protein_metadata.json").write_text('{"version":1,"proteins":[]}', encoding="utf-8")
    inf.validate_study_folder(str(tmp_path))


@pytest.mark.integration
def test_run_inference_calls_validate(tmp_path: Path) -> None:
    """run_inference fails fast on an invalid study folder."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        inf.run_inference(base_data_folder=str(tmp_path / "missing"))


# --- B: _write_prediction_files returns filtered list ---


@pytest.mark.unit
def test_write_prediction_files_returns_kept(tmp_path: Path) -> None:
    """_write_prediction_files returns the kept records without mutating the input."""
    records = [
        {"PEPTIDE": "ABC", "PTM.Protein.Position": 42},
        {"PEPTIDE": "DEF", "PTM.Protein.Position": -1},
    ]
    original_len = len(records)
    pred_base = str(tmp_path / "pred")
    kept = inf._write_prediction_files(records, pred_base)
    assert len(kept) == 1
    assert kept[0]["PEPTIDE"] == "ABC"
    assert len(records) == original_len, "Input list should not be mutated"


@pytest.mark.unit
def test_write_prediction_files_returns_empty_for_all_invalid(tmp_path: Path) -> None:
    """All-invalid input returns empty list, no files written."""
    records = [{"PEPTIDE": "X", "PTM.Protein.Position": -1}]
    kept = inf._write_prediction_files(records, str(tmp_path / "pred"))
    assert kept == []
    assert not (tmp_path / "pred.json").exists()


# --- C: load_protein_metadata validates entries ---


@pytest.mark.unit
def test_load_metadata_skips_entry_missing_uniprot_id(tmp_path: Path) -> None:
    """Entries without uniprot_id are dropped with a warning."""
    meta = {
        "version": 1,
        "proteins": [
            {"uniprot_id": "P123", "name": "Good"},
            {"name": "NoID"},
        ],
    }
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    proteins = load_protein_metadata(str(p))
    assert len(proteins) == 1
    assert proteins[0]["uniprot_id"] == "P123"


@pytest.mark.unit
def test_load_metadata_skips_non_string_uniprot_id(tmp_path: Path) -> None:
    """Entries with non-string uniprot_id are dropped."""
    meta = {
        "version": 1,
        "proteins": [
            {"uniprot_id": 123, "name": "BadType"},
            {"uniprot_id": "P456", "name": "Good"},
        ],
    }
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    proteins = load_protein_metadata(str(p))
    assert len(proteins) == 1
    assert proteins[0]["uniprot_id"] == "P456"


@pytest.mark.unit
def test_load_metadata_skips_non_dict_entries(tmp_path: Path) -> None:
    """Non-dict entries in proteins list are dropped."""
    meta = {
        "version": 1,
        "proteins": [
            {"uniprot_id": "P789", "name": "Good"},
            "not_a_dict",
            42,
        ],
    }
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    proteins = load_protein_metadata(str(p))
    assert len(proteins) == 1


# --- D: format_timestamp uniqueness in tight loop ---


@pytest.mark.unit
def test_format_timestamp_sequential_uniqueness() -> None:
    """Consecutive calls to format_timestamp produce distinct values."""
    timestamps = [inf.format_timestamp() for _ in range(20)]
    assert len(set(timestamps)) == 20, f"Expected 20 unique timestamps, got {len(set(timestamps))}"
