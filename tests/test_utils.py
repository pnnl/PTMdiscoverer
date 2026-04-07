"""Tests for delta-mass extraction utilities."""

from pathlib import Path

import pandas as pd
import pytest

from ptmdiscoverer.utils import extract_delta_masses


# Helper to build a fake psm.tsv DataFrame
def make_psm_rows(**overrides):  # noqa: D401 - simple factory
    """Return a dict representing one PSM row (modifiable via overrides)."""
    base = {
        "Spectrum": "spec1",
        "Peptide": "PEPTIDEK",
        "Charge": 2,
        "Calculated.Peptide.Mass": 1000.0,
        "Calculated.M/Z": 500.0,
        "Delta.Mass": 15.2345,
        "Best.Positions": 1,
        "Number.Best.Positions": 1,
        "Ions.Best.Position": 10,
        "Ions.Second.Best.Position": 5,
        "Score.Best.Position": 50.0,
        "Score.Second.Best.Position": 10.0,
        "Probability": 0.95,
        "Number.of.Missed.Cleavages": 0,
        "Protein.ID": "P12345",
        "Retention": 600.0,
        "Is.Unique": True,
        "Protein.Start": 100,
    }
    base.update(overrides)
    return base


def write_psm(path: Path, rows):  # noqa: D401 - helper
    """Write a minimal tab-delimited psm.tsv given iterable rows."""
    import csv

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.unit
def test_returns_empty_when_no_files(tmp_path: Path) -> None:  # noqa: D401
    """Return empty DataFrame when no PSM files exist."""
    msruns = pd.DataFrame({"MSRUNID": [1, 2]})
    df = extract_delta_masses("P12345", msruns, base_path=str(tmp_path))
    assert df.empty


@pytest.mark.unit
def test_filters_and_rounds_delta_mass(tmp_path: Path) -> None:  # noqa: D401
    """Filter invalid rows and round delta mass + compute RT."""
    msruns = pd.DataFrame({"MSRUNID": [1]})
    run_dir = tmp_path / "Results_MSFragger" / "1"
    run_dir.mkdir(parents=True)
    psm_file = run_dir / "psm.tsv"

    # Create a valid row and a few invalid ones for filtering checks
    valid = make_psm_rows()
    wrong_protein = make_psm_rows(**{"Protein.ID": "OTHER"})
    low_prob = make_psm_rows(Probability=0.5)
    high_missed_cleave = make_psm_rows(**{"Number.of.Missed.Cleavages": 1})
    low_delta = make_psm_rows(**{"Delta.Mass": 5.0})  # should be filtered (<=10)

    write_psm(psm_file, [valid, wrong_protein, low_prob, high_missed_cleave, low_delta])

    # Act
    df = extract_delta_masses("P12345", msruns, base_path=str(tmp_path))

    # Assert
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Delta.Mass"] == 15.234  # rounded to 3 decimals
    assert row["RT"] == 10.0  # 600 / 60
    assert row["MSRUNID"] == 1
    # Best.Positions=1, Protein.Start=100 -> Positions.In.Protein = 100 + 1 - 1 = 100
    assert row["Positions.In.Protein"] == 100.0


@pytest.mark.unit
def test_competition_filters(tmp_path: Path) -> None:  # noqa: D401
    """Require superiority in ions and scores for localization competition."""
    msruns = pd.DataFrame({"MSRUNID": [1]})
    run_dir = tmp_path / "Results_MSFragger" / "1"
    run_dir.mkdir(parents=True)
    psm_file = run_dir / "psm.tsv"

    fail_ions = make_psm_rows(**{"Ions.Best.Position": 3, "Ions.Second.Best.Position": 5})
    fail_scores = make_psm_rows(**{"Score.Best.Position": 5.0, "Score.Second.Best.Position": 10.0})
    good = make_psm_rows(Spectrum="specOK")

    write_psm(psm_file, [fail_ions, fail_scores, good])

    df = extract_delta_masses("P12345", msruns, base_path=str(tmp_path))
    assert len(df) == 1
    assert df.iloc[0]["Spectrum"] == "specOK"
    assert df.iloc[0]["Positions.In.Protein"] == 100.0


@pytest.mark.unit
def test_deduplication(tmp_path: Path) -> None:  # noqa: D401
    """De-duplicate rows by key subset across spectra variants."""
    msruns = pd.DataFrame({"MSRUNID": [1]})
    run_dir = tmp_path / "Results_MSFragger" / "1"
    run_dir.mkdir(parents=True)
    psm_file = run_dir / "psm.tsv"

    base = make_psm_rows(Spectrum="dup1")
    duplicate = make_psm_rows(Spectrum="dup2")  # differs only in Spectrum (not in subset keys)

    write_psm(psm_file, [base, duplicate])

    df = extract_delta_masses("P12345", msruns, base_path=str(tmp_path))
    # Only one should remain after drop_duplicates subset
    assert len(df) == 1
    assert df.iloc[0]["Positions.In.Protein"] == 100.0


@pytest.mark.unit
def test_multiple_runs_concat(tmp_path: Path) -> None:  # noqa: D401
    """Concatenate qualifying rows across multiple MS runs."""
    msruns = pd.DataFrame({"MSRUNID": [1, 2]})
    for run_id in [1, 2]:
        run_dir = tmp_path / "Results_MSFragger" / str(run_id)
        run_dir.mkdir(parents=True)
        psm_file = run_dir / "psm.tsv"
        write_psm(psm_file, [make_psm_rows(Spectrum=f"s{run_id}")])
    df = extract_delta_masses("P12345", msruns, base_path=str(tmp_path))
    assert set(df["MSRUNID"]) == {1, 2}
    assert len(df) == 2
    assert set(df["Positions.In.Protein"]) == {100.0}


@pytest.mark.unit
def test_select_columns_subset(tmp_path: Path) -> None:
    """Honor custom select_columns parameter limiting returned columns."""
    msruns = pd.DataFrame({"MSRUNID": [1]})
    run_dir = tmp_path / "Results_MSFragger" / "1"
    run_dir.mkdir(parents=True)
    psm_file = run_dir / "psm.tsv"
    write_psm(psm_file, [make_psm_rows()])
    subset_cols = ("Peptide", "Delta.Mass")
    df = extract_delta_masses("P12345", msruns, base_path=str(tmp_path), select_columns=subset_cols)
    assert list(df.columns) == list(subset_cols)

    # Ensure default column selection still includes Peptide/Delta.Mass when not specifying subset
    df_default = extract_delta_masses("P12345", msruns, base_path=str(tmp_path))
    assert {"Peptide", "Delta.Mass"}.issubset(df_default.columns)
