"""Pytest configuration and shared test helpers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class _FakeStreamChunk:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(delta=SimpleNamespace(content=content))]


class FakeStream:
    """Context manager simulating an OpenAI streaming response for tests.

    Usage in mocks::

        completions=SimpleNamespace(create=lambda **_: FakeStream(content))
    """

    def __init__(self, content: str = "") -> None:
        self._content = content

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __iter__(self):  # type: ignore[override]
        if self._content:
            yield _FakeStreamChunk(self._content)


def make_test_study(tmp_path: Path, *, with_psm: bool = True) -> Path:
    """Create a minimal PTM study folder for CLI and MCP tests."""
    base = tmp_path / "study"
    base.mkdir()
    (base / "output").mkdir()
    (base / "base_prompt.txt").write_text("Base prompt", encoding="utf-8")
    metadata = {
        "version": 1,
        "proteins": [
            {"uniprot_id": "P12345", "name": "ProtA", "delta_mass_file": "P12345_delta-masses.csv"},
            {"uniprot_id": "Q99999", "name": "ProtB", "delta_mass_file": ""},
        ],
    }
    (base / "protein_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame({"Peptide": ["ABCK"], "Delta.Mass": [15.2]}).to_csv(
        base / "P12345_delta-masses.csv",
        index=False,
    )
    if with_psm:
        (base / "dataset_labels.csv").write_text("MSRUNID\n1\n", encoding="utf-8")
        psm_dir = base / "Results_MSFragger" / "1"
        psm_dir.mkdir(parents=True)
        psm_content = (
            "Spectrum\tPeptide\tCharge\tCalculated Peptide Mass\tCalculated M/Z\t"
            "Delta Mass\tBest Positions\tNumber Best Positions\t"
            "Ions Best Position\tIons Second Best Position\t"
            "Score Best Position\tScore Second Best Position\t"
            "Probability\tNumber of Missed Cleavages\tProtein ID\tRetention\tIs Unique\n"
            "s1\tPEPTIDEK\t2\t1000\t500\t15.2\t1\t1\t10\t5\t60\t10\t0.95\t0\tQ99999\t600\tTrue\n"
        )
        (psm_dir / "psm.tsv").write_text(psm_content, encoding="utf-8")
    return base


def write_simple_aggregated_csv(
    tmp_path: Path,
    *,
    protein_id: str = "P12345",
    run_count: int = 3,
    source_suffix: str = ".csv",
) -> Path:
    """Write a small aggregated predictions fixture for analysis-facing tests."""
    path = tmp_path / "aggregated_predictions_fixture.csv"
    pd.DataFrame(
        [
            {
                "PEPTIDE": "AAAK",
                "PTM.Best.Positions": "[1]",
                "PTM.DeltaMass": 15.99,
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "PTM.NAME": "Oxidation",
                "PTM.Protein.Position": 5,
                "Source.File": f"run{run_index}{source_suffix}",
                "Run.Index": run_index,
                "Protein.ID": protein_id,
            }
            for run_index in range(run_count)
        ]
    ).to_csv(path, index=False)
    return path


def write_simple_run_dir(tmp_path: Path) -> Path:
    """Write a minimal run directory with one prediction CSV."""
    run_dir = tmp_path / "output_2024-01-01-00-00-00-000"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "PEPTIDE": "ABCK",
                "PTM.Best.Positions": "[1]",
                "PTM.DeltaMass": 15.99,
                "PTM.Impact.In.Enzyme.Function": "Unknown",
                "PTM.NAME": "Oxidation",
                "PTM.Protein.Position": 5,
            }
        ]
    ).to_csv(run_dir / "ptm_predictions_P12345_2024-01-01-00-00-00-000.csv", index=False)
    return run_dir
