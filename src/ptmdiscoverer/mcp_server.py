"""MCP server exposing PTMdiscoverer tools.

This module registers six tools on a FastMCP server instance:

- ``validate_study_folder_tool`` – validate a study folder before inference
- ``list_proteins``  – enumerate proteins in a study folder
- ``extract_protein_delta_masses`` – extract or read delta-mass data
- ``run_ptm_inference`` – run LLM-based PTM annotation (single or multi-run)
- ``get_run_results``  – retrieve prediction artifacts from a completed run
- ``analyze_run_results`` – generate a deterministic analysis report

Launch with::

    ptmdiscoverer-mcp          # if installed via [mcp] extra
    python -m ptmdiscoverer.mcp_server   # direct invocation
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any

import openai
import pandas as pd
from mcp.server.fastmcp import FastMCP

from ptmdiscoverer.inference import (
    assess_study_folder,
    run_and_aggregate_multiple_inferences,
    run_inference,
    validate_study_folder,
)
from ptmdiscoverer.reporting import analyze_run_results as analyze_run_results_report
from ptmdiscoverer.utils import extract_delta_masses, load_protein_metadata

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "ptmdiscoverer",
    instructions="Tools for LLM-assisted post-translational modification discovery in proteomics data.",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_client() -> openai.OpenAI:
    """Construct an OpenAI-compatible client from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Pass it via the MCP Inspector 'Environment Variables' panel, "
            "a .env file, or export it in your shell."
        )
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def _resolve_protein(proteins: list[dict[str, Any]], uniprot_id: str) -> dict[str, Any] | None:
    """Find a protein entry by uniprot_id (case-insensitive)."""
    uid = uniprot_id.upper()
    for p in proteins:
        if str(p.get("uniprot_id", "")).upper() == uid:
            return p
    return None


# ---------------------------------------------------------------------------
# Tool: validate_study_folder_tool
# ---------------------------------------------------------------------------


@mcp.tool()
async def validate_study_folder_tool(study_folder: str) -> dict[str, Any]:
    """Validate a PTM study folder and return structured feedback.

    Args:
        study_folder: Path to the study directory.
    """
    return assess_study_folder(study_folder)


# ---------------------------------------------------------------------------
# Tool: list_proteins
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_proteins(study_folder: str) -> dict[str, Any]:
    """List proteins defined in a study folder's metadata.

    Returns each protein's uniprot_id, name, and whether a delta-mass
    CSV is already available.

    Args:
        study_folder: Path to the study directory containing protein_metadata.json.
    """
    try:
        validate_study_folder(study_folder)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    metadata_path = os.path.join(study_folder, "protein_metadata.json")
    proteins = load_protein_metadata(metadata_path)

    result = []
    for p in proteins:
        dmf = p.get("delta_mass_file", "")
        if dmf and not os.path.isabs(dmf):
            dmf = os.path.join(study_folder, dmf)
        has_dm = bool(dmf and os.path.isfile(dmf))
        result.append(
            {
                "uniprot_id": p["uniprot_id"],
                "name": p.get("name", ""),
                "has_delta_masses": has_dm,
                "delta_mass_file": dmf if has_dm else None,
            }
        )
    return {"proteins": result, "study_folder": study_folder}


# ---------------------------------------------------------------------------
# Tool: extract_protein_delta_masses
# ---------------------------------------------------------------------------


@mcp.tool()
async def extract_protein_delta_masses(
    study_folder: str,
    uniprot_id: str,
    *,
    max_preview_rows: int = 10,
) -> dict[str, Any]:
    """Extract or read delta-mass data for a specific protein.

    If a pre-computed CSV exists it is read directly. Otherwise, extraction
    from MSFragger PSM data is attempted.

    Args:
        study_folder: Path to the study directory.
        uniprot_id: UniProt accession of the target protein.
        max_preview_rows: Number of rows to include in the preview (default 10).
    """
    try:
        validate_study_folder(study_folder)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    metadata_path = os.path.join(study_folder, "protein_metadata.json")
    proteins = load_protein_metadata(metadata_path)
    protein = _resolve_protein(proteins, uniprot_id)
    if protein is None:
        return {"error": f"Protein '{uniprot_id}' not found in metadata."}

    # Try existing CSV first
    dmf = protein.get("delta_mass_file", "")
    if dmf and not os.path.isabs(dmf):
        dmf = os.path.join(study_folder, dmf)
    if dmf and os.path.isfile(dmf):
        try:
            df = pd.read_csv(dmf)
            return {
                "uniprot_id": uniprot_id,
                "source": "existing_csv",
                "file": dmf,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head(max_preview_rows).to_dict(orient="records"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Failed reading {dmf}: {exc}"}

    # Attempt extraction from PSM data
    labels_path = os.path.join(study_folder, "dataset_labels.csv")
    if not os.path.isfile(labels_path):
        return {"error": f"No delta-mass CSV and no dataset_labels.csv in '{study_folder}'."}

    try:
        msrun_table = pd.read_csv(labels_path)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed reading dataset_labels.csv: {exc}"}

    extracted = extract_delta_masses(uniprot_id, msrun_table, study_folder)
    if extracted.empty:
        return {
            "uniprot_id": uniprot_id,
            "source": "extraction",
            "rows": 0,
            "message": "No qualifying delta masses found in PSM data.",
        }

    # Persist for future use
    out_path = os.path.join(study_folder, f"{uniprot_id}_delta-masses.csv")
    extracted.to_csv(out_path, index=False)
    return {
        "uniprot_id": uniprot_id,
        "source": "extraction",
        "file": out_path,
        "rows": len(extracted),
        "columns": list(extracted.columns),
        "preview": extracted.head(max_preview_rows).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Tool: run_ptm_inference
# ---------------------------------------------------------------------------


@mcp.tool()
async def run_ptm_inference(
    study_folder: str,
    *,
    n_runs: int = 1,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    """Run LLM-based PTM inference over all proteins in a study folder.

    With n_runs=1, runs a single inference pass. With n_runs>1, runs
    multiple passes and aggregates the results.

    Args:
        study_folder: Path to the study directory.
        n_runs: Number of inference runs (default 1). Use >1 for consensus.
        model: LLM model identifier.
    """
    try:
        validate_study_folder(study_folder)
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    try:
        client = _get_client()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Failed to create API client: {exc}"}

    if n_runs <= 1:
        result = run_inference(
            base_data_folder=study_folder,
            model=model,
            client=client,
        )
        return {
            "n_runs": 1,
            **result,
        }

    run_dirs, aggregated = run_and_aggregate_multiple_inferences(
        n_runs,
        base_data_folder=study_folder,
        model=model,
        client=client,
    )
    return {
        "n_runs": n_runs,
        "run_dirs": run_dirs,
        "aggregated_rows": len(aggregated),
        "proteins_processed": len(aggregated["Protein.ID"].unique()) if "Protein.ID" in aggregated.columns else 0,
    }


# ---------------------------------------------------------------------------
# Tool: get_run_results
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_run_results(
    *,
    run_dir: str | None = None,
    study_folder: str | None = None,
    max_preview_records: int = 20,
) -> dict[str, Any]:
    """Retrieve prediction artifacts from a completed inference run.

    Provide either ``run_dir`` (a specific run output directory) or
    ``study_folder`` (lists all available runs).

    Args:
        run_dir: Path to a specific output_<timestamp> directory.
        study_folder: Path to the study folder (lists available runs if run_dir not given).
        max_preview_records: Max records per protein to include in preview.
    """
    if run_dir is None and study_folder is None:
        return {"error": "Provide either 'run_dir' or 'study_folder'."}

    # List mode: enumerate available runs
    if run_dir is None:
        assert study_folder is not None
        output_dir = os.path.join(study_folder, "output")
        if not os.path.isdir(output_dir):
            return {"error": f"No output directory in '{study_folder}'."}
        runs = sorted(
            d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("output_")
        )
        return {
            "study_folder": study_folder,
            "available_runs": runs,
        }

    # Detail mode: read predictions from a specific run
    if not os.path.isdir(run_dir):
        return {"error": f"Run directory '{run_dir}' does not exist."}

    predictions: list[dict[str, Any]] = []
    for json_path in sorted(glob.glob(os.path.join(run_dir, "ptm_predictions_*.json"))):
        basename = os.path.basename(json_path)
        # Parse protein ID from filename: ptm_predictions_<ID>_<timestamp>.json
        stem = os.path.splitext(basename)[0]
        parts = stem.replace("ptm_predictions_", "").rsplit("_", 1)
        protein_id = parts[0] if parts else "unknown"

        try:
            with open(json_path, encoding="utf-8") as f:
                records = json.load(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading %s: %s", json_path, exc)
            continue

        predictions.append(
            {
                "protein_id": protein_id,
                "file": json_path,
                "total_records": len(records),
                "preview": records[:max_preview_records],
            }
        )

    return {
        "run_dir": run_dir,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Tool: analyze_run_results
# ---------------------------------------------------------------------------


@mcp.tool()
async def analyze_run_results(
    *,
    run_dir: str | None = None,
    aggregated_csv: str | None = None,
    top_k: int = 25,
    output_dir: str | None = None,
    create_plots: bool = True,
) -> dict[str, Any]:
    """Generate a deterministic analysis report for a PTM run or aggregated CSV."""
    try:
        return analyze_run_results_report(
            run_dir=run_dir,
            aggregated_csv=aggregated_csv,
            top_k=top_k,
            output_dir=output_dir,
            create_plots=create_plots,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
