"""PTMdiscoverer tools registered onto the ADEPT MCP server.

These tools wrap PTMdiscoverer's public library API so they can be
called via the shared ADEPT FastMCP instance (HTTP transport).
PTMdiscoverer remains a standalone library — no library code is
duplicated here.
"""

import asyncio
import glob
import json
import os
import time
from typing import Any

import openai
import pandas as pd
from fastmcp import Context, FastMCP

from ptmdiscoverer.inference import (
    assess_study_folder,
    run_and_aggregate_multiple_inferences,
    run_inference,
    validate_study_folder,
)
from ptmdiscoverer.reporting import analyze_run_results as analyze_run_results_report
from ptmdiscoverer.utils import extract_delta_masses, load_protein_metadata

from ...logger_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_client() -> openai.OpenAI:
    """Build an OpenAI-compatible client from ADEPT env vars.

    Prefers INTERNAL_LLM_* variables (primary OpenAI-compatible endpoint),
    falls back to OPENAI_API_KEY.
    Timeout is controlled by PTM_LLM_CALL_TIMEOUT_SECONDS (default 300 s).
    """
    api_key = os.getenv("INTERNAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set INTERNAL_LLM_API_KEY or OPENAI_API_KEY.")
    base_url = os.getenv("INTERNAL_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_timeout = float(os.getenv("PTM_LLM_CALL_TIMEOUT_SECONDS", "300"))
    logger.debug("Creating PTM client with base_url=%s timeout=%.0fs", base_url, llm_timeout)
    return openai.OpenAI(api_key=api_key, base_url=base_url, timeout=llm_timeout)


def _default_ptm_model() -> str:
    """Return the default LLM model for PTM inference.

    Priority: PTM_INFERENCE_MODEL > INTERNAL_LLM_MODEL > 'gpt-4o'.
    """
    return os.getenv("PTM_INFERENCE_MODEL") or os.getenv("INTERNAL_LLM_MODEL") or "gpt-4o"


def _resolve_protein(proteins: list[dict[str, Any]], uniprot_id: str) -> dict[str, Any] | None:
    """Find a protein entry by uniprot_id (case-insensitive)."""
    uid = uniprot_id.upper()
    for p in proteins:
        if str(p.get("uniprot_id", "")).upper() == uid:
            return p
    return None


# Base directory for PTM study folders inside the container.
# Matches the volume mount: ./ptm_studies:/app/ptm_studies
_PTM_STUDIES_BASE = os.getenv("PTM_STUDIES_BASE_DIR", "/app/ptm_studies")


def _resolve_study_folder(study_folder: str) -> str:
    """Normalize the study_folder path supplied by the LLM.

    The LLM sometimes passes an absolute path that doesn't exist
    (e.g. '/ptm_studies/s_elongatus') or a wrong relative path.  We
    strip any leading '/ptm_studies', '/app', or 'app' prefix and
    re-anchor to _PTM_STUDIES_BASE so simple names like
    's_elongatus' or 'ptm_studies/s_elongatus' all resolve correctly.
    """
    import re as _re

    # Strip common wrong prefixes the LLM tends to emit
    normalized = _re.sub(r"^(/app)?/?(ptm_studies/)?", "", study_folder.lstrip("/"))
    # normalized is now something like 's_elongatus'
    # If it already starts with the base, don't double-add
    candidate = os.path.join(_PTM_STUDIES_BASE, normalized)
    if os.path.isdir(candidate):
        return candidate
    # Fallback: return as-is for the error message to reflect the original value
    return study_folder


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_tools(mcp: FastMCP) -> None:
    """Register PTMdiscoverer tools on the given FastMCP instance."""

    @mcp.tool()
    async def ptm_validate_study_folder(
        study_folder: str,
        ctx: Context,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate a PTM study folder and return structured feedback.

        Args:
            study_folder: Path to the study directory.
            ctx: FastMCP context.
            mcp_session_id: MCP session ID (optional).
        """
        study_folder = _resolve_study_folder(study_folder)
        return assess_study_folder(study_folder)

    @mcp.tool()
    async def ptm_list_proteins(
        study_folder: str,
        ctx: Context,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """List proteins in a PTM study folder with delta-mass availability.

        Args:
            study_folder: Path to the study directory containing protein_metadata.json.
            ctx: FastMCP context.
            mcp_session_id: MCP session ID (optional).
        """
        study_folder = _resolve_study_folder(study_folder)
        try:
            validate_study_folder(study_folder)
        except FileNotFoundError as exc:
            return {"error": str(exc)}

        metadata_path = os.path.join(study_folder, "protein_metadata.json")
        proteins = load_protein_metadata(metadata_path)

        result = []
        for p in proteins:
            uid = p.get("uniprot_id", "")
            dmf = p.get("delta_mass_file", "")
            if dmf and not os.path.isabs(dmf):
                dmf = os.path.join(study_folder, dmf)
            has_dm = bool(dmf and os.path.isfile(dmf))
            result.append(
                {
                    "uniprot_id": uid,
                    "name": p.get("name", ""),
                    "has_delta_masses": has_dm,
                }
            )
        return {"proteins": result, "count": len(result)}

    @mcp.tool()
    async def ptm_extract_delta_masses(
        study_folder: str,
        uniprot_id: str,
        ctx: Context,
        *,
        max_preview_rows: int = 10,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract or read delta-mass data for a specific protein.

        If a pre-computed CSV exists it is read directly. Otherwise,
        extraction from MSFragger PSM data is attempted.

        Args:
            study_folder: Path to the study directory.
            uniprot_id: UniProt accession of the target protein.
            ctx: FastMCP context.
            max_preview_rows: Number of rows to include in the preview.
            mcp_session_id: MCP session ID (optional).
        """
        study_folder = _resolve_study_folder(study_folder)
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
            except Exception as exc:
                return {"error": f"Failed reading {dmf}: {exc}"}

        # Attempt extraction from PSM data
        labels_path = os.path.join(study_folder, "dataset_labels.csv")
        if not os.path.isfile(labels_path):
            return {"error": f"No delta-mass CSV and no dataset_labels.csv in '{study_folder}'."}

        try:
            msrun_table = pd.read_csv(labels_path)
        except Exception as exc:
            return {"error": f"Failed reading dataset_labels.csv: {exc}"}

        extracted = extract_delta_masses(uniprot_id, msrun_table, study_folder)
        if extracted.empty:
            return {
                "uniprot_id": uniprot_id,
                "source": "extraction",
                "rows": 0,
                "message": "No qualifying delta masses found in PSM data.",
            }

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

    @mcp.tool()
    async def ptm_run_inference(
        study_folder: str,
        ctx: Context,
        *,
        n_runs: int = 1,
        model: str = "",
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run LLM-based PTM inference over all proteins in a study folder.

        With n_runs=1 runs a single inference pass. With n_runs>1 runs
        multiple passes and aggregates the results.

        Args:
            study_folder: Path to the study directory.
            ctx: FastMCP context.
            n_runs: Number of inference runs (default 1).
            model: LLM model identifier (defaults to INTERNAL_LLM_MODEL env var).
            mcp_session_id: MCP session ID (optional).
        """
        # Resolve model: explicit arg > env-var chain > gpt-4o
        resolved_model = model if model else _default_ptm_model()
        study_folder = _resolve_study_folder(study_folder)
        try:
            validate_study_folder(study_folder)
        except FileNotFoundError as exc:
            return {"error": str(exc)}

        try:
            client = _build_client()
        except Exception as exc:
            return {"error": f"Failed to create API client: {exc}"}

        await ctx.info(f"Running PTM inference ({n_runs} run(s)) on {study_folder} with model {resolved_model}")

        if n_runs <= 1:
            try:
                _t0 = time.monotonic()
                result = await asyncio.to_thread(
                    run_inference,
                    base_data_folder=study_folder,
                    model=resolved_model,
                    client=client,
                )
                logger.debug("PTM single-run inference completed in %.1fs", time.monotonic() - _t0)
            except Exception as exc:
                logger.error("PTM single-run inference failed after %.1fs: %s", time.monotonic() - _t0, exc)
                return {"error": f"Inference failed: {exc}"}
            return {"n_runs": 1, **result}

        try:
            _t0 = time.monotonic()
            run_dirs, aggregated = await asyncio.to_thread(
                run_and_aggregate_multiple_inferences,
                n_runs,
                base_data_folder=study_folder,
                model=resolved_model,
                client=client,
            )
            logger.debug("PTM multi-run inference completed in %.1fs", time.monotonic() - _t0)
        except Exception as exc:
            logger.error("PTM multi-run inference failed after %.1fs: %s", time.monotonic() - _t0, exc)
            return {"error": f"Multi-run inference failed: {exc}"}
        return {
            "n_runs": n_runs,
            "run_dirs": run_dirs,
            "aggregated_rows": len(aggregated),
        }

    @mcp.tool()
    async def ptm_get_run_results(
        ctx: Context,
        *,
        run_dir: str | None = None,
        study_folder: str | None = None,
        max_preview_records: int = 20,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve prediction artifacts from a completed inference run.

        Provide either run_dir (a specific run output directory) or
        study_folder (lists all available runs).

        Args:
            ctx: FastMCP context.
            run_dir: Path to a specific output_<timestamp> directory.
            study_folder: Path to the study folder (lists available runs).
            max_preview_records: Max records per protein to include in preview.
            mcp_session_id: MCP session ID (optional).
        """
        if run_dir is None and study_folder is None:
            return {"error": "Provide either 'run_dir' or 'study_folder'."}

        if study_folder is not None:
            study_folder = _resolve_study_folder(study_folder)

        # List mode
        if run_dir is None:
            if study_folder is None:
                return {"error": "Provide either 'run_dir' or 'study_folder'."}

            output_dir = os.path.join(study_folder, "output")
            if not os.path.isdir(output_dir):
                return {"error": f"No output directory in '{study_folder}'."}
            runs = sorted(
                d
                for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("output_")
            )
            return {"study_folder": study_folder, "available_runs": runs}

        # Detail mode
        if not os.path.isdir(run_dir):
            return {"error": f"Run directory '{run_dir}' does not exist."}

        predictions: list[dict[str, Any]] = []
        for json_path in sorted(glob.glob(os.path.join(run_dir, "ptm_predictions_*.json"))):
            basename = os.path.basename(json_path)
            stem = os.path.splitext(basename)[0]
            parts = stem.replace("ptm_predictions_", "").rsplit("_", 1)
            protein_id = parts[0] if parts else "unknown"

            try:
                with open(json_path, encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as exc:
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

        return {"run_dir": run_dir, "predictions": predictions}

    @mcp.tool()
    async def ptm_analyze_run_results(
        ctx: Context,
        *,
        run_dir: str | None = None,
        aggregated_csv: str | None = None,
        top_k: int = 25,
        output_dir: str | None = None,
        create_plots: bool = True,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate a deterministic analysis report for a PTM run or aggregated CSV.

        Args:
            ctx: FastMCP context.
            run_dir: Path to a specific output_<timestamp> directory.
            aggregated_csv: Path to a specific aggregated predictions CSV.
            top_k: Number of top records to include in summary tables and plots.
            output_dir: Optional parent directory for the generated analysis_<timestamp> directory.
            create_plots: Whether PNG plots should be rendered.
            mcp_session_id: MCP session ID (optional).
        """
        await ctx.info("Generating deterministic PTM run analysis report")
        try:
            return await asyncio.to_thread(
                analyze_run_results_report,
                run_dir=run_dir,
                aggregated_csv=aggregated_csv,
                top_k=top_k,
                output_dir=output_dir,
                create_plots=create_plots,
            )
        except Exception as exc:
            logger.error("PTM run analysis failed: %s", exc)
            return {"error": str(exc)}

    logger.info("PTMdiscoverer MCP tools registered.")
