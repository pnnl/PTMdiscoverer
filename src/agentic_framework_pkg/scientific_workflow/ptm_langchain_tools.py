"""LangChain tool wrappers for PTMdiscoverer MCP tools.

Each factory function returns an :class:`MCPToolWrapper` configured
to call the corresponding ``ptm_*`` tool on the ADEPT MCP server.
Import these factories in :mod:`langchain_agent` and append to
``self.tools``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .mcp_langchain_tools import MCP_SERVER_URL_FOR_LANGCHAIN, MCPToolWrapper

# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------


class PTMListProteinsInput(BaseModel):
    study_folder: str = Field(description="Path to the study directory containing protein_metadata.json.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class PTMValidateStudyFolderInput(BaseModel):
    study_folder: str = Field(description="Path to the study directory.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class PTMExtractDeltaMassesInput(BaseModel):
    study_folder: str = Field(description="Path to the study directory.")
    uniprot_id: str = Field(description="UniProt accession of the target protein (e.g. Q9R6W2).")
    max_preview_rows: int = Field(10, description="Number of rows to include in the preview.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class PTMRunInferenceInput(BaseModel):
    study_folder: str = Field(description="Path to the study directory.")
    n_runs: int = Field(1, description="Number of inference runs (1 = single, >1 = aggregate).")
    model: str = Field(
        "",
        description=("LLM model identifier.  Leave empty to use the server's default (INTERNAL_LLM_MODEL env var)."),
    )
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class PTMGetRunResultsInput(BaseModel):
    run_dir: str | None = Field(None, description="Path to a specific output_<timestamp> directory.")
    study_folder: str | None = Field(
        None,
        description="Path to the study folder (lists all available runs).",
    )
    max_preview_records: int = Field(20, description="Max prediction records per protein to preview.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class PTMAnalyzeRunResultsInput(BaseModel):
    run_dir: str | None = Field(None, description="Path to a specific output_<timestamp> directory.")
    aggregated_csv: str | None = Field(None, description="Path to an aggregated predictions CSV.")
    top_k: int = Field(25, description="Number of top records to include in summary tables and plots.")
    output_dir: str | None = Field(
        None,
        description="Optional parent directory for the generated analysis_<timestamp> directory.",
    )
    create_plots: bool = Field(True, description="Whether PNG plots should be rendered.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def get_ptm_list_proteins_langchain(
    mcp_session_id: str | None = None,
) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="PTMListProteins",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="ptm_list_proteins",
        description=(
            "List all proteins in a PTM study folder (protein_metadata.json) "
            "and report whether each protein has pre-computed delta-mass data."
        ),
        args_schema=PTMListProteinsInput,
        mcp_session_id=mcp_session_id,
    )


def get_ptm_validate_study_folder_langchain(
    mcp_session_id: str | None = None,
) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="PTMValidateStudyFolder",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="ptm_validate_study_folder",
        description=(
            "Validate a PTM study folder before inference. Returns structured feedback "
            "about required files, optional files, and setup problems."
        ),
        args_schema=PTMValidateStudyFolderInput,
        mcp_session_id=mcp_session_id,
    )


def get_ptm_extract_delta_masses_langchain(
    mcp_session_id: str | None = None,
) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="PTMExtractDeltaMasses",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="ptm_extract_delta_masses",
        description=(
            "Extract or read delta-mass data for a given protein in a study folder. "
            "Returns a preview of observed delta masses that feed into PTM inference."
        ),
        args_schema=PTMExtractDeltaMassesInput,
        mcp_session_id=mcp_session_id,
    )


def get_ptm_run_inference_langchain(
    mcp_session_id: str | None = None,
) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="PTMRunInference",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="ptm_run_inference",
        description=(
            "Run LLM-based Post-Translational Modification inference across all "
            "proteins in a study folder. With n_runs > 1, runs multiple passes "
            "and aggregates predictions."
        ),
        args_schema=PTMRunInferenceInput,
        mcp_session_id=mcp_session_id,
    )


def get_ptm_get_run_results_langchain(
    mcp_session_id: str | None = None,
) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="PTMGetRunResults",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="ptm_get_run_results",
        description=(
            "Retrieve prediction artifacts from a completed PTM inference run. "
            "Provide run_dir for a specific run, or study_folder to list all "
            "available runs."
        ),
        args_schema=PTMGetRunResultsInput,
        mcp_session_id=mcp_session_id,
    )


def get_ptm_analyze_run_results_langchain(
    mcp_session_id: str | None = None,
) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="PTMAnalyzeRunResults",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="ptm_analyze_run_results",
        description=(
            "Generate a deterministic analysis report for PTMdiscoverer output. "
            "Provide either a specific run_dir or an aggregated_csv path. "
            "Returns timestamped JSON, Markdown, CSV, and optional PNG artifacts."
        ),
        args_schema=PTMAnalyzeRunResultsInput,
        mcp_session_id=mcp_session_id,
    )
