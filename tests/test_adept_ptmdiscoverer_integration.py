"""
Tests for PTMdiscoverer ADEPT integration.

Tests the MCP tool adapter (ptmdiscoverer_tool.py) and
LangChain wrappers (ptm_langchain_tools.py).

These are unit tests that mock external dependencies
(filesystem, PTMdiscoverer library, MCP server).
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from conftest import write_simple_aggregated_csv

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ptm_tools():
    """Register PTMdiscoverer tools on a mock MCP and return the captured dict."""
    from agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool import (
        register_tools,
    )

    tools: dict[str, object] = {}

    def capture_tool():
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mock_mcp = MagicMock()
    mock_mcp.tool = capture_tool
    register_tools(mock_mcp)
    return tools


@pytest.fixture()
def mock_ctx():
    """Return a MagicMock MCP context with an async ``info`` method."""
    ctx = MagicMock()
    ctx.info = AsyncMock()
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_study_folder(tmp_path):
    """Create a minimal valid study folder structure."""
    study = tmp_path / "study"
    study.mkdir()
    (study / "base_prompt.txt").write_text("You are a PTM expert.")
    metadata = {
        "proteins": [
            {
                "uniprot_id": "Q9R6W2",
                "name": "Protein A",
                "delta_mass_file": "Q9R6W2_delta-masses.csv",
            },
            {"uniprot_id": "P05067", "name": "Protein B"},
        ]
    }
    (study / "protein_metadata.json").write_text(json.dumps(metadata))
    # Create a delta-mass CSV for Q9R6W2
    (study / "Q9R6W2_delta-masses.csv").write_text("Peptide,DeltaMass\nABCDEFK,79.96\nGHIJKLR,-17.03\n")
    return str(study)


# ---------------------------------------------------------------------------
# Tests: ptmdiscoverer_tool.py — MCP tool adapter
# ---------------------------------------------------------------------------


class TestPTMToolRegistration:
    """Test that register_tools properly registers 6 tools on the MCP instance."""

    def test_register_tools_adds_six_tools(self):
        """register_tools should decorate 6 functions onto the mcp instance."""
        from agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool import (
            register_tools,
        )

        mock_mcp = MagicMock()
        # Make mcp.tool() return a decorator that just returns the function
        mock_mcp.tool.return_value = lambda fn: fn

        register_tools(mock_mcp)

        # Should have called mcp.tool() exactly 6 times
        assert mock_mcp.tool.call_count == 6


class TestPTMValidateStudyFolder:
    """Test ptm_validate_study_folder tool."""

    @pytest.mark.asyncio
    async def test_validate_study_folder_valid_folder(self, tmp_path, ptm_tools, mock_ctx):
        """Should return structured validation output for a valid folder."""
        study = _make_study_folder(tmp_path)

        result = await ptm_tools["ptm_validate_study_folder"](study, mock_ctx)

        assert result["valid"] is True
        assert result["protein_count"] == 2
        assert result["checks"]["study_folder_exists"] is True

    @pytest.mark.asyncio
    async def test_validate_study_folder_invalid_folder(self, ptm_tools, mock_ctx):
        """Should return structured validation output for an invalid folder."""
        result = await ptm_tools["ptm_validate_study_folder"]("/nonexistent/path", mock_ctx)
        assert result["valid"] is False
        assert result["errors"]


class TestPTMListProteins:
    """Test ptm_list_proteins tool."""

    @pytest.mark.asyncio
    async def test_list_proteins_valid_folder(self, tmp_path, ptm_tools, mock_ctx):
        """Should return protein list with delta-mass availability."""
        study = _make_study_folder(tmp_path)

        result = await ptm_tools["ptm_list_proteins"](study, mock_ctx)

        assert "proteins" in result
        assert result["count"] == 2
        proteins = result["proteins"]
        # Q9R6W2 has a CSV file
        q9 = next(p for p in proteins if p["uniprot_id"] == "Q9R6W2")
        assert q9["has_delta_masses"] is True
        # P05067 does not
        p05 = next(p for p in proteins if p["uniprot_id"] == "P05067")
        assert p05["has_delta_masses"] is False

    @pytest.mark.asyncio
    async def test_list_proteins_invalid_folder(self, ptm_tools, mock_ctx):
        """Should return error for nonexistent folder."""
        result = await ptm_tools["ptm_list_proteins"]("/nonexistent/path", mock_ctx)
        assert "error" in result


class TestPTMExtractDeltaMasses:
    """Test ptm_extract_delta_masses tool."""

    @pytest.mark.asyncio
    async def test_extract_existing_csv(self, tmp_path, ptm_tools, mock_ctx):
        """Should read existing CSV and return preview."""
        study = _make_study_folder(tmp_path)

        result = await ptm_tools["ptm_extract_delta_masses"](study, "Q9R6W2", mock_ctx)

        assert result["uniprot_id"] == "Q9R6W2"
        assert result["source"] == "existing_csv"
        assert result["rows"] == 2
        assert "preview" in result

    @pytest.mark.asyncio
    async def test_extract_unknown_protein(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error for unknown protein."""
        study = _make_study_folder(tmp_path)

        result = await ptm_tools["ptm_extract_delta_masses"](study, "ZZZZZZ", mock_ctx)
        assert "error" in result
        assert "ZZZZZZ" in result["error"]


class TestPTMRunInference:
    """Test ptm_run_inference tool."""

    @pytest.mark.asyncio
    async def test_run_inference_single(self, tmp_path, ptm_tools, mock_ctx):
        """Should call run_inference and return results."""
        study = _make_study_folder(tmp_path)

        fake_result = {
            "run_dir": "/tmp/output_123",  # noqa: S108 — hardcoded path used only as mock return value in test
            "timestamp": "2026-03-17",
            "proteins_processed": 2,
            "total_ptm_records": 10,
        }

        with (
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool.run_inference",
                return_value=fake_result,
            ),
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool._build_client",
                return_value=MagicMock(),
            ),
        ):
            result = await ptm_tools["ptm_run_inference"](study, mock_ctx, n_runs=1)

        assert result["n_runs"] == 1
        assert result["proteins_processed"] == 2
        assert result["total_ptm_records"] == 10

    @pytest.mark.asyncio
    async def test_run_inference_multi(self, tmp_path, ptm_tools, mock_ctx):
        """Should call run_and_aggregate for n_runs > 1."""
        study = _make_study_folder(tmp_path)

        import pandas as pd

        fake_agg = pd.DataFrame({"Protein.ID": ["Q9R6W2"] * 5})

        with (
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool.run_and_aggregate_multiple_inferences",
                return_value=(["/tmp/run1", "/tmp/run2", "/tmp/run3"], fake_agg),  # noqa: S108 — mock values only
            ),
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool._build_client",
                return_value=MagicMock(),
            ),
        ):
            result = await ptm_tools["ptm_run_inference"](study, mock_ctx, n_runs=3)

        assert result["n_runs"] == 3
        assert len(result["run_dirs"]) == 3
        assert result["aggregated_rows"] == 5


class TestPTMGetRunResults:
    """Test ptm_get_run_results tool."""

    @pytest.mark.asyncio
    async def test_list_runs(self, tmp_path, ptm_tools, mock_ctx):
        """Should list run directories when study_folder is given."""
        study = _make_study_folder(tmp_path)
        output_dir = os.path.join(study, "output")
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "output_2026-03-17-10-00"))
        os.makedirs(os.path.join(output_dir, "output_2026-03-17-11-00"))

        result = await ptm_tools["ptm_get_run_results"](mock_ctx, study_folder=study)
        assert "available_runs" in result
        assert len(result["available_runs"]) == 2

    @pytest.mark.asyncio
    async def test_get_specific_run(self, tmp_path, ptm_tools, mock_ctx):
        """Should return predictions from a specific run dir."""
        run_dir = str(tmp_path / "output_2026")
        os.makedirs(run_dir)
        predictions = [{"PEPTIDE": "ABCDEFK", "PTM.NAME": "Phosphorylation"}]
        with open(os.path.join(run_dir, "ptm_predictions_Q9R6W2_2026.json"), "w") as f:
            json.dump(predictions, f)

        result = await ptm_tools["ptm_get_run_results"](mock_ctx, run_dir=run_dir)
        assert "predictions" in result
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["protein_id"] == "Q9R6W2"

    @pytest.mark.asyncio
    async def test_no_args_returns_error(self, ptm_tools, mock_ctx):
        """Should return error when neither run_dir nor study_folder given."""
        result = await ptm_tools["ptm_get_run_results"](mock_ctx)
        assert "error" in result


class TestPTMAnalyzeRunResults:
    """Test ptm_analyze_run_results tool."""

    @pytest.mark.asyncio
    async def test_analyze_run_results_forwards_to_reporting_core(self, ptm_tools, mock_ctx):
        fake_result = {
            "analysis_mode": "aggregated_csv",
            "output": {"summary_json": "summary.json", "plots": [], "tables": []},
        }

        with patch(
            "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool.analyze_run_results_report",
            return_value=fake_result,
        ) as patched:
            result = await ptm_tools["ptm_analyze_run_results"](
                mock_ctx,
                aggregated_csv="aggregated.csv",
                top_k=7,
                output_dir="reports",
                create_plots=False,
            )

        patched.assert_called_once_with(
            run_dir=None,
            aggregated_csv="aggregated.csv",
            top_k=7,
            output_dir="reports",
            create_plots=False,
        )
        assert result["analysis_mode"] == "aggregated_csv"

    @pytest.mark.asyncio
    async def test_analyze_run_results_executes_real_reporting(self, tmp_path, ptm_tools, mock_ctx):
        aggregated_csv = str(write_simple_aggregated_csv(tmp_path, protein_id="Q9R6W2"))

        result = await ptm_tools["ptm_analyze_run_results"](
            mock_ctx,
            aggregated_csv=aggregated_csv,
            output_dir=str(tmp_path),
            create_plots=False,
        )

        assert result["analysis_mode"] == "aggregated_csv"
        assert result["source"]["input_files"] == [aggregated_csv]
        assert "protein_focus_summary" in result
        assert os.path.isfile(result["output"]["summary_json"])


# ---------------------------------------------------------------------------
# Tests: ptm_langchain_tools.py — LangChain wrappers
# ---------------------------------------------------------------------------


class TestPTMLangChainWrappers:
    """Test that the PTM wrapper factories produce valid MCPToolWrapper instances."""

    def test_list_proteins_wrapper(self):
        """Factory should return a properly configured MCPToolWrapper."""
        from agentic_framework_pkg.scientific_workflow.ptm_langchain_tools import (
            get_ptm_list_proteins_langchain,
        )

        tool = get_ptm_list_proteins_langchain(mcp_session_id="test-session")
        assert tool.name == "PTMListProteins"
        assert tool.actual_tool_name == "ptm_list_proteins"
        assert tool.mcp_session_id == "test-session"
        assert tool.args_schema is not None

    def test_validate_wrapper(self):
        """Factory should return a properly configured MCPToolWrapper."""
        from agentic_framework_pkg.scientific_workflow.ptm_langchain_tools import (
            get_ptm_validate_study_folder_langchain,
        )

        tool = get_ptm_validate_study_folder_langchain(mcp_session_id="test-session")
        assert tool.name == "PTMValidateStudyFolder"
        assert tool.actual_tool_name == "ptm_validate_study_folder"
        assert tool.mcp_session_id == "test-session"
        assert tool.args_schema is not None

    def test_extract_wrapper(self):
        """Factory should return a properly configured MCPToolWrapper."""
        from agentic_framework_pkg.scientific_workflow.ptm_langchain_tools import (
            get_ptm_extract_delta_masses_langchain,
        )

        tool = get_ptm_extract_delta_masses_langchain()
        assert tool.name == "PTMExtractDeltaMasses"
        assert tool.actual_tool_name == "ptm_extract_delta_masses"
        assert tool.args_schema is not None

    def test_run_inference_wrapper(self):
        """Factory should return a properly configured MCPToolWrapper."""
        from agentic_framework_pkg.scientific_workflow.ptm_langchain_tools import (
            get_ptm_run_inference_langchain,
        )

        tool = get_ptm_run_inference_langchain(mcp_session_id="sess-123")
        assert tool.name == "PTMRunInference"
        assert tool.actual_tool_name == "ptm_run_inference"
        assert tool.mcp_session_id == "sess-123"

    def test_get_results_wrapper(self):
        """Factory should return a properly configured MCPToolWrapper."""
        from agentic_framework_pkg.scientific_workflow.ptm_langchain_tools import (
            get_ptm_get_run_results_langchain,
        )

        tool = get_ptm_get_run_results_langchain()
        assert tool.name == "PTMGetRunResults"
        assert tool.actual_tool_name == "ptm_get_run_results"

    def test_analyze_results_wrapper(self):
        """Factory should return a properly configured MCPToolWrapper."""
        from agentic_framework_pkg.scientific_workflow.ptm_langchain_tools import (
            get_ptm_analyze_run_results_langchain,
        )

        tool = get_ptm_analyze_run_results_langchain()
        assert tool.name == "PTMAnalyzeRunResults"
        assert tool.actual_tool_name == "ptm_analyze_run_results"

    def test_all_wrappers_have_study_folder_or_run_dir_field(self):
        """All PTM wrapper schemas should include study_folder or run_dir."""
        from agentic_framework_pkg.scientific_workflow.ptm_langchain_tools import (
            get_ptm_analyze_run_results_langchain,
            get_ptm_extract_delta_masses_langchain,
            get_ptm_get_run_results_langchain,
            get_ptm_list_proteins_langchain,
            get_ptm_run_inference_langchain,
            get_ptm_validate_study_folder_langchain,
        )

        for factory in [
            get_ptm_validate_study_folder_langchain,
            get_ptm_list_proteins_langchain,
            get_ptm_extract_delta_masses_langchain,
            get_ptm_run_inference_langchain,
            get_ptm_get_run_results_langchain,
            get_ptm_analyze_run_results_langchain,
        ]:
            tool = factory()
            schema = tool.args_schema
            assert schema is not None, f"{tool.name} has no args_schema"
            fields = getattr(schema, "model_fields", {})
            has_location = "study_folder" in fields or "run_dir" in fields
            assert has_location, f"{tool.name} schema missing study_folder/run_dir"


class TestScientificWorkflowAgentPTMTools:
    """Test that the ADEPT agent exposes the PTM analysis tool in its tool list."""

    def test_agent_includes_ptm_analyze_results_tool(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True),
            patch(
                "agentic_framework_pkg.scientific_workflow.langchain_agent.create_openai_tools_agent",
                return_value=MagicMock(),
            ),
            patch("agentic_framework_pkg.scientific_workflow.langchain_agent.AgentExecutor", return_value=MagicMock()),
        ):
            from agentic_framework_pkg.scientific_workflow.langchain_agent import ScientificWorkflowAgent

            agent = ScientificWorkflowAgent(mcp_session_id="session-123")

        tool_names = {tool.name for tool in agent.tools}
        assert "PTMAnalyzeRunResults" in tool_names


class TestPTMBuildClient:
    """Test _build_client env var handling."""

    def test_build_client_internal_llm(self):
        """Should prefer INTERNAL_LLM_API_KEY when set."""
        from agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool import (
            _build_client,
        )

        with patch.dict(
            os.environ,
            {
                "INTERNAL_LLM_API_KEY": "test-internal-key",
                "INTERNAL_LLM_BASE_URL": "https://internal.example.com",
                "OPENAI_API_KEY": "test-openai-key",
            },
        ):
            with patch("agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool.openai.OpenAI") as mock_openai:
                _build_client()
                mock_openai.assert_called_once_with(
                    api_key="test-internal-key",
                    base_url="https://internal.example.com",
                    timeout=300.0,
                )

    def test_build_client_openai_fallback(self):
        """Should fall back to OPENAI_API_KEY when internal vars not set."""
        from agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool import (
            _build_client,
        )

        # Use clear=True to remove ALL env vars, then set only OPENAI_API_KEY
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=True):
            with patch("agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool.openai.OpenAI") as mock_openai:
                _build_client()
                mock_openai.assert_called_once_with(
                    api_key="test-openai-key",
                    base_url="https://api.openai.com/v1",
                    timeout=300.0,
                )

    def test_build_client_no_keys_raises(self):
        """Should raise ValueError when no API keys are set."""
        from agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool import (
            _build_client,
        )

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                _build_client()


# ---------------------------------------------------------------------------
# Tests: _resolve_study_folder — path normalisation
# ---------------------------------------------------------------------------


class TestResolveStudyFolder:
    """Test the LLM path normalisation helper."""

    def test_returns_candidate_when_dir_exists(self, tmp_path):
        """Should return the resolved candidate path when it exists under the base."""
        from agentic_framework_pkg.mcp_server.tools import ptmdiscoverer_tool

        study_dir = tmp_path / "my_study"
        study_dir.mkdir()

        original_base = ptmdiscoverer_tool._PTM_STUDIES_BASE
        try:
            ptmdiscoverer_tool._PTM_STUDIES_BASE = str(tmp_path)
            result = ptmdiscoverer_tool._resolve_study_folder("my_study")
        finally:
            ptmdiscoverer_tool._PTM_STUDIES_BASE = original_base

        assert result == str(study_dir)

    def test_falls_back_to_original_when_not_found(self):
        """Should return the original path when candidate dir does not exist."""
        from agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool import (
            _resolve_study_folder,
        )

        result = _resolve_study_folder("/nonexistent/path/to/study")
        # Fallback: returned as-is (after prefix stripping)
        assert "nonexistent" in result or result == "/nonexistent/path/to/study" or result.endswith("study")


# ---------------------------------------------------------------------------
# Tests: ptm_extract_delta_masses — additional edge cases
# ---------------------------------------------------------------------------


class TestPTMExtractDeltaMassesEdgeCases:
    """Additional edge-case tests for ptm_extract_delta_masses."""

    @pytest.mark.asyncio
    async def test_extract_invalid_folder(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when study folder does not exist."""
        result = await ptm_tools["ptm_extract_delta_masses"](str(tmp_path / "nope"), "Q9R6W2", mock_ctx)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_no_csv_no_labels(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when protein has no CSV and study has no dataset_labels."""
        study = _make_study_folder(tmp_path)
        # P05067 has no delta_mass_file; the study has no dataset_labels.csv
        result = await ptm_tools["ptm_extract_delta_masses"](study, "P05067", mock_ctx)
        assert "error" in result
        assert "dataset_labels.csv" in result["error"]

    @pytest.mark.asyncio
    async def test_extract_csv_read_failure(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when the existing delta-mass CSV cannot be read."""
        import pandas as pd

        study = _make_study_folder(tmp_path)
        with patch.object(pd, "read_csv", side_effect=OSError("disk error")):
            result = await ptm_tools["ptm_extract_delta_masses"](study, "Q9R6W2", mock_ctx)
        assert "error" in result
        assert "disk error" in result["error"]


# ---------------------------------------------------------------------------
# Tests: ptm_run_inference — exception paths
# ---------------------------------------------------------------------------


class TestPTMRunInferenceExceptions:
    """Test error-handling paths in ptm_run_inference."""

    @pytest.mark.asyncio
    async def test_build_client_failure(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when _build_client raises."""
        study = _make_study_folder(tmp_path)
        with patch(
            "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool._build_client",
            side_effect=ValueError("no API key configured"),
        ):
            result = await ptm_tools["ptm_run_inference"](study, mock_ctx, n_runs=1)
        assert "error" in result
        assert "no API key" in result["error"]

    @pytest.mark.asyncio
    async def test_run_inference_raises(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when the underlying run_inference call raises."""
        study = _make_study_folder(tmp_path)
        with (
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool._build_client",
                return_value=MagicMock(),
            ),
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool.asyncio.to_thread",
                side_effect=RuntimeError("LLM timeout"),
            ),
        ):
            result = await ptm_tools["ptm_run_inference"](study, mock_ctx, n_runs=1)
        assert "error" in result
        assert "LLM timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_multi_run_raises(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when run_and_aggregate raises for n_runs > 1."""
        study = _make_study_folder(tmp_path)
        with (
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool._build_client",
                return_value=MagicMock(),
            ),
            patch(
                "agentic_framework_pkg.mcp_server.tools.ptmdiscoverer_tool.asyncio.to_thread",
                side_effect=RuntimeError("aggregation error"),
            ),
        ):
            result = await ptm_tools["ptm_run_inference"](study, mock_ctx, n_runs=3)
        assert "error" in result
        assert "aggregation error" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_folder_returns_error(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when study folder is invalid."""
        result = await ptm_tools["ptm_run_inference"](str(tmp_path / "nope"), mock_ctx, n_runs=1)
        assert "error" in result


# ---------------------------------------------------------------------------
# Tests: ptm_get_run_results — additional edge cases
# ---------------------------------------------------------------------------


class TestPTMGetRunResultsEdgeCases:
    """Additional edge-case tests for ptm_get_run_results."""

    @pytest.mark.asyncio
    async def test_run_dir_not_found(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when run_dir does not exist."""
        result = await ptm_tools["ptm_get_run_results"](mock_ctx, run_dir=str(tmp_path / "output_nonexistent"))
        assert "error" in result
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_no_output_dir(self, tmp_path, ptm_tools, mock_ctx):
        """Should return error when study_folder has no output/ subdirectory."""
        study = _make_study_folder(tmp_path)
        # No output/ dir created
        result = await ptm_tools["ptm_get_run_results"](mock_ctx, study_folder=study)
        assert "error" in result
        assert "No output directory" in result["error"]

    @pytest.mark.asyncio
    async def test_corrupt_json_skipped(self, tmp_path, ptm_tools, mock_ctx):
        """Should skip corrupt JSON files and return empty predictions list."""
        run_dir = str(tmp_path / "output_corrupt")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "ptm_predictions_Q9R6W2_2026.json"), "w") as f:
            f.write("{invalid json}")
        result = await ptm_tools["ptm_get_run_results"](mock_ctx, run_dir=run_dir)
        assert "predictions" in result
        assert result["predictions"] == []
