"""
Unit tests for the Biopython-based BLAST tool (blastq_tool.py).

All NCBI network calls are mocked so tests run offline and fast.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock heavy transitive deps (chromadb) before importing the tool module.
# state_manager is imported by blastq_tool but not exercised in our tests
# (we mock the session helpers directly).
_state_mgr_mock = MagicMock()
_state_mgr_mock.create_session_if_not_exists = AsyncMock()
sys.modules.setdefault("chromadb", MagicMock())
sys.modules.setdefault("agentic_framework_pkg.mcp_server.state_manager", _state_mgr_mock)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_ncbi_email():
    """Ensure NCBI_EMAIL is set for all tests (individual tests can override)."""
    with patch.dict(os.environ, {"NCBI_EMAIL": "test@example.org"}):
        yield


@pytest.fixture()
def blast_tools():
    """Register BLAST tools on a mock MCP and return the captured dict."""
    from agentic_framework_pkg.mcp_server.tools.blastq_tool import register_tools

    tools = {}

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
    """MCP context mock with async info/error helpers."""
    ctx = MagicMock()
    ctx.info = AsyncMock()
    ctx.error = AsyncMock()
    ctx.session_id = "test-blast-session"
    ctx.request_id = "req-001"
    return ctx


def _make_blast_record(hits):
    """Build a minimal Biopython-style BLAST record from a list of hit dicts.

    Each hit dict: {"accession": str, "title": str, "evalue": float,
                    "identity": int, "align_len": int,
                    "q_start": int, "q_end": int, "s_start": int, "s_end": int}
    """
    alignments = []
    for h in hits:
        hsp = SimpleNamespace(
            bits=42.0,
            score=100,
            expect=h.get("evalue", 1e-10),
            identities=h.get("identity", 50),
            align_length=h.get("align_len", 100),
            query_start=h.get("q_start", 1),
            query_end=h.get("q_end", 50),
            sbjct_start=h.get("s_start", 1),
            sbjct_end=h.get("s_end", 50),
        )
        alignment = SimpleNamespace(
            hit_id=h["accession"],
            hit_def=h["title"],
            hsps=[hsp],
        )
        alignments.append(alignment)
    return SimpleNamespace(alignments=alignments)


# A short *S. elongatus* CikA sensory histidine kinase fragment (Q9R6W2)
SAMPLE_SEQUENCE = "MSDKIIHLTD"


# ---------------------------------------------------------------------------
# Tests: registration
# ---------------------------------------------------------------------------


class TestBlastToolRegistration:
    """Verify register_tools registers both blastp and blastn."""

    def test_registers_two_tools(self):
        from agentic_framework_pkg.mcp_server.tools.blastq_tool import register_tools

        mock_mcp = MagicMock()
        mock_mcp.tool.return_value = lambda fn: fn
        register_tools(mock_mcp)
        assert mock_mcp.tool.call_count == 2

    def test_ncbi_email_from_env(self):
        """NCBI email should be configurable via env var."""
        from Bio.Blast import NCBIWWW

        with patch.dict(os.environ, {"NCBI_EMAIL": "test@example.org"}):
            from agentic_framework_pkg.mcp_server.tools.blastq_tool import (
                register_tools,
            )

            mock_mcp = MagicMock()
            mock_mcp.tool.return_value = lambda fn: fn
            register_tools(mock_mcp)
            assert NCBIWWW.email == "test@example.org"


# ---------------------------------------------------------------------------
# Tests: BLASTp
# ---------------------------------------------------------------------------


class TestBlastpSearch:
    """Test perform_blastp_search_biopython tool."""

    @pytest.mark.asyncio
    async def test_blastp_missing_email_returns_error(self, blast_tools, mock_ctx):
        """Should return a clear error when NCBI_EMAIL is not set."""
        with patch.dict(os.environ, {"NCBI_EMAIL": ""}, clear=False):
            # Remove the key entirely so os.environ.get("NCBI_EMAIL") returns None
            os.environ.pop("NCBI_EMAIL", None)
            result = await blast_tools["perform_blastp_search_biopython"](SAMPLE_SEQUENCE, mock_ctx)

        assert "error" in result
        assert "NCBI_EMAIL" in result["error"]

    @pytest.mark.asyncio
    async def test_blastp_returns_hits(self, blast_tools, mock_ctx):
        """Should return parsed hits with accession, title, and HSP summary."""
        fake_record = _make_blast_record(
            [
                {
                    "accession": "sp|Q9R6W2|CIKA_SYNEL",
                    "title": "Circadian input kinase CikA",
                    "evalue": 1e-42,
                    "identity": 95,
                    "align_len": 100,
                    "q_start": 1,
                    "q_end": 100,
                    "s_start": 1,
                    "s_end": 100,
                },
                {
                    "accession": "sp|P12345|FAKE_ECOLI",
                    "title": "Hypothetical kinase",
                    "evalue": 1e-5,
                    "identity": 40,
                    "align_len": 80,
                    "q_start": 5,
                    "q_end": 85,
                    "s_start": 10,
                    "s_end": 90,
                },
            ]
        )

        mock_handle = MagicMock()
        mock_handle.close = MagicMock()

        with (
            patch(
                "agentic_framework_pkg.mcp_server.tools.blastq_tool.NCBIWWW.qblast",
                return_value=mock_handle,
            ),
            patch(
                "agentic_framework_pkg.mcp_server.tools.blastq_tool.NCBIXML.read",
                return_value=fake_record,
            ),
        ):
            result = await blast_tools["perform_blastp_search_biopython"](
                SAMPLE_SEQUENCE, mock_ctx, database="swissprot", hitlist_size=5
            )

        assert "results" in result
        assert len(result["results"]) == 2
        assert result["database"] == "swissprot"

        first = result["results"][0]
        assert first["accession"] == "sp|Q9R6W2|CIKA_SYNEL"
        assert "CikA" in first["title"]
        assert first["hsp_summary"]["evalue"] == 1e-42
        assert first["hsp_summary"]["identity_percent"] == 95.0

    @pytest.mark.asyncio
    async def test_blastp_no_hits(self, blast_tools, mock_ctx):
        """Should return empty results list and a message when no hits."""
        empty_record = SimpleNamespace(alignments=[])
        mock_handle = MagicMock()
        mock_handle.close = MagicMock()

        with (
            patch(
                "agentic_framework_pkg.mcp_server.tools.blastq_tool.NCBIWWW.qblast",
                return_value=mock_handle,
            ),
            patch(
                "agentic_framework_pkg.mcp_server.tools.blastq_tool.NCBIXML.read",
                return_value=empty_record,
            ),
        ):
            result = await blast_tools["perform_blastp_search_biopython"](SAMPLE_SEQUENCE, mock_ctx)

        assert result["results"] == []
        assert "no hits" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_blastp_error_handling(self, blast_tools, mock_ctx):
        """Should return an error dict when qblast raises."""
        with patch(
            "agentic_framework_pkg.mcp_server.tools.blastq_tool.NCBIWWW.qblast",
            side_effect=RuntimeError("NCBI connection refused"),
        ):
            result = await blast_tools["perform_blastp_search_biopython"](SAMPLE_SEQUENCE, mock_ctx)

        assert "error" in result
        assert "NCBI connection refused" in result["details"]


# ---------------------------------------------------------------------------
# Tests: BLASTn (light coverage — same code path as BLASTp)
# ---------------------------------------------------------------------------


class TestBlastnSearch:
    """Minimal test for perform_blastn_search_biopython."""

    @pytest.mark.asyncio
    async def test_blastn_returns_hits(self, blast_tools, mock_ctx):
        """Should work the same way as blastp but default to 'nt' database."""
        fake_record = _make_blast_record(
            [
                {
                    "accession": "NM_001101.5",
                    "title": "Homo sapiens actin beta (ACTB), mRNA",
                    "evalue": 0.0,
                    "identity": 200,
                    "align_len": 200,
                    "q_start": 1,
                    "q_end": 200,
                    "s_start": 1,
                    "s_end": 200,
                },
            ]
        )
        mock_handle = MagicMock()
        mock_handle.close = MagicMock()

        with (
            patch(
                "agentic_framework_pkg.mcp_server.tools.blastq_tool.NCBIWWW.qblast",
                return_value=mock_handle,
            ) as mock_qblast,
            patch(
                "agentic_framework_pkg.mcp_server.tools.blastq_tool.NCBIXML.read",
                return_value=fake_record,
            ),
        ):
            result = await blast_tools["perform_blastn_search_biopython"]("ATGCGATCGATCG", mock_ctx)
            # Verify default database is "nt"
            call_args = mock_qblast.call_args
            assert call_args[0][1] == "nt"  # second positional arg = database

        assert len(result["results"]) == 1
        assert result["results"][0]["accession"] == "NM_001101.5"
