import asyncio
import os
import uuid
from typing import Any, cast

# Import Biopython modules for BLAST
from Bio.Blast import NCBIWWW, NCBIXML
from fastmcp import Context, FastMCP

from ...logger_config import get_logger
from ..state_manager import create_session_if_not_exists

# Use the centralized logger
logger = get_logger(__name__)


# Helper for session ID - consider making this a common utility
async def ensure_session_initialized_blastq(ctx: Context, override_session_id: str | None = None) -> str:
    """Helper to ensure the session exists in the database for blastq_tool."""
    session_id = override_session_id or get_stable_session_id_blastq(ctx)
    client_id = ctx.client_id if hasattr(ctx, "client_id") else None
    await create_session_if_not_exists(session_id, client_id)
    return session_id


def get_stable_session_id_blastq(ctx: Context) -> str:
    """Gets a stable session ID for blastq_tool."""
    if hasattr(ctx, "session_id") and ctx.session_id:
        return ctx.session_id
    if ctx.request_id:
        return ctx.request_id
    logger.warning("BLASTQ Tool: Could not determine a stable session ID. Generating a new one.")
    return str(uuid.uuid4())


def register_tools(mcp: FastMCP):
    # Set NCBI contact info as recommended by NCBI usage policies.
    # NCBI_EMAIL is required — BLAST tools will return an error if unset.
    ncbi_email = os.environ.get("NCBI_EMAIL", "")
    if ncbi_email:
        cast(Any, NCBIWWW).email = ncbi_email
    else:
        logger.warning("NCBI_EMAIL env var not set — BLAST tools will refuse requests until configured.")
    cast(Any, NCBIWWW).tool = os.environ.get("NCBI_APP_NAME", "PTMdiscoverer_ADEPT")

    @mcp.tool()
    async def perform_blastp_search_biopython(
        sequence: str,
        ctx: Context,
        database: str = "nr",
        expect: float = 10.0,
        hitlist_size: int = 10,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Performs a protein sequence homology search (BLASTP) against a specified database using Biopython's NCBIWWW.qblast.

        Args:
            sequence (str): The protein sequence to search with (FASTA format or raw sequence).
            database (str): The BLAST database to search against (e.g., 'nr', 'swissprot', 'pdb'). Defaults to 'nr'.
            expect (float): The expectation value (E-value) threshold. Defaults to 10.0.
            hitlist_size (int): The maximum number of hits to return. Defaults to 10.
            ctx (Context): The FastMCP context object.
            mcp_session_id (str, optional): The MCP session ID.

        Returns:
            Dict[str, Any]: A dictionary containing the search results or an error message.
        """
        session_id = await ensure_session_initialized_blastq(ctx, override_session_id=mcp_session_id)

        if not os.environ.get("NCBI_EMAIL"):
            msg = (
                "BLAST is unavailable: the NCBI_EMAIL environment variable is not set. "
                "NCBI requires a contact email to use their BLAST service. "
                "Please ask your administrator to set NCBI_EMAIL in the .env file."
            )
            logger.warning(msg)
            return {"error": msg}

        await ctx.info(
            f"Session {session_id}: Submitting BLASTP (Biopython) search for sequence (first 20 chars: {sequence[:20]}...) against {database}"
        )

        try:
            result_handle = await asyncio.to_thread(
                NCBIWWW.qblast,
                "blastp",
                database,
                sequence,
                expect=expect,
                hitlist_size=hitlist_size,
                format_type="XML",
            )
            logger.info(f"BLASTP (Biopython) qblast call completed for session {session_id}.")

            blast_record = await asyncio.to_thread(NCBIXML.read, result_handle)
            result_handle.close()
            logger.info(f"BLASTP (Biopython) results parsed for session {session_id}.")

            hits = blast_record.alignments
            if not hits:
                await ctx.info(f"BLASTP (Biopython) search returned no hits for session {session_id}.")
                return {
                    "query": sequence[:20] + "...",
                    "database": database,
                    "results": [],
                    "message": "BLAST search completed, but found no hits.",
                }

            truncated_hits = []
            for hit_alignment in hits:  # Renamed 'hit' to 'hit_alignment' to avoid conflict with HSP's hit attribute
                accession = (
                    hit_alignment.hit_id
                )  # or hit_alignment.accession depending on Biopython version/XML structure
                title = hit_alignment.hit_def
                hsp_summary = None
                if hit_alignment.hsps:
                    best_hsp = hit_alignment.hsps[0]
                    hsp_summary = {
                        "bit_score": best_hsp.bits,
                        "score": best_hsp.score,
                        "evalue": best_hsp.expect,
                        "identity_percent": (best_hsp.identities / best_hsp.align_length) * 100
                        if best_hsp.align_length > 0
                        else 0,
                        "query_from": best_hsp.query_start,
                        "query_to": best_hsp.query_end,
                        "hit_from": best_hsp.sbjct_start,
                        "hit_to": best_hsp.sbjct_end,
                        "align_len": best_hsp.align_length,
                    }
                truncated_hits.append({"accession": accession, "title": title, "hsp_summary": hsp_summary})

            await ctx.info(
                f"Successfully retrieved {len(truncated_hits)} BLASTP (Biopython) hits for session {session_id}."
            )
            return {
                "query": sequence[:20] + "...",
                "database": database,
                "results": truncated_hits,
            }
        except Exception as e:
            error_message = f"An unexpected error occurred during BLASTP (Biopython) search: {e}"
            logger.error(error_message, exc_info=True)
            await ctx.error("BLASTP (Biopython) search failed.")
            return {
                "error": "An unexpected error occurred during BLASTP (Biopython) search.",
                "details": str(e),
            }

    @mcp.tool()
    async def perform_blastn_search_biopython(
        sequence: str,
        ctx: Context,
        database: str = "nt",
        expect: float = 10.0,
        hitlist_size: int = 10,
        mcp_session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Performs a nucleotide sequence homology search (BLASTN) against a specified database using Biopython's NCBIWWW.qblast.

        Args:
            sequence (str): The nucleotide sequence to search with (FASTA format or raw sequence).
            database (str): The BLAST database to search against (e.g., 'nt', 'refseq_rna'). Defaults to 'nt'.
            expect (float): The expectation value (E-value) threshold. Defaults to 10.0.
            hitlist_size (int): The maximum number of hits to return. Defaults to 10.
            ctx (Context): The FastMCP context object.
            mcp_session_id (str, optional): The MCP session ID.

        Returns:
            Dict[str, Any]: A dictionary containing the search results or an error message.
        """
        session_id = await ensure_session_initialized_blastq(ctx, override_session_id=mcp_session_id)

        if not os.environ.get("NCBI_EMAIL"):
            msg = (
                "BLAST is unavailable: the NCBI_EMAIL environment variable is not set. "
                "NCBI requires a contact email to use their BLAST service. "
                "Please ask your administrator to set NCBI_EMAIL in the .env file."
            )
            logger.warning(msg)
            return {"error": msg}

        await ctx.info(
            f"Session {session_id}: Submitting BLASTN (Biopython) search for sequence (first 20 chars: {sequence[:20]}...) against {database}"
        )

        try:
            result_handle = await asyncio.to_thread(
                NCBIWWW.qblast,
                "blastn",
                database,
                sequence,
                expect=expect,
                hitlist_size=hitlist_size,
                format_type="XML",
            )
            logger.info(f"BLASTN (Biopython) qblast call completed for session {session_id}.")

            blast_record = await asyncio.to_thread(NCBIXML.read, result_handle)
            result_handle.close()
            logger.info(f"BLASTN (Biopython) results parsed for session {session_id}.")

            hits = blast_record.alignments
            if not hits:
                await ctx.info(f"BLASTN (Biopython) search returned no hits for session {session_id}.")
                return {
                    "query": sequence[:20] + "...",
                    "database": database,
                    "results": [],
                    "message": "BLAST search completed, but found no hits.",
                }

            truncated_hits = []
            for hit_alignment in hits:
                accession = hit_alignment.hit_id
                title = hit_alignment.hit_def
                hsp_summary = None
                if hit_alignment.hsps:
                    best_hsp = hit_alignment.hsps[0]
                    hsp_summary = {
                        "bit_score": best_hsp.bits,
                        "score": best_hsp.score,
                        "evalue": best_hsp.expect,
                        "identity_percent": (best_hsp.identities / best_hsp.align_length) * 100
                        if best_hsp.align_length > 0
                        else 0,
                        "query_from": best_hsp.query_start,
                        "query_to": best_hsp.query_end,
                        "hit_from": best_hsp.sbjct_start,
                        "hit_to": best_hsp.sbjct_end,
                        "align_len": best_hsp.align_length,
                    }
                truncated_hits.append({"accession": accession, "title": title, "hsp_summary": hsp_summary})

            await ctx.info(
                f"Successfully retrieved {len(truncated_hits)} BLASTN (Biopython) hits for session {session_id}."
            )
            return {
                "query": sequence[:20] + "...",
                "database": database,
                "results": truncated_hits,
            }
        except Exception as e:
            error_message = f"An unexpected error occurred during BLASTN (Biopython) search: {e}"
            logger.error(error_message, exc_info=True)
            await ctx.error("BLASTN (Biopython) search failed.")
            return {
                "error": "An unexpected error occurred during BLASTN (Biopython) search.",
                "details": str(e),
            }

    logger.info("BLASTQ (Biopython) MCP tool registered.")
