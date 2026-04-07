import json
import uuid
from typing import Any

import httpx
from fastmcp import Context, FastMCP

from ...logger_config import get_logger
from ..state_manager import create_session_if_not_exists

logger = get_logger(__name__)


async def ensure_session_initialized_uniprot(ctx: Context):
    """Helper to ensure the session exists in the database."""
    session_id = get_stable_session_id_uniprot(ctx)
    client_id = ctx.client_id if hasattr(ctx, "client_id") else None
    await create_session_if_not_exists(session_id, client_id)
    return session_id


async def ensure_session_initialized_uniprot_with_override(ctx: Context, override_session_id: str | None = None) -> str:
    session_id = override_session_id or get_stable_session_id_uniprot(ctx)
    client_id = ctx.client_id if hasattr(ctx, "client_id") else None
    await create_session_if_not_exists(session_id, client_id)
    return session_id


def get_stable_session_id_uniprot(ctx: Context) -> str:
    if hasattr(ctx, "session_id") and ctx.session_id:
        return ctx.session_id
    if ctx.request_id:
        return ctx.request_id
    logger.warning("UniProt Tool: Could not determine a stable session ID. Generating a new one.")
    return str(uuid.uuid4())


def register_tools(mcp: FastMCP):
    @mcp.tool()
    async def query_uniprot_by_accession(
        accession_id: str, ctx: Context, mcp_session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Queries the UniProt database for a given protein accession ID and returns its data.
        Example accession ID: P05067 (for APP_HUMAN - Amyloid beta A4 protein)
        """
        session_id = await ensure_session_initialized_uniprot_with_override(ctx, mcp_session_id)
        await ctx.info(f"Session {session_id}: Querying UniProt for accession ID: {accession_id}")

        uniprot_api_url = f"https://rest.uniprot.org/uniprotkb/{accession_id}?format=json"

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(uniprot_api_url)
                response.raise_for_status()

                data = response.json()
                await ctx.info(f"Successfully retrieved data for UniProt ID {accession_id}")

                truncated_data: dict[str, Any] = {
                    "accession": data.get("accession"),
                    "id": data.get("id"),
                    "proteinDescription": data.get("proteinDescription"),
                    "genes": data.get("genes"),
                    "organism": data.get("organism"),
                    "sequenceLength": data.get("sequence", {}).get("length"),
                    "sequenceMass": data.get("sequence", {}).get("mass"),
                    "featuresSummary": [
                        f.get("type")
                        for f in data.get("features", [])
                        if f.get("type") in ["Domain", "Binding site", "Active site", "Chain", "Peptide"]
                    ],
                    "commentsSummary": [
                        c.get("type")
                        for c in data.get("comments", [])
                        if c.get("type") in ["Function", "Subunit structure", "Interaction", "Disease"]
                    ],
                }

                return truncated_data
        except httpx.HTTPStatusError as e:
            error_message = f"UniProt API request failed for {accession_id} with status {e.response.status_code}. Response: {e.response.text[:200]}"
            logger.error(error_message, exc_info=True)
            await ctx.error(f"UniProt API error: {e.response.status_code}")
            return {"error": "Failed to fetch data from UniProt.", "details": error_message}
        except httpx.RequestError as e:
            error_message = f"Request to UniProt failed for {accession_id}: {e}"
            logger.error(error_message, exc_info=True)
            await ctx.error("UniProt request failed.")
            return {"error": "Failed to connect to UniProt.", "details": str(e)}
        except json.JSONDecodeError as e:
            error_message = f"Failed to parse JSON response from UniProt for {accession_id}: {e}"
            logger.error(error_message, exc_info=True)
            await ctx.error("UniProt response parsing error.")
            return {"error": "Invalid response format from UniProt.", "details": str(e)}

    @mcp.tool()
    async def query_uniprot_by_gene(
        gene_query: str, ctx: Context, organism_id: str = "9606", limit: int = 5, mcp_session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Queries the UniProt database for proteins associated with a given gene name or symbol.
        Defaults to human (organism_id: 9606) and returns up to 5 results.
        Example gene_query: APP
        Example organism_id: 9606 (Homo sapiens), 10090 (Mus musculus)
        """
        session_id = await ensure_session_initialized_uniprot_with_override(ctx, mcp_session_id)
        await ctx.info(
            f"Session {session_id}: Querying UniProt for gene: '{gene_query}', organism: {organism_id}, limit: {limit}"
        )

        query_string = f"gene:{gene_query} AND organism_id:{organism_id}"

        uniprot_search_url = f"https://rest.uniprot.org/uniprotkb/search?query={query_string}&format=json&size={limit}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(uniprot_search_url)
                response.raise_for_status()

                data = response.json()
                results = data.get("results", [])

                if not results:
                    await ctx.info(f"No UniProt entries found for gene query: '{gene_query}' in organism {organism_id}")
                    return {
                        "query": gene_query,
                        "organism_id": organism_id,
                        "results": [],
                        "message": "No matching entries found.",
                    }

                truncated_results = []
                for entry in results:
                    truncated_entry = {
                        "accession": entry.get("primaryAccession"),
                        "id": entry.get("uniProtkbId"),
                        "proteinDescription": entry.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value"),
                        "geneNames": [
                            gn.get("geneName", {}).get("value") for gn in entry.get("genes", []) if gn.get("geneName")
                        ],
                        "organism": entry.get("organism", {}).get("scientificName"),
                    }
                    truncated_results.append(truncated_entry)

                await ctx.info(f"Successfully retrieved {len(truncated_results)} entries for gene query '{gene_query}'")
                return {"query": gene_query, "organism_id": organism_id, "results": truncated_results}

        except httpx.HTTPStatusError as e:
            error_message = f"UniProt API search failed for gene '{gene_query}' with status {e.response.status_code}. Response: {e.response.text[:200]}"
            logger.error(error_message, exc_info=True)
            await ctx.error(f"UniProt API search error: {e.response.status_code}")
            return {"error": "Failed to search UniProt by gene.", "details": error_message}
        except Exception as e:
            error_message = f"An error occurred while searching UniProt for gene '{gene_query}': {e}"
            logger.error(error_message, exc_info=True)
            await ctx.error("UniProt gene search failed.")
            return {"error": "An unexpected error occurred during UniProt gene search.", "details": str(e)}

    logger.info("UniProt query MCP tool registered.")
