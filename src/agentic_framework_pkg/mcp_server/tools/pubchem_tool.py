import asyncio
from typing import Any

import pubchempy as pcp
from fastmcp import Context, FastMCP

from ...logger_config import get_logger

logger = get_logger(__name__)


def register_tools(mcp: FastMCP):
    @mcp.tool()
    async def search_pubchem_by_name(
        ctx: Context, chemical_name: str, max_results: int = 5, mcp_session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Searches PubChem for compounds by chemical name using the pubchempy library.

        Args:
            chemical_name (str): The name of the chemical to search for (e.g., "aspirin", "glucose").
            max_results (int): Maximum number of results to return. Defaults to 5.
            ctx (Context): The FastMCP context object.
            mcp_session_id (str, optional): The MCP session ID.

        Returns:
            Dict[str, Any]: A dictionary containing a list of matching compound CIDs and their synonyms, or an error.
        """
        await ctx.info(
            f"Searching PubChem (pubchempy) for chemical name: '{chemical_name}', max_results: {max_results}"
        )

        try:
            compounds: list[pcp.Compound] = await asyncio.to_thread(
                pcp.get_compounds, chemical_name, "name", listkey_count=max_results
            )

            if not compounds:
                await ctx.info(f"No compounds found for '{chemical_name}' using pubchempy.")
                return {"query": chemical_name, "results": [], "message": "No compounds found for the given name."}

            results = []
            for compound in compounds[:max_results]:
                syns = compound.synonyms[:5] if compound.synonyms else []
                results.append(
                    {
                        "cid": compound.cid,
                        "synonyms": syns,
                        "title": compound.iupac_name or (syns[0] if syns else "N/A"),
                    }
                )

            return {"query": chemical_name, "results": results}

        except pcp.PubChemPyError as e:
            err_msg = f"PubChemPy API request failed for '{chemical_name}': {e}"
            logger.error(err_msg, exc_info=True)
            await ctx.error(f"PubChemPy API error: {e}")
            return {"error": "Failed to fetch data from PubChem using pubchempy.", "details": err_msg}
        except Exception as e:
            err_msg = f"An unexpected error occurred during PubChem (pubchempy) search for '{chemical_name}': {e}"
            logger.error(err_msg, exc_info=True)
            await ctx.error("PubChem (pubchempy) search failed.")
            return {"error": "An unexpected error occurred during PubChem (pubchempy) search.", "details": str(e)}

    @mcp.tool()
    async def get_pubchem_compound_properties(
        ctx: Context, cid: int, properties: list[str] | None = None, mcp_session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Retrieves specified properties for a given PubChem Compound ID (CID) using the pubchempy library.

        Args:
            cid (int): The PubChem Compound ID (e.g., 2244 for aspirin).
            properties (List[str], optional): A list of PubChemPy Compound attribute names to retrieve
                                              (e.g., ["molecular_formula", "molecular_weight", "inchi_key", "iupac_name"]).
                                              Defaults to a basic set if None.
            ctx (Context): The FastMCP context object.
            mcp_session_id (str, optional): The MCP session ID.

        Returns:
            Dict[str, Any]: A dictionary containing the compound's properties or an error.
        """
        await ctx.info(f"Fetching properties (pubchempy) for PubChem CID: {cid}")

        if properties is None:
            properties = [
                "molecular_formula",
                "molecular_weight",
                "canonical_smiles",
                "isomeric_smiles",
                "inchi",
                "inchi_key",
                "iupac_name",
                "synonyms",
            ]

        try:
            compound: pcp.Compound | None = await asyncio.to_thread(pcp.Compound.from_cid, cid)

            if not compound:
                await ctx.warning(f"Compound with CID {cid} not found using pubchempy.")
                return {"cid": cid, "properties": {}, "message": f"Compound with CID {cid} not found."}

            compound_props: dict[str, Any] = {}
            for prop_name in properties:
                if hasattr(compound, prop_name):
                    value = getattr(compound, prop_name)
                    if prop_name == "synonyms" and isinstance(value, list):
                        compound_props[prop_name] = value[:5]
                    else:
                        compound_props[prop_name] = value
                else:
                    compound_props[prop_name] = None

            return {"cid": cid, "properties": compound_props}

        except pcp.PubChemPyError as e:
            err_msg = f"PubChemPy API property request failed for CID {cid}: {e}"
            logger.error(err_msg, exc_info=True)
            await ctx.error(f"PubChemPy API error: {e}")
            return {
                "error": f"Failed to fetch properties for CID {cid} from PubChem using pubchempy.",
                "details": err_msg,
            }
        except Exception as e:
            err_msg = f"An unexpected error occurred fetching properties (pubchempy) for CID {cid}: {e}"
            logger.error(err_msg, exc_info=True)
            await ctx.error("PubChem (pubchempy) property retrieval failed.")
            return {"error": f"An unexpected error occurred for CID {cid} (pubchempy).", "details": str(e)}

    @mcp.tool()
    async def search_pubchem_by_query(
        ctx: Context, query: str, max_results: int = 5, mcp_session_id: str | None = None
    ) -> dict[str, Any]:
        """
        Searches PubChem for compounds by general terms using the pubchempy library.

        Args:
            query (str): The general query terms to search for (e.g., "aspirin", "glucose").
            max_results (int): Maximum number of results to return. Defaults to 5.
            ctx (Context): The FastMCP context object.
            mcp_session_id (str, optional): The MCP session ID.

        Returns:
            Dict[str, Any]: A dictionary containing a list of matching compound CIDs and their synonyms, or an error.
        """
        await ctx.info(f"Searching PubChem (pubchempy) for query: '{query}', max_results: {max_results}")

        try:
            compounds: list[pcp.Compound] = await asyncio.to_thread(
                pcp.get_compounds, query, "name", listkey_count=max_results
            )

            if not compounds:
                await ctx.info(f"No compounds found for query '{query}' using pubchempy.")
                return {"query": query, "results": [], "message": "No compounds found for the given query."}

            results: list[dict[str, Any]] = []
            for compound in compounds[:max_results]:
                syns = compound.synonyms[:5] if compound.synonyms else []
                results.append(
                    {
                        "cid": compound.cid,
                        "synonyms": syns,
                        "title": compound.iupac_name or (syns[0] if syns else "N/A"),
                    }
                )

            return {"query": query, "results": results}

        except pcp.PubChemPyError as e:
            err_msg = f"PubChemPy API request failed for query '{query}': {e}"
            logger.error(err_msg, exc_info=True)
            await ctx.error(f"PubChemPy API error: {e}")
            return {"error": "Failed to fetch data from PubChem using pubchempy.", "details": err_msg}
        except Exception as e:
            err_msg = f"An unexpected error occurred during PubChem (pubchempy) search for query '{query}': {e}"
            logger.error(err_msg, exc_info=True)
            await ctx.error("PubChem (pubchempy) search failed.")
            return {"error": "An unexpected error occurred during PubChem (pubchempy) search.", "details": str(e)}

    logger.info("PubChem MCP tools (using pubchempy) registered.")
