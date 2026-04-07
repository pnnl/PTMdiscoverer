import asyncio
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import Client as MCPClient
from langchain.tools import BaseTool
from pydantic import (
    BaseModel,
    Field,
)  # For describing tool inputs if not using default string

from ..logger_config import get_logger  # Use centralized logger

logger = get_logger(__name__)

# Load environment variables from .env file at the root of the codebase
root_dir = Path(__file__).parent.parent.parent.parent  # Navigate to root from current file
dotenv_path = root_dir / ".env"
load_dotenv(dotenv_path)

# Determine MCP Server URL for Langchain tools.
# Priority:
# 1. MCP_SERVER_URL_FOR_LANGCHAIN (explicit override)
# 2. MCP_SERVER_URL_LANGCHAIN_LOCAL (if defined, for local dev)
# 3. MCP_SERVER_URL_LANGCHAIN_LOCAL_DEFAULT (default for local)
# If running in Docker and need to connect to 'mcp_server' service,
# set MCP_SERVER_URL_FOR_LANGCHAIN=http://mcp_server:8080/mcp in the Langchain agent's environment.

MCP_SERVER_URL_LANGCHAIN_LOCAL_DEFAULT = "http://localhost:8080/mcp"
MCP_SERVER_URL_FOR_LANGCHAIN = os.getenv(
    "MCP_SERVER_URL_FOR_LANGCHAIN",
    os.getenv("MCP_SERVER_URL_LANGCHAIN_LOCAL", MCP_SERVER_URL_LANGCHAIN_LOCAL_DEFAULT),
)

# Timeout for long-running MCP tool calls (e.g. multi-run PTM inference).
# The SSE read timeout controls how long the client waits for the server
# to finish processing before giving up.  Default: 60 minutes.
_MCP_TOOL_TIMEOUT_SECONDS = int(os.getenv("MCP_TOOL_TIMEOUT", "3600"))


class MCPToolWrapper(BaseTool):
    """
    A Langchain BaseTool wrapper for calling tools on a remote MCP server.
    """

    mcp_client_url: str = Field(default_factory=lambda: MCP_SERVER_URL_FOR_LANGCHAIN)
    actual_tool_name: str
    mcp_session_id: str | None = Field(None, description="The MCP session ID for stateful operations.")
    # args_schema is inherited from BaseTool as Optional[Type[BaseModel]] = None.
    # Do NOT redeclare it here — in LangChain 1.x / Pydantic v2, a TypeVar-bound
    # redeclaration triggers a name-construction path that doubles the tool name
    # when the schema class is named {tool_name}Input.

    @property
    def tool_call_schema(self):
        """Return args_schema directly, bypassing BaseTool._create_subset_model.

        BaseTool.tool_call_schema calls _create_subset_model(self.name, ...) which
        produces a new Pydantic model whose title == self.name.  In LangChain 1.x,
        _convert_json_schema_to_openai_function concatenates the explicit name= kwarg
        with the schema title, doubling the tool name (e.g. 'FooFoo').

        Returning args_schema directly gives the schema a distinct title (the class
        name of the Input model), so _format_tool_to_openai_function honours the
        explicit name=tool.name and the final OpenAI function name is correct.
        """
        return self.args_schema

    # This method is called by Langchain if the tool is synchronous
    def _run(self, **kwargs: Any) -> str:
        try:
            logger.info(f"Running MCPToolWrapper _run for {self.actual_tool_name} with kwargs: {kwargs}")
            loop = asyncio.get_event_loop()
            if loop.is_running():
                try:
                    logger.debug(
                        f"Loop is running: Calling MCP tool '{self.actual_tool_name}' in sync context with params: {kwargs}"
                    )
                    return asyncio.run(self._arun(**kwargs))
                except RuntimeError as e:
                    if "cannot be called when another loop is running" in str(e) or "Event loop is closed" in str(e):
                        logger.warning("Asyncio loop issue in _run, trying nest_asyncio or new loop.")
                        import nest_asyncio

                        nest_asyncio.apply()
                        return asyncio.run(self._arun(**kwargs))
                    raise e
            logger.debug(f"Calling MCP tool '{self.actual_tool_name}' with params: {kwargs}")
            return asyncio.run(self._arun(**kwargs))
        except Exception as e:
            logger.error(
                f"Error in MCPToolWrapper _run for {self.actual_tool_name}: {e}",
                exc_info=True,
            )
            return f"Error executing tool {self.actual_tool_name}: {e}"

    # This method is called by Langchain if the tool is asynchronous
    async def _arun(self, **kwargs: Any) -> str:
        # When args_schema is used, Langchain passes parsed arguments as kwargs.
        logger.info(f"Running MCPToolWrapper _arun for {self.actual_tool_name} with kwargs: {kwargs}")
        params = kwargs

        # These are the args from Langchain, based on the tool's args_schema.
        # The mcp_session_id should be included here by the LLM if the schema defines it.
        params_for_tool_call = params.copy()

        # Check if mcp_session_id was provided by the LLM (as per args_schema).
        # If not, and if the wrapper has a default mcp_session_id, use that.
        # Otherwise, the tool will be called without an explicit mcp_session_id in its params,
        # which might be fine for some tools or lead to issues for stateful ones.
        if "mcp_session_id" not in params_for_tool_call or params_for_tool_call.get("mcp_session_id") is None:
            logger.info(
                f"mcp_session_id not provided by LLM for tool '{self.actual_tool_name}'. Checking wrapper's default."
            )
            if self.mcp_session_id:  # Fallback to wrapper's own session_id if LLM didn't provide
                params_for_tool_call["mcp_session_id"] = self.mcp_session_id
                logger.debug(
                    f"LLM did not provide mcp_session_id for tool '{self.actual_tool_name}'. "
                    f"Using wrapper's default: {self.mcp_session_id}"
                )
            else:
                # This warning is important. If it appears, the LLM isn't providing the session_id
                # despite the prompt and schema, and no default is set on the wrapper.
                logger.warning(
                    f"MCP Session ID not provided by LLM and no default in wrapper for tool '{self.actual_tool_name}'. "
                    f"Tool may operate in a new/default session or fail if it strictly requires one. "
                    f"Params received from LLM: {kwargs}"
                )
        else:
            logger.debug(
                f"MCP Session ID '{params_for_tool_call.get('mcp_session_id')}' provided by LLM for tool '{self.actual_tool_name}'."
            )

        logger.debug(
            f"MCPToolWrapper _arun for {self.actual_tool_name}: "
            f"raw_kwargs_from_langchain={kwargs}, "
            f"final_params_for_tool_call={params_for_tool_call}"
        )
        try:
            logger.debug(
                f"Connecting to MCP server at {self.mcp_client_url} for tool {self.actual_tool_name} using URL: {self.mcp_client_url}"
            )
            # Use a generous timeout so long-running tools (e.g. multi-run
            # PTM inference) don't trigger httpx.ReadTimeout on the SSE
            # stream.  fastmcp >= 2.6 / mcp >= 1.6 accept a timedelta.
            async with MCPClient(
                self.mcp_client_url,
                timeout=timedelta(seconds=_MCP_TOOL_TIMEOUT_SECONDS),
            ) as client:
                logger.debug(
                    f"Calling MCP tool '{self.actual_tool_name}' on server with params: {params_for_tool_call}"
                )
                result_content_list = await client.call_tool(self.actual_tool_name, params_for_tool_call)

                # Normalise FastMCP 2.x (List[ContentPart]) and
                # FastMCP 3.x (CallToolResult with .content) responses.
                content_list = (
                    result_content_list.content
                    if hasattr(result_content_list, "content")
                    else (result_content_list if isinstance(result_content_list, list) else [])
                )
                if content_list:
                    first_content_part = content_list[0]
                    if hasattr(first_content_part, "text"):  # Handles TextContent
                        return str(first_content_part.text)
                    else:
                        return str(content_list)
                elif result_content_list is None:
                    return ""

                return str(result_content_list)  # Fallback for other types or empty list
        except ConnectionRefusedError:
            logger.error(
                f"MCP Server connection refused at {self.mcp_client_url} for tool {self.actual_tool_name}.",
                exc_info=True,
            )
            return f"Error: Could not connect to MCP server to run tool {self.actual_tool_name}."
        except Exception as e:
            logger.error(
                f"Exception calling MCP tool {self.actual_tool_name}: {e}",
                exc_info=True,
            )
            return f"Error during MCP tool {self.actual_tool_name} execution: {e}"


# --- Define Pydantic models for structured tool inputs (recommended) ---
class QueryCSVToolInput(BaseModel):
    user_query: str = Field(description="The natural language query for the CSV data.")
    file_id: str = Field(description="The file_id of the CSV registered via 'register_uploaded_csv'.")
    mcp_session_id: str | None = Field(
        None, description="The MCP session ID for stateful context."
    )  # Agent needs to manage this
    top_k: int = Field(3, description="Number of top relevant chunks to retrieve for RAG.")


class PerformCalculationInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class StoreNoteInput(BaseModel):
    note_text: str = Field(description="The text of the note to store.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class RetrieveNotesInput(BaseModel):
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class QueryUniProtArgs(BaseModel):
    accession_id: str = Field(description="The UniProt accession ID (e.g., P05067).")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query for the web search.")
    num_results: int = Field(5, description="Number of search results to return.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


# --- Pydantic models for Biopython-based BLAST tools (can be same as above if params match) ---
class PerformBlastpSearchBiopythonInput(BaseModel):
    sequence: str = Field(description="The protein sequence to search with (FASTA format or raw sequence).")
    database: str = Field(
        "nr",
        description="The BLAST database to search against (e.g., 'nr', 'swissprot', 'pdb'). Defaults to 'nr'.",
    )
    expect: float = Field(10.0, description="The expectation value (E-value) threshold. Defaults to 10.0.")
    hitlist_size: int = Field(10, description="The maximum number of hits to return. Defaults to 10.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class PerformBlastnSearchBiopythonInput(BaseModel):
    sequence: str = Field(description="The nucleotide sequence to search with (FASTA format or raw sequence).")
    database: str = Field(
        "nt",
        description="The BLAST database to search against (e.g., 'nt', 'refseq_rna'). Defaults to 'nt'.",
    )
    expect: float = Field(10.0, description="The expectation value (E-value) threshold. Defaults to 10.0.")
    hitlist_size: int = Field(10, description="The maximum number of hits to return. Defaults to 10.")
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


# --- Pydantic models for PubChem tools (matching pubchem_tool.py parameters) ---
class SearchPubChemQueryInput(BaseModel):
    query: str = Field(
        description="The compound name, identifier (e.g., CID, SMILES), or general search query for PubChem."
    )
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")
    max_results: int = Field(5, description="Maximum number of results to return. Defaults to 5.")


class SearchPubChemByNameArgs(BaseModel):
    chemical_name: str = Field(
        description="The compound name, identifier (e.g., CID, SMILES), or search query for PubChem."
    )
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")
    max_results: int = Field(5, description="Maximum number of results to return. Defaults to 5.")


class GetPubChemCompoundPropertiesArgs(BaseModel):
    cid: int = Field(description="The PubChem Compound ID (CID) for which to retrieve properties.")
    properties: list[str] | None = Field(
        None,
        description="A list of specific properties to retrieve (e.g., ['molecular_formula', 'iupac_name']). If None, a default set is returned.",
    )
    mcp_session_id: str | None = Field(None, description="The MCP session ID.")


class ListUploadedFilesArgs(BaseModel):
    mcp_session_id: str | None = Field(None, description="The MCP session ID for context.")


# --- Factory functions to create Langchain tool instances ---
def get_mcp_query_csv_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="QueryProcessedDocumentData",  # Renamed for broader scope
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,  # Use the determined URL
        actual_tool_name="query_file_content",  # Name on MCP Server
        description="Queries a previously uploaded and processed document (e.g., CSV, PDF, DOCX, image) using its file_id. Use this to get answers from the document's content.",
        args_schema=QueryCSVToolInput,  # Crucial for LLM to know how to structure input
        mcp_session_id=mcp_session_id,
    )


def get_mcp_perform_calculation_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="PerformSimpleCalculation",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,  # Use the determined URL
        actual_tool_name="perform_calculation",
        description="Evaluates a simple mathematical expression (e.g., '2 + 2 * 3').",
        args_schema=PerformCalculationInput,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_store_note_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="StoreNoteInSession",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,  # Use the determined URL
        actual_tool_name="store_note_in_session",
        description="Stores a textual note in the current user's session.",
        args_schema=StoreNoteInput,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_retrieve_notes_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="RetrieveSessionNotes",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,  # Use the determined URL
        actual_tool_name="retrieve_session_notes",
        description="Retrieves all notes previously stored in the current user's session.",
        args_schema=RetrieveNotesInput,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_query_uniprot_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="QueryUniProt",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,  # Use the determined URL
        actual_tool_name="query_uniprot_by_accession",  # Name on MCP Server
        description="Queries the UniProt database for protein information using an accession ID.",
        args_schema=QueryUniProtArgs,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_web_search_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="PerformWebSearch",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,  # Use the determined URL
        actual_tool_name="perform_web_search",
        description="Performs a web search using a search engine (e.g., Google) and returns a list of results including titles, links, and snippets.",
        args_schema=WebSearchInput,
        mcp_session_id=mcp_session_id,
    )


# --- Factory function for PubChem tool ---
def get_mcp_search_pubchem_by_query_tool_langchain(
    mcp_session_id: str | None = None,
):
    return MCPToolWrapper(
        name="SearchPubChemByQuery",  # Descriptive name for the agent
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="search_pubchem_by_query",  # Correct name on MCP Server from pubchem_tool.py
        description="Searches the PubChem database for chemical compounds by query string and returns matching CIDs and basic info.",
        args_schema=SearchPubChemQueryInput,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_search_pubchem_by_name_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="SearchPubChemByName",  # Descriptive name for the agent
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="search_pubchem_by_name",  # Correct name on MCP Server from pubchem_tool.py
        description="Searches the PubChem database for chemical compounds by name or identifier and returns matching CIDs and basic info.",
        args_schema=SearchPubChemByNameArgs,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_get_pubchem_compound_properties_tool_langchain(
    mcp_session_id: str | None = None,
):
    return MCPToolWrapper(
        name="GetPubChemCompoundProperties",  # Descriptive name for the agent
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="get_pubchem_compound_properties",  # Correct name on MCP Server from pubchem_tool.py
        description="Retrieves detailed properties (like formula, weight, SMILES, InChI, IUPAC name) for a specific PubChem Compound ID (CID).",
        args_schema=GetPubChemCompoundPropertiesArgs,
        mcp_session_id=mcp_session_id,
    )


# --- Factory functions for Biopython-based BLAST tools ---
def get_mcp_blastp_biopython_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="PerformProteinBlastSearchBiopython",  # Descriptive name for the agent
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="perform_blastp_search_biopython",  # Name on MCP Server from blastq_tool.py
        description="Performs a protein sequence homology search (BLASTP) using Biopython against a database via the NCBI web service. Requires a protein sequence.",
        args_schema=PerformBlastpSearchBiopythonInput,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_blastn_biopython_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="PerformNucleotideBlastSearchBiopython",  # Descriptive name for the agent
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="perform_blastn_search_biopython",  # Name on MCP Server from blastq_tool.py
        description="Performs a nucleotide sequence homology search (BLASTN) using Biopython against a database via the NCBI web service. Requires a nucleotide sequence.",
        args_schema=PerformBlastnSearchBiopythonInput,
        mcp_session_id=mcp_session_id,
    )


def get_mcp_list_uploaded_files_tool_langchain(mcp_session_id: str | None = None):
    return MCPToolWrapper(
        name="ListUploadedFiles",
        mcp_client_url=MCP_SERVER_URL_FOR_LANGCHAIN,
        actual_tool_name="list_uploaded_files",
        description="Lists all files that have been uploaded and processed in the current session, showing their file_id and original filename. Use this to know what files are available for querying.",
        args_schema=ListUploadedFilesArgs,
        mcp_session_id=mcp_session_id,
    )
