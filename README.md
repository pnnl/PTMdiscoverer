
# PTMdiscoverer

PTMdiscoverer performs LLM-assisted PTM hypothesis generation from proteomics study inputs. It identifies delta masses in peptide fragmentation patterns from mass spectrometry data, combines a study-specific base prompt with observed protein-level evidence, and sends those prompts to an OpenAI-compatible LLM for PTM annotation and interpretation. Inference is prompt-based and grounded in the supplied study context and extracted peptide-level delta-mass evidence.

## Requirements

### Runtime

1. `uv` for installation and command execution
2. An OpenAI-compatible API key in `OPENAI_API_KEY`
3. Optionally `OPENAI_MODEL` to set the default model for CLI inference
4. Optionally `OPENAI_BASE_URL` for non-default endpoints

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Data requirements

PTMdiscoverer expects a study folder containing these inputs:

1. `base_prompt.txt`
2. `protein_metadata.json`
3. `dataset_labels.csv` if delta-mass CSVs still need to be extracted
4. `Results_MSFragger/<run>/psm.tsv` outputs when using extraction from MSFragger results

Minimum practical rule:

1. If per-protein delta-mass CSVs already exist, you can run inference from those plus the base prompt and metadata.
2. If per-protein delta-mass CSVs do not exist yet, you must also provide `dataset_labels.csv` and matching `Results_MSFragger/<run>/psm.tsv` files.

## Quick start

Use the standalone CLI for the fastest path:

```bash
uv sync
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=your-model-name              # for example: gpt-4o on OpenAI
export OPENAI_BASE_URL=https://api.openai.com/v1   # optional override

# Validate a study before inference
uv run ptmdiscoverer validate data/s_elongatus_case_study

# List proteins in a study
uv run ptmdiscoverer list data/s_elongatus_case_study

# Read or extract delta masses for one protein
uv run ptmdiscoverer extract data/s_elongatus_case_study Q9R6W2

# Run one inference pass
uv run ptmdiscoverer run data/s_elongatus_case_study --model your-model-name

# Run multiple passes and aggregate results
uv run ptmdiscoverer run data/s_elongatus_case_study --n-runs 3 --model your-model-name

# Inspect completed runs
uv run ptmdiscoverer results --study-folder data/s_elongatus_case_study

# Analyze one completed run using the per-protein prediction CSVs in that run directory
uv run ptmdiscoverer analyze --run-dir data/s_elongatus_case_study/output/output_<timestamp>

# Analyze an aggregated multi-run results CSV
uv run ptmdiscoverer analyze --aggregated-csv data/s_elongatus_case_study/output/aggregated_predictions_<timestamp>.csv
```

Model names are provider-specific. The examples above use `your-model-name` as a placeholder.
If you are using the public OpenAI endpoint, use a valid OpenAI model such as `gpt-4o`.
Internal aliases such as `o4-mini-project` may be valid only on private deployments.

All CLI commands print structured JSON so they are easy to inspect manually or wrap in shell scripts.

For `analyze`, the contract is intentionally simple:

1. `--run-dir` analyzes the per-protein `ptm_predictions_*.csv` files from one completed run.
2. `--aggregated-csv` analyzes one `aggregated_predictions_*.csv` file from a multi-run aggregation.

Both modes write a timestamped `analysis_<timestamp>/` directory containing:

1. `summary_<timestamp>.json`
2. `summary_<timestamp>.md`
3. timestamped CSV tables in `tables/`
4. timestamped PNG plots in `plots/` unless `--no-plots` is used

The analysis report also records provenance including the input path and the exact CSV files used.

The default analysis currently includes:

1. site-level PTM recurrence summaries
2. top PTM calls per protein
3. peptide-position recurrence summaries
4. PTM composition across top sites
5. protein-level rollups showing dominant sites and dominant PTM types
6. run-to-run disagreement summaries for sites with competing PTM identities
7. delta-mass hotspot summaries across recurring mass shifts

The default plots currently include:

1. recurrent PTM calls by site
2. top PTM calls across proteins
3. top peptide-position sites
4. PTM composition across top sites
5. a protein-by-site heatmap for top recurrent sites
6. recurring delta-mass hotspots
7. PTM composition by protein

## App usage
Use the integrated app, built using the [ADEPT framework](https://github.com/pnnl/adept-agentic-framework-core), when you want the browser UI and the broader multi-tool workflow.

Minimal setup:
```bash
cp .env.template .env
# fill in INTERNAL_LLM_API_KEY / INTERNAL_LLM_BASE_URL / INTERNAL_LLM_MODEL
mkdir -p data/app_data/ptm_studies
./scripts/start-app.sh
```

Then open `http://localhost:8501` and refer to studies as `ptm_studies/<study_name>`.

Recommended app flow:

1. validate the study folder
2. inspect proteins or delta masses if needed
3. run PTM inference
4. retrieve or reference the run outputs
5. analyze the completed run or aggregated CSV deterministically
6. optionally ground the results further with UniProt or PubChem

Example app prompts:

```text
Validate the study folder ptm_studies/s_elongatus, list the proteins, and extract delta masses for Q9R6W2.
```

```text
Run one PTM inference pass for ptm_studies/s_elongatus with model your-model-name and report the run directory that was created.
```

```text
Analyze the completed PTM run in ptm_studies/s_elongatus/output/output_<timestamp> and summarize the key findings, including high-consensus sites, disagreement sites, protein-level rollups, and delta-mass hotspots.
```

```text
Analyze the aggregated CSV ptm_studies/s_elongatus/output/aggregated_predictions_<timestamp>.csv and tell me which proteins have the most focused signal and which sites show competing PTM identities across runs.
```

In the app, the PTM analysis tool is deterministic: it generates timestamped JSON, Markdown, CSV, and optional PNG outputs that the agent can then summarize or contextualize.

## Input processing details

1. Inference reads `protein_metadata.json` from the current working directory or the supplied `base_data_folder`.
2. If a protein's delta-mass CSV is missing, PTMdiscoverer attempts extraction from `dataset_labels.csv` and `Results_MSFragger/<run>/psm.tsv`.
3. Extraction keeps rows with probability `> 0.9`, zero missed cleavages, delta mass `> 10`, and unique / stronger localization.
4. If extraction finds no qualifying rows for a protein, that protein is skipped and no placeholder file is created.
5. When new delta-mass CSVs are generated, `protein_metadata.json` is updated, so it should remain under version control.

## Development
For local development:

```bash
uv sync --group dev
uv run mypy .
uv run pytest -q
uv run ruff check .
uv run --extra adept pdoc -o docs/api ptmdiscoverer agentic_framework_pkg
```

If you want to run the ADEPT app stack locally without Docker for development:

```bash
uv sync --extra adept --group dev
```

PTMdiscoverer also exposes an MCP server and is tested with MCP Inspector for development:

```bash
uv sync --extra mcp
mcp dev src/ptmdiscoverer/mcp_server.py
```

A sample standalone MCP client config is available at `docs/examples/mcp_config.example.json`.

## Acknowledgements
This research was funded by the Generative AI for Science, Energy, and Security Science & Technology Investment under the Laboratory Directed Research and Development Program at Pacific Northwest National Laboratory (PNNL), a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy; 
Portions of this work were supported by the Center for AI at PNNL, the NW-BRaVE for Biopreparedness project funded by the U. S. Department of Energy (DOE), Office of Science, Biological and Environmental Research program, under FWP 81832, and performed at the Environmental Molecular Sciences Laboratory (EMSL), a DOE Office of Science User Facility sponsored by the Biological and Environmental Research program under Contract No. DE-AC05-76RL01830. 


## Author's Contact

aivett.bilbao@pnnl.gov and august.george@pnnl.gov
