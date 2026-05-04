
# PTMdiscoverer

[![CI](https://github.com/pnnl/PTMdiscoverer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pnnl/PTMdiscoverer/actions/workflows/ci.yml)
[![Security Scan](https://github.com/pnnl/PTMdiscoverer/actions/workflows/security.yml/badge.svg?branch=main)](https://github.com/pnnl/PTMdiscoverer/actions/workflows/security.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: BSD](https://img.shields.io/badge/license-BSD-lightgrey.svg)](LICENSE)

LLM-assisted PTM hypothesis generation from proteomics mass spectrometry data. Identifies delta masses in peptide fragmentation patterns and uses an OpenAI-compatible LLM to annotate and interpret them.

## Install

```bash
# requires uv — https://astral.sh/uv
uv sync
```

For the [ADEPT](https://github.com/pnnl/adept-agentic-framework-core) browser UI and full agentic workflow:

```bash
uv sync --extra adept
```

## Quick start

```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o          # any OpenAI-compatible model name

uv run ptmdiscoverer validate data/my_study
uv run ptmdiscoverer run     data/my_study --model gpt-4o
uv run ptmdiscoverer analyze --run-dir data/my_study/output/output_<timestamp>
```

For the browser UI:

```bash
cp .env.template .env   # fill in INTERNAL_LLM_* vars
./scripts/start-app.sh  # opens http://localhost:8501
```

→ **[Full usage guide](docs/USAGE.md)** — CLI reference, app workflow, data requirements, development setup, and acknowledgements.

## Acknowledgements

This research was funded by the Generative AI for Science, Energy, and Security Science & Technology Investment under the Laboratory Directed Research and Development Program at Pacific Northwest National Laboratory (PNNL), a multiprogram national laboratory operated by Battelle for the U.S. Department of Energy.

Portions of this work were supported by the Center for AI at PNNL, the NW-BRaVE for Biopreparedness project funded by the U.S. Department of Energy (DOE), Office of Science, Biological and Environmental Research program, under FWP 81832, and performed at the Environmental Molecular Sciences Laboratory (EMSL), a DOE Office of Science User Facility sponsored by the Biological and Environmental Research program under Contract No. DE-AC05-76RL01830.

## Contact

aivett.bilbao@pnnl.gov · august.george@pnnl.gov

## Citation

Software citation for version 1.0: [https://doi.org/10.5281/zenodo.20030405](https://doi.org/10.5281/zenodo.20030405) 
