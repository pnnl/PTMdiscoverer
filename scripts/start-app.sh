#!/bin/bash
# ==============================================================================
# PTMdiscoverer + ADEPT  —  Service Launcher
#
# Builds and starts the Docker containers (MCP server + Streamlit app).
# Press Ctrl+C to stop; containers are torn down automatically.
# ==============================================================================

set -euo pipefail

BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}=======================================================${NC}"
    echo -e "${BLUE}  PTMdiscoverer + ADEPT — Service Manager${NC}"
    echo -e "${BLUE}=======================================================${NC}"
}

# --- Docker Compose detection ---
# Must be run from the repo root (where docker-compose.yaml lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

COMPOSE_FILE="docker-compose.yaml"
COMPOSE_CMD="docker compose --env-file .env -f $COMPOSE_FILE"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}Error: $COMPOSE_FILE not found. Ensure you run from the repo root or via scripts/start-app.sh.${NC}"
    exit 1
fi

# --- Check .env ---
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: No .env file found.${NC}"
    echo -e "${RED}  Copy .env.template → .env and fill in your API keys before launching.${NC}"
    exit 1
fi

# Load .env into this script so docker compose interpolation cannot be
# accidentally overridden by stale shell exports from a previous session.
set -a
# shellcheck disable=SC1091
source .env
set +a

# --- Sanity-check: pyproject.toml and src/ must exist at repo root ---
if [ ! -f "pyproject.toml" ] || [ ! -d "src/ptmdiscoverer" ]; then
    echo -e "${RED}Error: Run this script from the ptmdiscoverer repo root.${NC}"
    exit 1
fi

# --- Ensure runtime data directories exist for volume mounts ---
mkdir -p data/app_data/ptm_studies data/app_data/chroma_data data/app_data/uploaded_files

# --- Cleanup on exit ---
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    $COMPOSE_CMD down --remove-orphans 2>/dev/null || true
    echo -e "${GREEN}Services stopped. Goodbye!${NC}"
}

trap cleanup EXIT SIGINT SIGTERM

# --- Main ---
print_header

# Check for running containers
RUNNING=$($COMPOSE_CMD ps -q 2>/dev/null || true)
if [ -n "$RUNNING" ]; then
    echo -e "${YELLOW}Found existing containers for this project.${NC}"
    $COMPOSE_CMD ps
    read -p "$(echo -e "${YELLOW}Tear down before restarting? (y/N) ${NC}")" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $COMPOSE_CMD down --remove-orphans
        echo -e "${GREEN}Teardown complete.${NC}"
    else
        echo -e "${RED}Aborted. Stop existing containers first.${NC}"
        trap - EXIT SIGINT SIGTERM
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Building and starting services...${NC}"
echo -e "${GREEN}  MCP Server  → http://localhost:8080/mcp${NC}"
echo -e "${GREEN}  Streamlit   → http://localhost:8501${NC}"
echo -e "${GREEN}  INTERNAL_LLM_MODEL → ${INTERNAL_LLM_MODEL:-<unset>}${NC}"
echo -e "${GREEN}Press ${YELLOW}Ctrl+C${GREEN} to stop.${NC}"
echo ""

$COMPOSE_CMD up --build --force-recreate --remove-orphans
