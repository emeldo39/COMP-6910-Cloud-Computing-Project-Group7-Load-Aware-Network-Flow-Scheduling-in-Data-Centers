#!/usr/bin/env bash
# =============================================================================
# LAFS Project — Python Virtual Environment Setup
# Usage: source setup/setup_env.sh         (from project root)
#   OR:  bash setup/setup_env.sh           (creates venv only, doesn't activate)
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

# ── Project root detection ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"

info "Project root: $PROJECT_ROOT"

# ── Python version check ──────────────────────────────────────────────────────
PYTHON_BIN=$(which python3 2>/dev/null || which python 2>/dev/null)
[[ -z "$PYTHON_BIN" ]] && error "Python 3 not found. Install it first."

PY_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON_BIN" -c "import sys; print(sys.version_info.minor)")

info "Python version: $PY_VER (at $PYTHON_BIN)"

# Ryu supports Python 3.8–3.11; flag 3.12+ as warning
if [[ $PY_MAJOR -lt 3 ]] || [[ $PY_MAJOR -eq 3 && $PY_MINOR -lt 8 ]]; then
    error "Python 3.8+ required. Found: $PY_VER"
fi
if [[ $PY_MAJOR -eq 3 && $PY_MINOR -ge 12 ]]; then
    warn "Python $PY_VER detected. Ryu 4.34 may have issues on Python 3.12+."
    warn "Recommended: Python 3.10 or 3.11."
fi

# ── Create virtual environment ────────────────────────────────────────────────
step "Creating virtual environment at $VENV_DIR ..."
if [ -d "$VENV_DIR" ]; then
    warn "venv already exists. Delete it first to recreate: rm -rf $VENV_DIR"
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    info "Virtual environment created."
fi

# ── Activate ──────────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
info "Virtual environment activated: $(which python)"

# ── Upgrade pip & install build tools ────────────────────────────────────────
step "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --quiet

# ── Install project requirements ──────────────────────────────────────────────
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
[[ ! -f "$REQUIREMENTS" ]] && error "requirements.txt not found at $REQUIREMENTS"

step "Installing Python dependencies from requirements.txt..."
pip install -r "$REQUIREMENTS" --quiet

# ── Handle Gurobi separately (optional — may not have license yet) ────────────
step "Attempting Gurobi Python API install..."
if pip install gurobipy==11.0.3 --quiet 2>/dev/null; then
    info "gurobipy installed. Remember to activate your academic license."
    info "See setup/gurobi_setup.md for license instructions."
else
    warn "gurobipy install failed (likely no license). Using PuLP/CBC as fallback."
    pip install pulp --quiet
fi

# ── Install project package in editable mode ──────────────────────────────────
if [ -f "$PROJECT_ROOT/setup.py" ] || [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
    step "Installing LAFS package in editable mode..."
    pip install -e "$PROJECT_ROOT" --quiet
fi

# ── Create .env file for environment variables ────────────────────────────────
ENV_FILE="$PROJECT_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    step "Creating .env file for environment variables..."
    cat > "$ENV_FILE" << 'ENVFILE'
# LAFS Project Environment Variables
# Copy this file and adjust as needed.

# Gurobi license (set if using WLS/cloud license)
# GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic

# Experiment configuration
LAFS_LOG_LEVEL=INFO
LAFS_DATA_DIR=./data
LAFS_RESULTS_DIR=./results

# Ryu controller settings
RYU_CONTROLLER_HOST=127.0.0.1
RYU_CONTROLLER_PORT=6633
RYU_REST_PORT=8080

# MILP solver selection: gurobi | pulp | scipy
LAFS_SOLVER=gurobi

# Prediction window in seconds (0.1–1.0)
LAFS_PRED_INTERVAL=0.5
ENVFILE
    info ".env file created."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
info "Environment setup complete!"
echo ""
echo "  Virtual env : $VENV_DIR"
echo "  Python      : $(python --version)"
echo "  pip         : $(pip --version | cut -d' ' -f1-2)"
echo ""
echo "  To activate in future sessions:"
echo "    source venv/bin/activate"
echo ""
echo "  To verify all dependencies:"
echo "    python setup/verify_deps.py"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
