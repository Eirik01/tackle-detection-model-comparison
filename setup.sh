# --- Set up the job environment ---
set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

echo "Loading modules..."
module --quiet purge
# Load the latest Python 3.11 or 3.12 (check with: module spider Python)
module load Python/3.13.1-GCCcore-14.2.0
module load CUDA/12.1.1  # Make sure this CUDA version works with your PyTorch install

# Deterministic cuBLAS GEMMs for reproducible training
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# --- Install uv if not already available ---
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# --- Sync dependencies with uv ---
echo "Syncing project dependencies with uv..."
uv sync

# --- Load environment variables ---
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# --- Work-area root for large, regenerable data (features, predictions,
#     models, results). Override via .env or the environment to run off-cluster;
#     defaults to the UiO FOX HPC work area. Mirrors FOX_DATADIR_PATH in
#     src/config.py so the shell scripts and Python agree on one location. ---
export FOX_DATADIR_PATH="${FOX_DATADIR_PATH:-/cluster/work/projects/ec12/ec-eirikto}"
echo "Work-area root (FOX_DATADIR_PATH): $FOX_DATADIR_PATH"