# dinov3-action-spotting

## Overview

This project contains the implementation and analysis for action spotting using DINOv3.

## Project Structure

- `data/`: Data files used in the project
- `processed_data/`: Intermediate files from the analysis
- `manuscript/`: Manuscript describing the results
- `results/`: Results of the analysis (data, tables, figures)
- `src/`: Contains all code in the project
- `doc/`: Documentation for the project

## Setup

This project uses `pyproject.toml` for dependency management and follows modern Python packaging standards.

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

#### Option 1: Using uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

#### Option 2: Using pip

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Migrating from Old Setup

If you're upgrading from the old `requirements.txt` setup:

1. **Remove old virtual environment:**
   ```bash
   rm -rf src/dinov3_action_spotting
   ```

2. **Create new virtual environment at root:**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

3. **The project now uses:**
   - `pyproject.toml` for dependency specification
   - `uv.lock` for exact version locking (auto-generated)
   - `.venv/` at project root instead of inside `src/`

## Getting Started

See `pyproject.toml` for all software dependencies and their exact versions.

## License

See `src/LICENSE` for license information.

## Download data

Navigate to src folder and run the download script.


## Environment