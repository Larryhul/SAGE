# Installation

## Environment

The original SAGE Conda specification targets Python 3.9 and CUDA 11.8. The CPU smoke check is validated separately from training.

Create a dedicated environment in this repository:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

For GPU training, install a PyTorch build compatible with the machine's CUDA runtime before installing this project. Legacy MID/pickle pipelines additionally use `python -m pip install -e ".[legacy]"`.

Run experiment commands from the repository root because configurations and artifacts use relative paths. An editable install makes Python imports available; it does not download data or weights.

## Check the installation

```bash
python tools/check_repository.py
python -m sage.experiments.smoke_test --device cpu --stride 10
```

The CPU check exercises forward/backward computation and guided generation on synthetic tensors. It does not retrain the paper's models or replace external simulator evaluation.
