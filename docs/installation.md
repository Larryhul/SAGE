# Installing SAGE

## CPU environment

Python 3.12 is the standard development environment.

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the CPU dependencies and project:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m sage.demo --output outputs/demo
```

The [demo guide](demo.md) describes the generated outputs. Run experiments from the repository root so configuration paths resolve consistently.

## Research environments

Install a CUDA-compatible PyTorch build for GPU training. The optional `legacy` dependencies support archived trajectory inputs:

```bash
python -m pip install -e ".[legacy]"
```

The original Python 3.9 / CUDA 11.8 environment is preserved in `environment-legacy.yml` for historical reproducibility. Current CPU development uses `environment.yml`.

## Verify

```bash
python tools/check_repository.py
python -m pytest -q
```
