# Verification

The release checks cover installation, model serialization, the public CPU regression suite and the [demonstration](demo.md). CI runs on Linux and Windows.

Local v0.2.0 checks on Windows passed all five tests under Python 3.12 and the legacy Python 3.9 environment. All six released research checkpoints passed strict state loading and finite CPU generation. The installed wheel also ran the animated demo outside the source directory.

```bash
python tools/check_repository.py
python -m pytest -q
python -m sage.demo --output outputs/demo
```

The asset check verifies retained file hashes and local documentation links. Test output reports its passed and skipped cases. Tests requiring CUDA or archived formal-run assets carry explicit prerequisites.

The release process also builds a wheel, installs it in a separate directory and checks imports and the demo outside the source tree. Original research checkpoints are checked against their retained SHA-256 and loaded for a finite CPU generation check.

These are CPU software checks. Full benchmark reruns and physical-robot validation are separate research workflows.
