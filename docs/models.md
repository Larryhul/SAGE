# SAGE models

The v0.2.0 model archive contains six original goal-conditioned research checkpoints: ETH, HOTEL, UNIV, ZARA1, ZARA2 and SDD. Their SHA-256 values match the retained research inventory.

```bash
python tools/download_models.py
```

The command verifies the archive and each checkpoint, then writes them to `data/external/checkpoints/`.

| Checkpoint | Bytes |
| --- | --- |
| `eth_sage_epoch100.pt` | 2,157,979 |
| `hotel_sage_epoch100.pt` | 2,158,105 |
| `univ_sage_epoch100.pt` | 2,158,042 |
| `zara1_sage_epoch100.pt` | 2,158,105 |
| `zara2_sage_epoch100.pt` | 2,158,105 |
| `sdd_sage_epoch80.pt` | 2,158,108 |

[Direct archive download](https://github.com/Larryhul/SAGE/releases/download/v0.2.0/sage-research-checkpoints-v0.2.0.zip) · [Checksums](../provenance/model_release.json)

Each checkpoint includes its model state and saved configuration. The release check loads all six states strictly and performs finite CPU generation. Dataset-level evaluation uses the matching preparation and role-labeling protocol in the [experiment guide](reproduction.md).

For an ETH checkpoint, after preparing the ETH test inputs:

```bash
python -m sage.experiments.eval_real_sage --config configs/sage.yaml --dataset eth --split test --checkpoint data/external/checkpoints/eth_sage_epoch100.pt --device cpu
```

The archive includes the repository MIT license. Dataset inputs follow their provider terms. The full retained model-selection inventory remains in [checkpoints.csv](../provenance/checkpoints.csv).
