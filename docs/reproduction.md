# Reproducing SAGE

## 1. Verify the data-free path

```bash
python -m sage.experiments.smoke_test --device cpu --stride 10
python -m sage.experiments.train_sage --device cpu --epochs 1 --save experiments/example/sage.pt
```

These commands exercise synthetic tensors and a tiny training run. They are installation checks rather than benchmark estimates.

## 2. Prepare benchmark inputs

Obtain ETH/UCY or SDD from the providers listed in [assets](assets.md). The MID adapter expects processed pickles such as `models/processed_data_noise/eth_test.pkl`. Keep the original train/validation/test separation.

Check an ETH/UCY input before training:

```bash
python -m sage.experiments.check_real_data --config configs/sage.yaml --dataset eth --split test --limit-samples 8
```

Legacy pickles require the compatible MID modules and optional legacy dependencies. The source repository does not include those input files.

## 3. Train and evaluate

```bash
python -m sage.experiments.train_real_sage --config configs/sage.yaml --dataset eth --epochs 100 --device cuda
python -m sage.experiments.eval_real_sage --config configs/sage.yaml --dataset eth --split test --checkpoint experiments/sage/eth_ucy/goal/eth_sage_epoch100.pt --device cuda
```

Use `--help` on each module for overrides. The wrapper `scripts/eval_sage_real.sh` uses the same canonical checkpoint layout. Shell wrappers target Bash; the Python module commands also work in PowerShell.

For controlled synthetic navigation, use `train_synthetic_sage`, `eval_synthetic_sage` and `configs/sage_synthetic.yaml`. For SDD, use `train_sdd_sage`, `eval_sdd_sage` and the SDD configurations. The dataset-specific entry points preserve their distinct labeling and coordinate conventions.

## 4. Match the protocol

Record checkpoint SHA-256, split, sampling count, diffusion stride, guidance weights, role assignment and random seeds. ADE/FDE can use best-of-K sampling while safety diagnostics aggregate samples; preserve the reported aggregation rule when comparing methods.

ETH/UCY contains pedestrian labels. The adapter treats a target trajectory as a controllable robot proxy and assigns entity pseudo-roles from speed. Such tests do not establish recognition of heterogeneous semantic categories. Synthetic experiments and SDD semantic annotations address separate settings.

Results and important intermediate records are indexed in [results](../results/README.md). A retained CSV is not automatically a final manuscript table; model-selection and prior-control records keep their own roles.

## Verification boundary

The local source preparation checked the SAGE synthetic forward/backward/guided-generation path. The standalone repository checks are recorded in [verification](verification.md). Benchmark retraining and physical robot deployment are separate validation steps.
