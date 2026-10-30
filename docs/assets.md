# SAGE assets

## Available in the repository

Source code, indexed configurations, selected numerical results, method graphics and a runnable CPU demonstration.

## Model downloads

Original goal-conditioned research checkpoints for ETH/UCY and SDD.

Use [the model guide](models.md) to download and verify the release archive. File identities are recorded in [model_release.json](../provenance/model_release.json).

## Dataset and simulator inputs

| Input | Source | Local placement |
| --- | --- | --- |
| ETH/UCY trajectories | [MID preprocessing](https://github.com/Gutianpei/MID), with original datasets from their providers | `models/processed_data_noise/<dataset>_<split>.pkl` |
| Stanford Drone Dataset | [Original dataset](https://cvgl.stanford.edu/projects/uav_data/) | Annotation root supplied to SDD entry points |
| Synthetic scenes | `sage.data.synthetic_navigation` | Generated at runtime |

The [experiment guide](reproduction.md) records split, role labels and aggregation conventions. The [retained checkpoint inventory](../provenance/checkpoints.csv) identifies additional model-selection records.

Dataset files, trained outputs and full simulator trees stay outside source control. Copied source identities are recorded in [provenance](../provenance/README.md).
