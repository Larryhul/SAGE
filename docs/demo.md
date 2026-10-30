# SAGE CPU demonstration

```bash
python -m sage.demo --output outputs/demo
```

The CPU example refines a crossing-scene robot trajectory while keeping entity predictions fixed. It writes an animation, a final plot and an energy trace.

| Output | Contents |
| --- | --- |
| `sage_demo.gif` | Animated execution or refinement trace |
| `sage_demo.png` | Final overview |
| `summary.json` | Inputs, seed, scope and recorded trajectories |


SAGE uses the repository safety-social energy and robot-only gradient updates. Entity predictions stay fixed throughout the example. The energy trace is computed from those updates.

This is a controlled illustration. Published benchmark results and their protocols are indexed in [results](../results/README.md).

Set `--iterations` and `--seed` to control the refinement example.
