# Benchmarks

Run:

```bash
python -m ai_lab.benchmarks.run_benchmarks
```

Or, after installing the package:

```bash
ai-lab-benchmark
```

If your Python scripts directory is not on `PATH`, use:

```bash
python -m ai_lab.benchmarks.run_benchmarks
```

The benchmark table is intentionally small and fast. It is designed to verify that the
core implementation can solve representative tasks:

- XOR classification
- linear regression
- synthetic multiclass classification
- optional sklearn comparison when installed

The goal is reproducibility and signal, not leaderboard performance.
