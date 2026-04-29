from __future__ import annotations

from benchmarks.run_benchmarks import format_results, run_all


def benchmark_main() -> None:
    print(format_results(run_all()))
