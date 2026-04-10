import subprocess
import time


def run_bench(cmd):
    try:
        start = time.perf_counter()
        result = subprocess.run(  # noqa: S602 — benchmark cmds are hardcoded
            cmd, shell=True, capture_output=True, text=True, check=True
        )
        elapsed = time.perf_counter() - start
        return result.stdout.strip(), elapsed
    except Exception as e:
        return f"Error: {e}", 0


def main():
    print("SCPN Phase Orchestrator — Optimization Summary")
    print("=" * 50)

    benchmarks = [
        ("UPDE (Dense, N=1000)", "python benchmarks/scaling_benchmark.py"),
        ("Sparse UPDE (N=1000, d=0.01)", "python benchmarks/sparse_benchmark.py"),
        ("Stuart-Landau (N=1000)", "python benchmarks/sl_benchmark.py"),
        ("Simplicial (N=1000)", "python benchmarks/simplicial_benchmark.py"),
        ("Hypergraph (N=1000)", "python benchmarks/hypergraph_benchmark.py"),
        ("Inertial (N=1000)", "python benchmarks/inertial_benchmark.py"),
        ("Delayed (N=1000, tau=10)", "python benchmarks/delayed_benchmark.py"),
        ("Swarmalator (N=500)", "python benchmarks/swarmalator_benchmark.py"),
        ("Recurrence (T=2000, d=10)", "python benchmarks/recurrence_benchmark.py"),
    ]

    for name, cmd in benchmarks:
        print(f"Running {name}...")
        output, _ = run_bench(f"export PYTHONPATH=$PYTHONPATH:$(pwd)/src && {cmd}")
        print(output)
        print("-" * 30)


if __name__ == "__main__":
    main()
