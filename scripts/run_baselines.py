#!/usr/bin/env python
"""Run all ablation modes and produce cross-mode comparison.

Usage:
    python scripts/run_baselines.py "Stocks with high momentum" --num-runs 3
    python scripts/run_baselines.py "Stocks with high momentum" --num-runs 1 --modes random-gp single-agent
"""

import argparse
import os
import subprocess
import sys
import time

MODES = ["random-gp", "single-agent", "single-agent-gp", "debate-only", "full"]


def _run_live(cmd: list[str]) -> int:
    """Run a command with live stdout/stderr streaming."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )
    proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all ablation baselines")
    parser.add_argument("idea", nargs="+", help="Trading idea in natural language")
    parser.add_argument("--num-runs", type=int, default=3, help="Runs per mode (default: 3)")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=MODES,
        metavar="MODE",
        help=f"Which modes to run (default: all). Choices: {MODES}",
    )
    args = parser.parse_args()

    idea = " ".join(args.idea)
    print(f"Trading idea: {idea}", flush=True)
    print(f"Modes: {args.modes}", flush=True)
    print(f"Runs per mode: {args.num_runs}", flush=True)
    print(flush=True)

    results = {}
    for mode in args.modes:
        print(f"\n{'='*60}", flush=True)
        print(f"  Running: {mode} ({args.num_runs} run{'s' if args.num_runs > 1 else ''})", flush=True)
        print(f"{'='*60}\n", flush=True)

        t0 = time.time()
        cmd = [
            sys.executable, "-u", "-m", "alpha_gpt.main",
            idea,
            "--mode", mode,
            "--num-runs", str(args.num_runs),
        ]
        rc = _run_live(cmd)
        elapsed = time.time() - t0

        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        results[mode] = f"{status} in {elapsed:.0f}s"
        print(f"\n[{mode}] {status} in {elapsed:.0f}s", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("  Run Summary", flush=True)
    print(f"{'='*60}", flush=True)
    for mode, status in results.items():
        print(f"  {mode:<20} {status}", flush=True)

    # Compare
    print(f"\n{'='*60}", flush=True)
    print("  Running comparison...", flush=True)
    print(f"{'='*60}\n", flush=True)
    _run_live([sys.executable, "-u", "-m", "alpha_gpt.main", "--compare"])


if __name__ == "__main__":
    main()
