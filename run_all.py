# -*- coding: utf-8 -*-
"""Regenerate every number, table and figure of the manuscript.

    python run_all.py            # everything (about 30-60 minutes on one core)
    python run_all.py --quick    # only the verification of the theory and the round-off check

Order: experiments (results/*.json) -> figures -> tables -> CSV export ->
verification of the theoretical statements -> round-off comparison.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    t0 = time.time()
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"   done in {time.time() - t0:.1f} s", flush=True)


def main():
    quick = "--quick" in sys.argv[1:]
    if not quick:
        run("experiments.py", "1", "2", "3", "4", "5", "6", "fig")
        run("make_tables.py")
        run("export_csv.py")
    run("verify_theory.py")
    run("check_roundoff.py")


if __name__ == "__main__":
    main()
