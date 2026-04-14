"""Cross-platform pipeline runner for FastHalluCheck project.

Routes to the new layered CLI. For the original flat-script pipeline,
see legacy/run_all.py.
"""
import subprocess
import sys


def main():
    print("=" * 50)
    print("  FastHalluCheck — Full Pipeline")
    print("=" * 50)
    print("  Routing to new CLI layer...\n")
    subprocess.run([sys.executable, "-m", "src.cli.commands", "run"])


if __name__ == "__main__":
    main()
