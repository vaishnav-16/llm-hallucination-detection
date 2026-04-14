"""Cross-platform setup script for FastHalluCheck project."""
import subprocess
import sys
import os


def run(cmd: str, desc: str) -> None:
    print(f"\n=== {desc} ===")
    subprocess.run(cmd, shell=True, check=True)


def main():
    print("=" * 50)
    print("  FastHalluCheck Setup")
    print("=" * 50)

    # Create directories
    for d in ["data", "results", "figures", "report", "src"]:
        os.makedirs(d, exist_ok=True)
        print(f"  Created {d}/")

    # Install PyTorch with CUDA support
    run(
        f"{sys.executable} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "Installing PyTorch with CUDA 12.4 support"
    )

    # Verify CUDA
    print("\n=== Verifying GPU ===")
    verify_script = (
        "import torch; "
        "print('PyTorch', torch.__version__); "
        "print('CUDA available:', torch.cuda.is_available()); "
        "print('GPU:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('No GPU detected - will use CPU')"
    )
    subprocess.run([sys.executable, "-c", verify_script])

    # Install remaining dependencies
    run(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing other dependencies"
    )

    print("\n" + "=" * 50)
    print("  Setup complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
