"""
Check if PyTorch can use your NVIDIA GPU.
Run this after installing PyTorch to verify CUDA works.
"""
import subprocess
import sys

def main():
    print("=== NVIDIA GPU (nvidia-smi) ===")
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except FileNotFoundError:
        print("nvidia-smi not found. Install NVIDIA drivers.")
        return 1

    print("\n=== PyTorch CUDA ===")
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Install with:")
        print('  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128')
        return 1

    cuda_ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_ok}")
    if cuda_ok:
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
    else:
        print("\nPyTorch is using the CPU-only build.")
        print("To install the CUDA build, run:")
        print("  pip uninstall torch torchaudio -y")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print("\nIf cu128 fails (e.g. no Windows wheel), try cu126 or cu121:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
