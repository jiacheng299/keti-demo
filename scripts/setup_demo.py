"""Bootstrap a standalone Windows x64 / Python 3.13 demo environment."""

import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform != "win32" or struct.calcsize("P") != 8 or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("This installer targets Windows 10/11 x64. See docs/11_fresh_windows_setup.md.")
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Install Python 3.13 (64-bit), then run setup.cmd again.")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    # Honor the user's existing system proxy; never assume this laptop's port.
    for protocol in ("http", "https"):
        value = urllib.request.getproxies().get(protocol)
        if value and not environment.get(protocol.upper() + "_PROXY"):
            environment[protocol.upper() + "_PROXY"] = value
    python = ROOT / ".venv/Scripts/python.exe"
    if not python.exists():
        print("Creating isolated .venv (no system site-packages) ...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    else:
        subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:2] == (3,13), 'Existing .venv needs Python 3.13'"], check=True)
    def run(*args):
        subprocess.run([str(python), *args], cwd=ROOT, env=environment, check=True)
    print("Installing CPU PyTorch ...", flush=True)
    run("-m", "pip", "install", "torch==2.14.0", "torchvision==0.29.0", "--index-url", "https://download.pytorch.org/whl/cpu")
    print("Installing pinned dependencies ...", flush=True)
    run("-m", "pip", "install", "-r", "requirements.txt", "-c", "requirements-windows.lock.txt")
    run("-m", "pip", "check")
    print("Downloading and checking all model weights ...", flush=True)
    run("scripts/download_models.py")
    print("Running offline model and video checks ...", flush=True)
    run("scripts/check_installation.py")
    print("\nSetup completed. Run start.cmd and open http://127.0.0.1:8501", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nSetup did not complete: {error}", file=sys.stderr)
        print("Fix the reported issue and rerun setup.cmd. See docs/11_fresh_windows_setup.md.", file=sys.stderr)
        raise SystemExit(1)
