# autoinstall.py
import importlib
import subprocess
import sys

def require(pkg, asname=None):
    try:
        return importlib.import_module(asname or pkg)
    except ImportError:
        print(f"⚡ Installing missing package: {pkg}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(asname or pkg)
