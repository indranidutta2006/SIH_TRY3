# Add project root to PYTHONPATH for test imports
import sys
import os

# Ensure the repository root directory is in sys.path
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
