import sys
from pathlib import Path

# independent_horizon.py lives beside the tests and is imported by name.
sys.path.insert(0, str(Path(__file__).resolve().parent))
