from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

# Constants
DIRECT = "direct"
ONE_TOUCH = "1_touch"
TWO_TOUCH = "2_touch"
VALID_TYPES = {"launch", "hub", "hybrid"}
DAY_TYPES = {"offpeak", "peak"}
