# config.py
from pathlib import Path

# Adjust this to your actual path
ACTS_ROOT = Path("D:/acts_pdf")

DATA_ROOT = Path("C:/Users/vaibh/Desktop/New folder/data")
LINES_DIR = Path("C:/Users/vaibh/Desktop/New folder/data/lines")
SECTIONS_DIR = Path("C:/Users/vaibh/Desktop/New folder/data/sections")
ENRICHED_DIR = Path("C:/Users/vaibh/Desktop/New folder/data/enriched")
KG_READY_DIR = Path("C:/Users/vaibh/Desktop/New folder/data/kg_ready")
MANIFEST_PATH = Path("C:/Users/vaibh/Desktop/New folder/data/acts_manifest.csv")

for p in [DATA_ROOT, LINES_DIR, SECTIONS_DIR, ENRICHED_DIR, KG_READY_DIR]:
    p.mkdir(parents=True, exist_ok=True)
