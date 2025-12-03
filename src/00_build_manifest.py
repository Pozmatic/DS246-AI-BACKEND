# src/00_build_manifest.py
import csv
from pathlib import Path
from config import ACTS_ROOT, MANIFEST_PATH

def build_manifest():
    rows = []
    for year_dir in sorted(ACTS_ROOT.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        for pdf_file in sorted(year_dir.glob("*.pdf"), key=lambda p: int(p.stem)):
            seq = int(pdf_file.stem)
            act_id = f"{year}_{seq}"
            rows.append({
                "act_id": act_id,
                "year": year,
                "seq": seq,
                "file_path": str(pdf_file),
                "act_title": "",        # filled later if you want
                "status": "raw"
            })

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["act_id", "year", "seq", "file_path", "act_title", "status"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote manifest with {len(rows)} rows to {MANIFEST_PATH}")

if __name__ == "__main__":
    build_manifest()
