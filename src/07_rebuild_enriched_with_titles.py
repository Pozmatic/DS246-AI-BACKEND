# src/07_rebuild_enriched_with_titles.py

import re
import csv
import json
from pathlib import Path

import pdfplumber  # pip install pdfplumber

from config import MANIFEST_PATH, ENRICHED_DIR, ACTS_ROOT

NEW_ENRICHED_DIR = Path("data/new_enriched")
NEW_ENRICHED_DIR.mkdir(parents=True, exist_ok=True)

TITLE_REGEX = re.compile(
    r".*\b(Act|Ordinance|Code)\b.*\b(18|19|20)\d{2}\b",
    re.IGNORECASE,
)


def extract_title(pdf_path: Path) -> str | None:
    """Extract a likely act title from the first 1–2 pages of the PDF."""
    if not pdf_path.exists():
        print("PDF not found:", pdf_path)
        return None

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            num_pages = len(pdf.pages)
            pages_to_read = min(2, num_pages)

            for i in range(pages_to_read):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

                for ln in lines[:10]:
                    if TITLE_REGEX.match(ln):
                        return ln
    except Exception as e:
        print("Error extracting title from", pdf_path, "->", type(e).__name__, e)

    return None


def find_enriched_file(year: int, act_id: str) -> Path | None:
    """
    Old enriched files are named like:
        data/enriched/<year>/<year>_<seq>_enriched.jsonl
    where act_id = "<year>_<seq>".
    """
    year_dir = ENRICHED_DIR / str(year)
    if not year_dir.exists():
        return None

    parts = act_id.split("_", 1)
    seq = parts[1] if len(parts) > 1 else act_id

    # main pattern: year_seq_enriched.jsonl
    candidate = year_dir / f"{year}_{seq}_enriched.jsonl"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate

    # fallback: anything containing "_<seq>_enriched"
    for cand in year_dir.glob(f"*_{seq}_enriched*.json*"):
        if cand.stat().st_size > 0:
            return cand

    return None


def process_act(year: int, act_id: str):
    # PDF path: acts_pdf/<year>/<seq>.pdf
    parts = act_id.split("_", 1)
    seq = parts[1] if len(parts) > 1 else act_id
    pdf_path = ACTS_ROOT / str(year) / f"{seq}.pdf"

    old_enriched = find_enriched_file(year, act_id)
    if old_enriched is None:
        print(f"Skipping {act_id}: could not find non-empty enriched file in {ENRICHED_DIR/str(year)}")
        return

    new_dir = NEW_ENRICHED_DIR / str(year)
    new_dir.mkdir(parents=True, exist_ok=True)
    new_file = new_dir / f"{act_id}_enriched.jsonl"

    title = extract_title(pdf_path) or ""

    with old_enriched.open(encoding="utf-8") as f_in, \
         new_file.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

        # add act_title to each section
            section = json.loads(line)
            section["act_title"] = title
            f_out.write(json.dumps(section, ensure_ascii=False) + "\n")

    print(f"{act_id}: used {old_enriched.name}, wrote new_enriched with title -> {title}")


def main():
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            act_id = row["act_id"]
            process_act(year, act_id)


if __name__ == "__main__":
    main()
