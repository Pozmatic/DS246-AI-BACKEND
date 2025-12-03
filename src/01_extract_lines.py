# src/01_extract_lines.py
import json
import csv
from pathlib import Path
import fitz  # PyMuPDF

from config import MANIFEST_PATH, LINES_DIR


def normalize_line(s: str) -> str:
    s = s.replace("\t", " ")
    s = " ".join(s.split())
    return s.strip()


def process_pdf(act_id: str, file_path: str, year: int, seq: int) -> None:
    out_dir = LINES_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{act_id}_lines.jsonl"

    doc = fitz.open(file_path)

    with out_path.open("w", encoding="utf-8") as out_f:
        for page_index in range(len(doc)):
            page = doc[page_index]

            # get raw page text
            page_text = page.get_text("text")

            # Pylance thinks this *might* be a list; guard for both
            if isinstance(page_text, str):
                lines = page_text.splitlines()
            else:
                # very defensive: if some backend returns list-of-lines
                lines = list(page_text)

            for li, raw in enumerate(lines):
                # ensure each line is a string
                if not isinstance(raw, str):
                    raw = str(raw)

                norm = normalize_line(raw)
                if not norm:
                    continue

                rec = {
                    "act_id": act_id,
                    "year": year,
                    "seq": seq,
                    "page": page_index + 1,
                    "line_index": li,
                    "raw": raw,
                    "text": norm,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"{act_id}: wrote lines to {out_path}")


def main() -> None:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            process_pdf(
                act_id=row["act_id"],
                file_path=row["file_path"],
                year=int(row["year"]),
                seq=int(row["seq"]),
            )


if __name__ == "__main__":
    main()
