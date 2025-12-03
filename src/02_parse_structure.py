# src/02_parse_structure.py

import json
import csv
import re
from pathlib import Path
from config import MANIFEST_PATH, LINES_DIR, SECTIONS_DIR

SECTION_PATTERNS = [
    re.compile(r"^Section\s+(\d+[A-Za-z]?)\.?\s*(.*)$"),
    re.compile(r"^(\d+[A-Za-z]?)\.\s+(.*)$"),
]

CHAPTER_PATTERN = re.compile(r"^CHAPTER\s+([IVXLC]+)\b")


def detect_section_header(text: str):
    for pat in SECTION_PATTERNS:
        m = pat.match(text)
        if m:
            section_num = m.group(1)
            heading = m.group(2).strip()
            return section_num, heading
    return None, None


def detect_chapter(text: str):
    m = CHAPTER_PATTERN.match(text)
    if m:
        return m.group(1)
    return None


def process_act(row):
    act_id = row["act_id"]
    year = int(row["year"])
    seq = int(row["seq"])

    in_path = LINES_DIR / str(year) / f"{act_id}_lines.jsonl"
    out_dir = SECTIONS_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{act_id}_sections.jsonl"

    current_chapter = None
    current_section = None
    current_heading = ""
    buffer = []
    pages = set()
    raw_lines = []

    def flush_section(outfile):
        nonlocal buffer, pages, raw_lines, current_section, current_heading
        if current_section is None:
            return

        text = "\n".join(l["text"] for l in buffer)
        if not text.strip():
            return

        record = {
            "act_id": act_id,
            "act_year": year,
            "act_seq": seq,
            "section_id": f"{act_id}-sec-{current_section}",
            "unit_type": "section",
            "hierarchy": {
                "part": None,
                "chapter": current_chapter,
                "section": current_section,
                "subsection": None,
                "clause_path": []
            },
            "citation": f"Section {current_section}",
            "heading": current_heading or None,
            "text": text,
            "pages": sorted(pages),
            "raw_lines": raw_lines,
        }

        outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

        buffer = []
        pages = set()
        raw_lines = []

    # -------- RUN THE PARSER --------
    with in_path.open(encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            t = rec["text"]

            chap = detect_chapter(t)
            if chap:
                current_chapter = chap
                continue

            sec_num, heading = detect_section_header(t)
            if sec_num:
                flush_section(f_out)
                current_section = sec_num
                current_heading = heading
                buffer = []
                pages = set()
                raw_lines = []
                continue

            if current_section is None:
                continue  # ignore preamble for now

            buffer.append(rec)
            pages.add(rec["page"])
            raw_lines.append({
                "page": rec["page"],
                "line_index": rec["line_index"],
                "text": rec["text"]
            })

        flush_section(f_out)

    print(f"{act_id}: sections written to {out_path}")


def main():
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            process_act(row)


if __name__ == "__main__":
    main()
