# src/03_enrich_rule_based.py

import json
import csv
import re
from pathlib import Path
from config import MANIFEST_PATH, SECTIONS_DIR, ENRICHED_DIR

SECTION_CITE_RE = re.compile(r"\bsections?\s+\d+[A-Za-z]*(?:\s*,\s*\d+[A-Za-z]*)*")
ACT_CITE_RE = re.compile(r"\b([A-Z][A-Za-z\s]+ Act,\s*\d{4})\b")
DEFINITION_RE = re.compile(r"\"([^\"]+)\"\s+(means|includes)\s+", re.IGNORECASE)
AMENDMENT_RE = re.compile(r"is hereby (repealed|substituted|omitted)", re.IGNORECASE)
NOTWITH_RE = re.compile(r"notwithstanding anything contained in", re.IGNORECASE)


def enrich(sec: dict) -> dict:
    text = sec["text"]

    citations = []
    for m in SECTION_CITE_RE.finditer(text):
        citations.append({"kind": "section", "raw": m.group(0), "normalized": {}})
    for m in ACT_CITE_RE.finditer(text):
        citations.append({
            "kind": "act",
            "raw": m.group(1),
            "normalized": {"title": m.group(1)}
        })

    defined_terms = [{
        "term": m.group(1),
        "definition_span": m.group(0)
    } for m in DEFINITION_RE.finditer(text)]

    amendments = [{
        "type": m.group(1).lower(),
        "raw": m.group(0),
        "target_text": None
    } for m in AMENDMENT_RE.finditer(text)]

    precedence_clauses = [{
        "kind": "notwithstanding",
        "raw": m.group(0)
    } for m in NOTWITH_RE.finditer(text)]

    sec["citations"] = citations
    sec["defined_terms"] = defined_terms
    sec["amendments"] = amendments
    sec["precedence_clauses"] = precedence_clauses

    sec["roles"] = sec.get("roles", [])
    sec["obligations"] = sec.get("obligations", [])
    sec["powers"] = sec.get("powers", [])
    sec["penalties"] = sec.get("penalties", [])
    sec["rights"] = sec.get("rights", [])

    meta = sec.get("processing_meta", {}) or {}
    meta.setdefault("rule_based_flags", [])
    if citations:
        meta["rule_based_flags"].append("has_citations")
    if amendments:
        meta["rule_based_flags"].append("has_amendments")
    sec["processing_meta"] = meta

    return sec


def process_act(row: dict) -> None:
    act_id = row["act_id"]
    year = int(row["year"])

    in_path = SECTIONS_DIR / str(year) / f"{act_id}_sections.jsonl"
    out_dir = ENRICHED_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{act_id}_enriched.jsonl"

    with in_path.open(encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            sec = json.loads(line)
            sec = enrich(sec)
            f_out.write(json.dumps(sec, ensure_ascii=False) + "\n")

    print(f"{act_id}: enriched -> {out_path}")


def main() -> None:
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            process_act(row)


if __name__ == "__main__":
    main()
