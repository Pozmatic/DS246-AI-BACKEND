# src/04_enrich_llm_local.py
"""
Use local Ollama (qwen2.5:7b-instruct) to enrich sections from `enriched`
and write final `kg_ready` files.

Input:  data/enriched/<year>/<act_id>_enriched.jsonl
Output: data/kg_ready/<year>/<act_id>_kg.jsonl
"""

import json
import csv
import re
import os
import time
from pathlib import Path

import ollama  # pip install ollama

from config import MANIFEST_PATH, ENRICHED_DIR, KG_READY_DIR


# ---------- LLM CONFIG ----------

MODEL = "qwen2.5:7b-instruct"  # make sure you ran: `ollama pull qwen2.5:7b-instruct`
BATCH_SIZE = 5  # number of sections per LLM call


# ---------- SENTENCE SELECTION ----------

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")

# you can tune this list to adjust how many sections trigger LLM calls
KEYWORDS = [
    " shall ",
    " shall be bound ",
    " may ",
    " punishable ",
    " liable ",
    " offence ",
    " offense ",
]


def find_candidate_sentences(text: str, max_sentences: int = 8):
    """Return a list of 'interesting' sentences for the LLM."""
    sentences = SENTENCE_SPLIT_RE.split(text)
    cand = []
    for s in sentences:
        ls = " " + s.lower() + " "
        if any(k in ls for k in KEYWORDS):
            s_clean = s.strip()
            if s_clean:
                cand.append(s_clean)
        if len(cand) >= max_sentences:
            break
    return cand


# ---------- PROMPT TEMPLATE ----------

PROMPT_TEMPLATE_BATCH = """
You are a legal information extraction system for Indian statutes.

For EACH SECTION below, extract the following information:

- roles: distinct entities who act in the law (e.g. "District Magistrate", "keeper of a sarai").
- obligations: duties that MUST be performed.
- powers: actions that an authority MAY do or has power to do.
- penalties: legal consequences like imprisonment, fine, etc.
- rights: any explicit rights granted to persons.

Return STRICT JSON **array**. Each element corresponds to ONE section and must have schema:

{{
  "section_index": <integer index exactly as given>,
  "roles": ["role1", "role2", ...],
  "obligations": [
    {{
      "actor": "string",
      "action": "string",
      "conditions": "string or null",
      "source_span": "exact sentence or phrase"
    }}
  ],
  "powers": [
    {{
      "actor": "string",
      "action": "string",
      "conditions": "string or null",
      "source_span": "exact sentence or phrase"
    }}
  ],
  "penalties": [
    {{
      "subject": "string",
      "description": "string",
      "imprisonment": "string or null",
      "fine_amount": "number or null",
      "source_span": "exact sentence or phrase"
    }}
  ],
  "rights": [
    {{
      "holder": "string",
      "description": "string",
      "conditions": "string or null",
      "source_span": "exact sentence or phrase"
    }}
  ]
}}

If something is not present for a section, use empty lists.

SECTIONS (each starts with 'SECTION <index>'):

{sections_block}
"""


# ---------- LLM CALL (OLLAMA) ----------

def call_llm_batch(batch_inputs):
    """
    batch_inputs: list of dicts:
        { "section_index": int, "sentences": [str, ...] }

    Returns: dict mapping section_index -> result dict
    """
    if not batch_inputs:
        return {}

    # Build the sections block
    blocks = []
    for entry in batch_inputs:
        idx = entry["section_index"]
        sent_list = entry["sentences"]
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sent_list))
        block = f"SECTION {idx}:\n{numbered}"
        blocks.append(block)

    sections_block = "\n\n".join(blocks)
    prompt = PROMPT_TEMPLATE_BATCH.format(sections_block=sections_block)

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a structured legal information extractor. "
                        "You MUST respond with valid JSON only, matching the requested schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        text = response["message"]["content"]

        if not text or not isinstance(text, str):
            return {}

        # try JSON parse
        try:
            parsed = json.loads(text)
        except Exception as e:
            print("LLM JSON parse error:", type(e).__name__, e)
            # if JSON can't be parsed, skip this batch
            return {}

        if not isinstance(parsed, list):
            parsed = [parsed]

        out = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            idx = item.get("section_index")
            if idx is None:
                continue

            out[idx] = {
                "roles": item.get("roles") or [],
                "obligations": item.get("obligations") or [],
                "powers": item.get("powers") or [],
                "penalties": item.get("penalties") or [],
                "rights": item.get("rights") or [],
            }

        return out

    except Exception as e:
        print("Ollama LLM batch error:", type(e).__name__, e)
        time.sleep(0.2)
        return {}


# ---------- MERGE LLM SEMANTICS INTO SECTION ----------

def merge_semantics(sec, llm_result):
    # roles: dedupe strings
    existing_roles = set(sec.get("roles") or [])
    for r in llm_result.get("roles") or []:
        if isinstance(r, str):
            existing_roles.add(r.strip())
    sec["roles"] = sorted(existing_roles)

    # obligations / powers / penalties / rights: append lists of dicts
    for key in ["obligations", "powers", "penalties", "rights"]:
        existing = sec.get(key) or []
        new_vals = llm_result.get(key) or []
        new_clean = [v for v in new_vals if isinstance(v, dict)]
        sec[key] = existing + new_clean

    meta = sec.get("processing_meta", {}) or {}
    used = any(llm_result.get(k) for k in ["roles", "obligations", "powers", "penalties", "rights"])
    meta["llm_used"] = bool(used)
    meta["llm_model"] = MODEL if used else None
    sec["processing_meta"] = meta
    return sec


# ---------- PER-ACT PROCESSING ----------

def process_act(row):
    act_id = row["act_id"]
    year = int(row["year"])

    in_path = ENRICHED_DIR / str(year) / f"{act_id}_enriched.jsonl"
    out_dir = KG_READY_DIR / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{act_id}_kg.jsonl"

    if not in_path.exists():
        print(f"{act_id}: no enriched file at {in_path}, skipping")
        return

    # Skip if we already have non-empty kg_ready (from earlier cloud runs, etc.)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"{act_id}: kg_ready already exists, skipping")
        return

    # Load all sections
    sections = []
    with in_path.open(encoding="utf-8") as f_in:
        for line in f_in:
            if line.strip():
                sections.append(json.loads(line))

    if not sections:
        print(f"{act_id}: enriched file empty, nothing to do")
        return

    # Build LLM inputs for sections that look interesting
    llm_inputs = []
    for idx, sec in enumerate(sections):
        candidates = find_candidate_sentences(sec["text"])
        if candidates:
            llm_inputs.append({"section_index": idx, "sentences": candidates})
        else:
            # mark as no-LLM
            meta = sec.get("processing_meta", {}) or {}
            meta["llm_used"] = False
            meta["llm_model"] = None
            sec["processing_meta"] = meta

    # Call LLM in batches
    results_by_index = {}
    for i in range(0, len(llm_inputs), BATCH_SIZE):
        batch = llm_inputs[i : i + BATCH_SIZE]
        batch_result = call_llm_batch(batch)
        results_by_index.update(batch_result)

    # Merge back into sections
    for idx, sec in enumerate(sections):
        if idx in results_by_index:
            sec = merge_semantics(sec, results_by_index[idx])
        sections[idx] = sec

    # Write KG-ready file
    with out_path.open("w", encoding="utf-8") as f_out:
        for sec in sections:
            f_out.write(json.dumps(sec, ensure_ascii=False) + "\n")

    print(f"{act_id}: KG-ready -> {out_path}")


def main():
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["year"]) < 1889:
                continue
            process_act(row)


if __name__ == "__main__":
    main()
