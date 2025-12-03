"""
05_stream_to_neo4j.py

Builds the knowledge graph in Neo4j from:
- data/kg_ready/<year>/<act_id>_kg.jsonl  (preferred)
- data/enriched/<year>/<act_id>_enriched.jsonl  (fallback if kg_ready missing/empty)
"""

import csv
import json
import os
from pathlib import Path

from neo4j import GraphDatabase

from config import MANIFEST_PATH, ENRICHED_DIR, KG_READY_DIR


# ---------- Neo4j connection settings ----------

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ds246@IISc")   # change or set env vars


# ---------- Cypher write logic for a single section ----------

def merge_section_tx(tx, sec: dict):
    """
    Upsert Act, Section and semantic nodes/edges for one section dict.
    """

    act_id = sec["act_id"]
    act_year = sec.get("act_year")
    section_id = sec["section_id"]
    citation = sec.get("citation")
    heading = sec.get("heading")
    text = sec.get("text")

    hierarchy = sec.get("hierarchy") or {}
    chapter = hierarchy.get("chapter")
    section_no = hierarchy.get("section")
    pages = sec.get("pages") or []

    meta = sec.get("processing_meta") or {}
    has_llm = bool(meta.get("llm_used"))

    # 1) Act and Section
    tx.run(
        """
        MERGE (a:Act {id: $act_id})
        ON CREATE SET a.year = $act_year
        ON MATCH  SET a.year = coalesce(a.year, $act_year)

        MERGE (s:Section {id: $section_id})
        SET s.citation  = $citation,
            s.heading   = $heading,
            s.text      = $text,
            s.chapter   = $chapter,
            s.sectionNo = $section_no,
            s.pages     = $pages,
            s.has_llm   = $has_llm

        MERGE (a)-[:HAS_SECTION]->(s)
        """,
        act_id=act_id,
        act_year=act_year,
        section_id=section_id,
        citation=citation,
        heading=heading,
        text=text,
        chapter=chapter,
        section_no=section_no,
        pages=pages,
        has_llm=has_llm,
    )

    # 2) Defined terms
    for term in sec.get("defined_terms") or []:
        name = (term.get("term") or "").strip()
        if not name:
            continue
        term_id = f"{act_id}|{name.lower()}"

        tx.run(
            """
            MATCH (s:Section {id: $section_id})
            MERGE (t:DefinedTerm {id: $term_id})
            ON CREATE SET t.name = $name, t.act_id = $act_id
            MERGE (s)-[:DEFINES]->(t)
            """,
            section_id=section_id,
            term_id=term_id,
            name=name,
            act_id=act_id,
        )

    # 3) Roles
    roles = sec.get("roles") or []
    for role_name in roles:
        if not isinstance(role_name, str):
            continue
        clean = role_name.strip()
        if not clean:
            continue
        role_id = clean.lower()

        tx.run(
            """
            MATCH (s:Section {id: $section_id})
            MERGE (r:Role {id: $role_id})
            ON CREATE SET r.name = $role_name
            MERGE (s)-[:MENTIONS_ROLE]->(r)
            """,
            section_id=section_id,
            role_id=role_id,
            role_name=clean,
        )

    # 4) Obligations
    for idx, ob in enumerate(sec.get("obligations") or []):
        if not isinstance(ob, dict):
            continue

        actor = (ob.get("actor") or "").strip()
        action = ob.get("action")
        conditions = ob.get("conditions")
        source_span = ob.get("source_span")

        obl_id = f"{section_id}|ob|{idx}"

        tx.run(
            """
            MATCH (s:Section {id: $section_id})
            MERGE (o:Obligation {id: $obl_id})
            SET o.action      = $action,
                o.conditions  = $conditions,
                o.source_span = $source_span
            MERGE (s)-[:IMPOSES_OBLIGATION]->(o)
            """,
            section_id=section_id,
            obl_id=obl_id,
            action=action,
            conditions=conditions,
            source_span=source_span,
        )

        if actor:
            role_id = actor.lower()
            tx.run(
                """
                MERGE (r:Role {id: $role_id})
                ON CREATE SET r.name = $actor
                MERGE (o:Obligation {id: $obl_id})-[:OBLIGATION_ON]->(r)
                """,
                role_id=role_id,
                actor=actor,
                obl_id=obl_id,
            )

    # 5) Penalties
    for idx, pen in enumerate(sec.get("penalties") or []):
        if not isinstance(pen, dict):
            continue

        subject = (pen.get("subject") or "").strip()
        description = pen.get("description")
        imprisonment = pen.get("imprisonment")
        fine_amount = pen.get("fine_amount")
        source_span = pen.get("source_span")

        fine_str = None
        if fine_amount is not None:
            fine_str = str(fine_amount)

        pen_id = f"{section_id}|pen|{idx}"

        tx.run(
            """
            MATCH (s:Section {id: $section_id})
            MERGE (p:Penalty {id: $pen_id})
            SET p.description  = $description,
                p.imprisonment = $imprisonment,
                p.fine_amount  = $fine_amount,
                p.source_span  = $source_span
            MERGE (s)-[:PRESCRIBES_PENALTY]->(p)
            """,
            section_id=section_id,
            pen_id=pen_id,
            description=description,
            imprisonment=imprisonment,
            fine_amount=fine_str,
            source_span=source_span,
        )

        if subject:
            role_id = subject.lower()
            tx.run(
                """
                MERGE (r:Role {id: $role_id})
                ON CREATE SET r.name = $subject
                MERGE (p:Penalty {id: $pen_id})-[:APPLIES_TO]->(r)
                """,
                role_id=role_id,
                subject=subject,
                pen_id=pen_id,
            )


# ---------- choose kg_ready vs enriched ----------

def choose_source_file(year: int, act_id: str) -> Path | None:
    """
    Prefer kg_ready if non-empty; otherwise enriched if non-empty; else None.
    """
    kg_path = KG_READY_DIR / str(year) / f"{act_id}_kg.jsonl"
    if kg_path.exists() and kg_path.stat().st_size > 0:
        return kg_path

    enr_path = ENRICHED_DIR / str(year) / f"{act_id}_enriched.jsonl"
    if enr_path.exists() and enr_path.stat().st_size > 0:
        return enr_path

    return None


def process_act_file(session, year: int, act_id: str, file_path: Path):
    count = 0
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sec = json.loads(line)
            session.execute_write(merge_section_tx, sec)
            count += 1
    print(f"{act_id}: ingested {count} sections from {file_path}")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver, driver.session() as session, MANIFEST_PATH.open(
        newline="", encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            act_id = row["act_id"]

            src = choose_source_file(year, act_id)
            if src is None:
                print(f"{act_id}: no non-empty kg_ready or enriched file, skipping")
                continue

            process_act_file(session, year, act_id, src)


if __name__ == "__main__":
    main()
