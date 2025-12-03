"""
06_enrich_graph_full.py

Post-processing enrichment for the legal KG in Neo4j:

1. Add CITES relationships between sections based on citations in kg_ready/enriched.
2. Link DefinedTerm nodes with SAME_TERM_AS when they share the same name.
3. Link Role -> Act via APPEARS_IN_ACT when a role is mentioned in an act's sections.
4. Build a role co-occurrence network: CO_OCCURS_WITH with a `count` property.
5. Compute a simple severity_score for each Section based on its penalties.

Run AFTER 05_stream_to_neo4j.py has populated the base graph.
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
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ds246@IISc")  # change or set env vars


# ---------- helper: choose kg_ready or enriched ----------

def choose_source_file(year: int, act_id: str) -> Path | None:
    """Prefer non-empty kg_ready; else non-empty enriched; else None."""
    kg_path = KG_READY_DIR / str(year) / f"{act_id}_kg.jsonl"
    if kg_path.exists() and kg_path.stat().st_size > 0:
        return kg_path

    enr_path = ENRICHED_DIR / str(year) / f"{act_id}_enriched.jsonl"
    if enr_path.exists() and enr_path.stat().st_size > 0:
        return enr_path

    return None


# ---------- citation extraction heuristics ----------

def iter_citations_from_section(sec: dict):
    """
    Yield citation dicts from a section.

    Output format (all strings):
      {
        "source_section_id": "...",
        # either:
        "target_section_id": "...",
        # or:
        "target_act_id": "...",
        "target_section_no": "..."
      }

    This is defensive and supports several possible field names.
    You can tweak the key names here if your JSON uses a different schema.
    """
    source_section_id = sec.get("section_id")
    if not source_section_id:
        return

    act_id = sec.get("act_id")

    citations = (
        sec.get("citations")
        or sec.get("references")
        or sec.get("cross_refs")
        or sec.get("cross_references")
        or []
    )
    if not isinstance(citations, list):
        return

    for c in citations:
        # Case 1: citation is a simple string, e.g. "1860_1:302" or "302"
        if isinstance(c, str):
            text = c.strip()
            if not text:
                continue
            if ":" in text:
                t_act, t_sec = text.split(":", 1)
                yield {
                    "source_section_id": source_section_id,
                    "target_act_id": t_act.strip(),
                    "target_section_no": t_sec.strip(),
                }
            else:
                # assume same act, different section
                yield {
                    "source_section_id": source_section_id,
                    "target_act_id": act_id,
                    "target_section_no": text,
                }
            continue

        # Case 2: citation is a dict
        if not isinstance(c, dict):
            continue

        t_sec_id = (
            c.get("target_section_id")
            or c.get("section_id")
            or c.get("target_section")
        )
        t_act_id = (
            c.get("target_act_id")
            or c.get("act_id")
            or c.get("target_act")
            or c.get("act")
        )
        t_sec_no = (
            c.get("target_section_no")
            or c.get("section_no")
            or c.get("section")
        )

        if t_sec_id:
            yield {
                "source_section_id": source_section_id,
                "target_section_id": str(t_sec_id).strip(),
            }
            continue

        if t_act_id and t_sec_no:
            yield {
                "source_section_id": source_section_id,
                "target_act_id": str(t_act_id).strip(),
                "target_section_no": str(t_sec_no).strip(),
            }
            continue

        if t_sec_no and act_id:
            yield {
                "source_section_id": source_section_id,
                "target_act_id": act_id,
                "target_section_no": str(t_sec_no).strip(),
            }


# ---------- Neo4j write helpers for CITES ----------

def merge_cites_by_id_tx(tx, source_section_id: str, target_section_id: str):
    tx.run(
        """
        MATCH (s1:Section {id: $source_id})
        MATCH (s2:Section {id: $target_id})
        MERGE (s1)-[:CITES]->(s2)
        """,
        source_id=source_section_id,
        target_id=target_section_id,
    )


def merge_cites_by_act_and_no_tx(
    tx, source_section_id: str, target_act_id: str, target_section_no: str
):
    tx.run(
        """
        MATCH (s1:Section {id: $source_id})
        MATCH (a2:Act {id: $target_act_id})-[:HAS_SECTION]->(s2:Section)
        WHERE s2.sectionNo = $target_section_no
        MERGE (s1)-[:CITES]->(s2)
        """,
        source_id=source_section_id,
        target_act_id=target_act_id,
        target_section_no=target_section_no,
    )


def process_act_citations(session, year: int, act_id: str, file_path: Path):
    total_cites = 0
    resolved_by_id = 0
    resolved_by_act = 0

    with file_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sec = json.loads(line)
            for cit in iter_citations_from_section(sec):
                total_cites += 1
                if "target_section_id" in cit:
                    session.execute_write(
                        merge_cites_by_id_tx,
                        cit["source_section_id"],
                        cit["target_section_id"],
                    )
                    resolved_by_id += 1
                elif "target_act_id" in cit and "target_section_no" in cit:
                    session.execute_write(
                        merge_cites_by_act_and_no_tx,
                        cit["source_section_id"],
                        cit["target_act_id"],
                        cit["target_section_no"],
                    )
                    resolved_by_act += 1

    print(
        f"{act_id}: citations processed={total_cites}, "
        f"resolved_by_id={resolved_by_id}, resolved_by_act+secNo={resolved_by_act}"
    )


# ---------- high-level enrichment steps ----------

def add_cites_edges(session):
    print("=== Step 1: Adding CITES edges from kg_ready/enriched ===")
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year = int(row["year"])
            act_id = row["act_id"]

            src = choose_source_file(year, act_id)
            if src is None:
                print(f"{act_id}: no non-empty kg_ready or enriched file, skipping")
                continue

            process_act_citations(session, year, act_id, src)


def add_same_term_as(session):
    print("=== Step 2: Linking DefinedTerm nodes with SAME_TERM_AS ===")
    cypher = """
    MATCH (t:DefinedTerm)
    WITH toLower(t.name) AS key, collect(t) AS terms
    WHERE size(terms) > 1
    UNWIND terms AS t1
    UNWIND terms AS t2
    WITH key, t1, t2 WHERE id(t1) < id(t2)
    MERGE (t1)-[:SAME_TERM_AS]->(t2)
    """
    session.run(cypher)
    print("SAME_TERM_AS relationships created (or updated).")


def add_appears_in_act(session):
    print("=== Step 3: Linking Role -> Act via APPEARS_IN_ACT ===")
    cypher = """
    MATCH (a:Act)-[:HAS_SECTION]->(s:Section)-[:MENTIONS_ROLE]->(r:Role)
    MERGE (r)-[:APPEARS_IN_ACT]->(a)
    """
    session.run(cypher)
    print("APPEARS_IN_ACT relationships created (or updated).")


def add_role_cooccurrence(session):
    print("=== Step 4: Building role CO_OCCURS_WITH network ===")
    cypher = """
    MATCH (s:Section)-[:MENTIONS_ROLE]->(r:Role)
    WITH s, collect(DISTINCT r) AS roles
    UNWIND roles AS r1
    UNWIND roles AS r2
    WITH r1, r2
    WHERE id(r1) < id(r2)
    MERGE (r1)-[c:CO_OCCURS_WITH]->(r2)
    ON CREATE SET c.count = 1
    ON MATCH  SET c.count = c.count + 1
    """
    session.run(cypher)
    print("CO_OCCURS_WITH relationships created/updated with counts.")


def add_severity_scores(session):
    print("=== Step 5: Computing severity_score for sections ===")
    cypher = """
    MATCH (s:Section)-[:PRESCRIBES_PENALTY]->(p:Penalty)
    WITH s, collect(p) AS pens
    UNWIND pens AS p
    WITH s, pens,
         sum(
           CASE WHEN p.imprisonment IS NOT NULL AND p.imprisonment <> "" THEN 2 ELSE 0 END +
           CASE WHEN p.fine_amount IS NOT NULL AND p.fine_amount <> "" THEN 1 ELSE 0 END
         ) AS baseScore
    WITH s, baseScore, size(pens) AS numPens
    SET s.severity_score = baseScore + numPens
    """
    session.run(cypher)
    print("severity_score set on sections with penalties.")


# ---------- main ----------

def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver, driver.session() as session:
        add_cites_edges(session)
        add_same_term_as(session)
        add_appears_in_act(session)
        add_role_cooccurrence(session)
        add_severity_scores(session)
    print("All enrichment steps completed.")


if __name__ == "__main__":
    main()
