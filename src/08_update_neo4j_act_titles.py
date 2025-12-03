# src/08_update_neo4j_act_titles.py

from neo4j import GraphDatabase
import json
import os
from pathlib import Path

NEW_ENRICHED_DIR = Path("data/new_enriched")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ds246@IISc")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    if not NEW_ENRICHED_DIR.exists():
        print("new_enriched directory not found:", NEW_ENRICHED_DIR)
        return

    with driver.session() as session:
        # iterate year folders
        for year_dir in sorted(NEW_ENRICHED_DIR.iterdir()):
            if not year_dir.is_dir():
                continue

            for file in sorted(year_dir.glob("*_enriched.jsonl")):
                print("Processing", file)

                act_id = None
                title = None

                # We only need the first non-empty line for that act
                with file.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        sec = json.loads(line)
                        act_id = sec.get("act_id")
                        title = (sec.get("act_title") or "").strip()
                        break  # first section line is enough

                if not act_id or not title:
                    print("  -> missing act_id or title, skipped")
                    continue

                session.run(
                    """
                    MATCH (a:Act {id: $act_id})
                    SET a.title = $title
                    """,
                    act_id=act_id,
                    title=title,
                )
                print(f"  -> Updated Act {act_id} with title: {title}")

    driver.close()
    print("Done updating Act titles in Neo4j.")


if __name__ == "__main__":
    main()
