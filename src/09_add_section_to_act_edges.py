# src/09_add_section_to_act_edges.py

from neo4j import GraphDatabase
import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "ds246@IISc")


def main():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    cypher = """
    MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
    MERGE (s)-[:OF_ACT]->(a)
    """

    with driver.session() as session:
        session.run(cypher)
        print("Created OF_ACT relationships from Section -> Act.")

    driver.close()


if __name__ == "__main__":
    main()
