from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from neo4j import GraphDatabase
import os
from tqdm import tqdm   # progress bar
 
# -----------------------------
# 1. Embedding model (optional – used inside Chroma EF)
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
 
# If you want a shared model object for other code, you can still keep this:
embedding_model = SentenceTransformer(MODEL_NAME)
 
# Chroma embedding function wrapper
chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_NAME
)
 
 
# -----------------------------
# 2. Persistent Chroma client + collections
# -----------------------------
CHROMA_DB_PATH = "./chroma_db_vecc"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
 
# IMPORTANT: use chromadb.PersistentClient (not chroma.PersistentClient)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
 
# Sections (vector search over section text)
sections_collection = chroma_client.get_or_create_collection(
    name="legal_sections",
    embedding_function=chroma_ef,
    metadata={"hnsw:space": "cosine"}
)
 
# Acts (vector search over act titles)
acts_collection = chroma_client.get_or_create_collection(
    name="legal_acts",
    embedding_function=chroma_ef,
    metadata={"hnsw:space": "cosine"}
)
 
# Other entities: DefinedTerm, Role, Obligation, Penalty
entities_collection = chroma_client.get_or_create_collection(
    name="legal_entities",
    embedding_function=chroma_ef,
    metadata={"hnsw:space": "cosine"}
)
 
 
# -----------------------------
# 3. Neo4j connection
# -----------------------------
driver = GraphDatabase.driver(
    uri="bolt://localhost:7687",
    auth=("neo4j", "ds246@IISc")
)
 
 
# -----------------------------
# 4. Indexing functions
# -----------------------------
def index_sections_into_chroma(batch_size: int = 30):
    """
    Pulls sections from Neo4j and stores their embeddings + metadata
    into a persistent Chroma vector DB with progress bar.
 
    IMPORTANT: We intentionally DO NOT embed section numbers / IDs
    in the document text to avoid similarity artifacts (e.g., 112 vs 113).
    Those stay only in metadata.
    """
 
    with driver.session() as session:
        records = session.run(
            """
            MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
            RETURN id(s) AS sid, a, s
            """
        ).data()
 
    print(f"Found {len(records)} sections to index into Chroma")
 
    ids, texts, metadatas = [], [], []
 
    for rec in records:
        act = rec["a"]
        sec = rec["s"]
        sid = rec["sid"]
 
        # Section identifier stored only as metadata, not in embedded text
        article_id = sec.get("section_id") or sec.get("article_id")
 
        act_title = act.get("title") or act.get("name") or ""
        heading = sec.get("heading") or ""
        summary = sec.get("summary") or ""
        text = sec.get("text") or ""
 
        # 👇 This is what gets embedded: NO sectionNo, NO citation, NO ID.
        body = (
            f"{act_title}\n"
            f"{heading}\n\n"
            f"{summary}\n\n"
            f"{text}"
        ).strip()
 
        ids.append(str(sid))
        texts.append(body)
        metadatas.append(
            {
                "neo4j_id": sid,
                "act_title": act.get("title") or act.get("name"),
                "act_year": act.get("year"),
                "act_number": act.get("act_number"),
                "section_id": article_id,
                "sectionNo": sec.get("sectionNo"),   # kept only as metadata
                "citation": sec.get("citation"),
                "node_label": "Section",
            }
        )
 
    # Batch upload with progress bar
    for i in tqdm(range(0, len(ids), batch_size), desc="🔧 Indexing Sections into Chroma"):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
 
        sections_collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_meta,
        )
 
    print(f"✔ Completed indexing: {len(ids)} sections stored in ChromaDB at `{CHROMA_DB_PATH}`")
    print(f"Sections in Chroma collection now: {sections_collection.count()}")
 
 
def index_acts_into_chroma():
    """
    Index Act titles into Chroma for act-level semantic search.
    """
 
    with driver.session() as session:
        acts = session.run(
            "MATCH (a:Act) RETURN id(a) AS aid, a.title AS title, a.year AS year, a.act_number AS act_number"
        ).data()
 
    print(f"Found {len(acts)} acts to index into Chroma")
 
    ids, docs, meta = [], [], []
 
    for rec in acts:
        ids.append(str(rec["aid"]))
        docs.append(rec["title"] or "")
        meta.append(
            {
                "neo4j_id": rec["aid"],
                "title": rec["title"],
                "year": rec["year"],
                "act_number": rec["act_number"],
                "node_label": "Act",
            }
        )
 
    acts_collection.upsert(
        ids=ids,
        documents=docs,
        metadatas=meta,
    )
 
    print(f"✔ Completed indexing: {len(ids)} acts stored in ChromaDB at `{CHROMA_DB_PATH}`")
    print(f"Acts in Chroma collection now: {acts_collection.count()}")
 
 
def index_entities_into_chroma(batch_size: int = 50):
    """
    Index other entities: DefinedTerm, Role, Obligation, Penalty
    into `legal_entities` collection.
 
    We embed rich natural-language descriptions + context (Act + Section text)
    but keep all IDs / numeric fields only in metadata.
    """
 
    with driver.session() as session:
        # DefinedTerm with context of the defining section
        term_recs = session.run(
            """
            MATCH (a:Act)-[:HAS_SECTION]->(s:Section)-[:DEFINES]->(t:DefinedTerm)
            RETURN
              id(t) AS nid,
              t.id AS entity_id,
              t.name AS name,
              a.title AS act_title,
              a.year AS act_year,
              s.sectionNo AS section_no,
              s.heading AS heading,
              s.summary AS summary,
              s.text AS section_text
            """
        ).data()
 
        # Roles, with Acts they appear in
        role_recs = session.run(
            """
            MATCH (r:Role)
            OPTIONAL MATCH (r)-[:APPEARS_IN_ACT]->(a:Act)
            OPTIONAL MATCH (s:Section)-[:MENTIONS_ROLE]->(r)
            RETURN
              id(r) AS nid,
              r.id AS entity_id,
              r.name AS name,
              collect(DISTINCT a.title) AS act_titles,
              collect(DISTINCT s.heading) AS headings
            """
        ).data()
 
        # Obligations, linked to Section + Role
        obl_recs = session.run(
            """
            MATCH (a:Act)-[:HAS_SECTION]->(s:Section)-[:IMPOSES_OBLIGATION]->(o:Obligation)
            OPTIONAL MATCH (o)-[:OBLIGATION_ON]->(r:Role)
            RETURN
              id(o) AS nid,
              o.id AS entity_id,
              o.action AS action,
              o.conditions AS conditions,
              o.source_span AS source_span,
              a.title AS act_title,
              a.year AS act_year,
              s.sectionNo AS section_no,
              s.heading AS heading,
              s.summary AS summary,
              s.text AS section_text,
              collect(DISTINCT r.name) AS roles
            """
        ).data()
 
        # Penalties, linked to Section + Role
        pen_recs = session.run(
            """
            MATCH (a:Act)-[:HAS_SECTION]->(s:Section)-[:PRESCRIBES_PENALTY]->(p:Penalty)
            OPTIONAL MATCH (p)-[:APPLIES_TO]->(r:Role)
            RETURN
              id(p) AS nid,
              p.id AS entity_id,
              p.description AS description,
              p.imprisonment AS imprisonment,
              p.fine_amount AS fine_amount,
              p.source_span AS source_span,
              a.title AS act_title,
              a.year AS act_year,
              s.sectionNo AS section_no,
              s.heading AS heading,
              s.summary AS summary,
              s.text AS section_text,
              collect(DISTINCT r.name) AS roles
            """
        ).data()
 
    print("Found entities to index:")
    print(f"  DefinedTerm: {len(term_recs)}")
    print(f"  Role:        {len(role_recs)}")
    print(f"  Obligation:  {len(obl_recs)}")
    print(f"  Penalty:     {len(pen_recs)}")
 
    ids, docs, metas = [], [], []
 
    # -------- DefinedTerm --------
    for rec in term_recs:
        nid = rec["nid"]
        name = rec["name"] or ""
        act_title = rec["act_title"] or ""
        heading = rec["heading"] or ""
        summary = rec["summary"] or ""
        section_text = rec["section_text"] or ""
 
        # No IDs / section numbers in the doc text
        doc = (
            f"Defined term: {name}\n"
            f"Act: {act_title}\n"
            f"Section heading: {heading}\n\n"
            f"{summary}\n\n"
            f"{section_text}"
        ).strip()
 
        ids.append(str(nid))
        docs.append(doc)
        metas.append(
            {
                "neo4j_id": nid,
                "entity_id": rec["entity_id"],
                "node_label": "DefinedTerm",
                "name": name,
                "act_title": act_title,
                "section_no": rec["section_no"],
            }
        )
 
    # -------- Role --------
    for rec in role_recs:
        nid = rec["nid"]
        name = rec["name"] or ""
        act_titles = [a for a in rec["act_titles"] if a]
        headings = [h for h in rec["headings"] if h]
 
        doc = (
            f"Role: {name}\n"
            f"Appears in Acts: {', '.join(act_titles)}\n"
            f"Typical sections: {', '.join(headings)}"
        ).strip()
 
        ids.append(str(nid))
        docs.append(doc)
        metas.append(
            {
                "neo4j_id": nid,
                "entity_id": rec["entity_id"],
                "node_label": "Role",
                "name": name,
                "act_titles": act_titles,
            }
        )
 
    # -------- Obligation --------
    for rec in obl_recs:
        nid = rec["nid"]
        act_title = rec["act_title"] or ""
        heading = rec["heading"] or ""
        summary = rec["summary"] or ""
        section_text = rec["section_text"] or ""
        action = rec["action"] or ""
        conditions = rec["conditions"] or ""
        roles = [r for r in rec["roles"] if r]
 
        doc = (
            f"Obligation in Act: {act_title}\n"
            f"Section heading: {heading}\n\n"
            f"Obligation action: {action}\n"
            f"Conditions: {conditions}\n"
            f"Applies to roles: {', '.join(roles)}\n\n"
            f"{summary}\n\n"
            f"{section_text}"
        ).strip()
 
        ids.append(str(nid))
        docs.append(doc)
        metas.append(
            {
                "neo4j_id": nid,
                "entity_id": rec["entity_id"],
                "node_label": "Obligation",
                "act_title": act_title,
                "section_no": rec["section_no"],
                "roles": roles,
            }
        )
 
    # -------- Penalty --------
    for rec in pen_recs:
        nid = rec["nid"]
        act_title = rec["act_title"] or ""
        heading = rec["heading"] or ""
        summary = rec["summary"] or ""
        section_text = rec["section_text"] or ""
        description = rec["description"] or ""
        imprisonment = rec["imprisonment"] or ""
        fine_amount = rec["fine_amount"] or ""
        roles = [r for r in rec["roles"] if r]
 
        doc = (
            f"Penalty in Act: {act_title}\n"
            f"Section heading: {heading}\n\n"
            f"Description: {description}\n"
            f"Imprisonment: {imprisonment}\n"
            f"Fine: {fine_amount}\n"
            f"Applies to roles: {', '.join(roles)}\n\n"
            f"{summary}\n\n"
            f"{section_text}"
        ).strip()
 
        ids.append(str(nid))
        docs.append(doc)
        metas.append(
            {
                "neo4j_id": nid,
                "entity_id": rec["entity_id"],
                "node_label": "Penalty",
                "act_title": act_title,
                "section_no": rec["section_no"],
                "roles": roles,
            }
        )
 
    # -------- Batch upsert --------
    for i in tqdm(range(0, len(ids), batch_size), desc="🔧 Indexing Entities into Chroma"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = docs[i:i+batch_size]
        batch_meta = metas[i:i+batch_size]
 
        entities_collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )
 
    print(f"✔ Completed indexing: {len(ids)} entities stored in ChromaDB at `{CHROMA_DB_PATH}`")
    print(f"Entities in Chroma collection now: {entities_collection.count()}")
 
 
# -----------------------------
# 5. Debug helpers
# -----------------------------
def debug_neo4j_counts():
    """
    Print how many Acts / Sections / entities exist in Neo4j.
    Useful to compare with Chroma counts.
    """
    with driver.session() as session:
        acts = session.run("MATCH (a:Act) RETURN count(a) AS c").data()[0]["c"]
        sections = session.run("MATCH (s:Section) RETURN count(s) AS c").data()[0]["c"]
        terms = session.run("MATCH (t:DefinedTerm) RETURN count(t) AS c").data()[0]["c"]
        roles = session.run("MATCH (r:Role) RETURN count(r) AS c").data()[0]["c"]
        obls = session.run("MATCH (o:Obligation) RETURN count(o) AS c").data()[0]["c"]
        pens = session.run("MATCH (p:Penalty) RETURN count(p) AS c").data()[0]["c"]
 
    print("Neo4j node counts:")
    print(f"  Act:         {acts}")
    print(f"  Section:     {sections}")
    print(f"  DefinedTerm: {terms}")
    print(f"  Role:        {roles}")
    print(f"  Obligation:  {obls}")
    print(f"  Penalty:     {pens}")
 
 
def debug_chroma_counts():
    """
    Print how many vectors are in each Chroma collection.
    """
    print("Chroma collection counts:")
    print(f"  legal_acts:      {acts_collection.count()}")
    print(f"  legal_sections:  {sections_collection.count()}")
    print(f"  legal_entities:  {entities_collection.count()}")
 
 
 
if __name__ == "__main__":
    debug_neo4j_counts()
    index_acts_into_chroma()
    index_sections_into_chroma()
    index_entities_into_chroma()
    debug_chroma_counts()
 
 