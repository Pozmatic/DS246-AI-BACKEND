# rag_search.py
import re
from collections import defaultdict
from typing import List, Dict, Any
 
from Indexing import driver, sections_collection, acts_collection  # reuse driver & collection
 
# Only capture Section/Article numbers
ARTICLE_REGEX = re.compile(r"(?i)\b(article|section)\s+(\d+[A-Z-]*)")
 
 
 
def extract_article_id(query: str) -> str | None:
    """
    Extracts things like 'Article 24-A' / 'Section 420B' â†’ '24-A' / '420B'.
    """
    m = ARTICLE_REGEX.search(query)
    if not m:
        return None
    return m.group(2).upper()
 
 
# Search.py
 
def fulltext_search_sections(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Lexical search over Section.text / Section.summary without using a fulltext index.
    Does NOT modify the KG or require db.index.fulltext.*.
    """
    q = query.lower()
 
    with driver.session() as session:
        recs = session.run(
            """
            MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
            WHERE
              (s.text IS NOT NULL AND toLower(s.text) CONTAINS $q) OR
              (s.summary IS NOT NULL AND toLower(s.summary) CONTAINS $q)
            RETURN id(s) AS sid,
                   1.0 AS score,   // dummy score to keep same shape
                   a,
                   s
            LIMIT $limit
            """,
            {"q": q, "limit": limit},
        ).data()
    return recs
 
def vector_search_acts(query: str, limit: int = 10):
    res = acts_collection.query(
        query_texts=[query],
        n_results=limit,
    )
    results = []
    for i, aid in enumerate(res["ids"][0]):
        md = res["metadatas"][0][i]
        results.append(
            {
                "aid": int(md["neo4j_id"]),
                "score": float(res["distances"][0][i]),
                "metadata": md,
            }
        )
    return results
 
def vector_search_sections(query: str, article_id: str | None = None, limit: int = 10):
    """
    Semantic search over section text+summary in Chroma.
    If article_id is provided, we filter on that section_id to avoid 24-A vs 24-B confusion.
    """
    where = {}
    if article_id:
        where["section_id"] = article_id
 
    res = sections_collection.query(
        query_texts=[query],
        n_results=limit,
        where=where or None,
    )
 
    results = []
    for i, sid in enumerate(res["ids"][0]):
        md = res["metadatas"][0][i]
        results.append(
            {
                "sid": int(md["neo4j_id"]),
                "score": float(res["distances"][0][i]),
                "metadata": md,
            }
        )
    return results
 
def smart_semantic_search(query: str, limit: int = 10):
    # Section vector search
    sec_res = sections_collection.query(query_texts=[query], n_results=limit)
 
    # Act title vector search
    act_res = acts_collection.query(query_texts=[query], n_results=limit)
 
    merged = []
 
    for i, sid in enumerate(sec_res["ids"][0]):
        merged.append(
            {"type": "section", "id": int(sec_res["metadatas"][0][i]["neo4j_id"]),
             "score": 1-sec_res["distances"][0][i], "meta": sec_res["metadatas"][0][i]}
        )
 
    for i, aid in enumerate(act_res["ids"][0]):
        merged.append(
            {"type": "act", "id": int(act_res["metadatas"][0][i]["neo4j_id"]),
             "score": 1-act_res["distances"][0][i], "meta": act_res["metadatas"][0][i]}
        )
 
    merged = sorted(merged, key=lambda x: x["score"], reverse=True)
    return merged[:limit]
 
 
 
def rrf_fuse(
    vector_results: List[Dict[str, Any]],
    lexical_results: List[Dict[str, Any]],
    k_rrf: int = 60,
    top_k: int = 15,
) -> List[int]:
    """
    Reciprocal Rank Fusion over vector + lexical results.
    Returns top_k Neo4j section ids.
    """
    scores: Dict[int, float] = defaultdict(float)
 
    for rank, r in enumerate(vector_results):
        sid = r["sid"]
        scores[sid] += 1.0 / (k_rrf + rank + 1)
 
    for rank, r in enumerate(lexical_results):
        sid = r["sid"]
        scores[sid] += 1.0 / (k_rrf + rank + 1)
 
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [sid for sid, _ in fused[:top_k]]
 