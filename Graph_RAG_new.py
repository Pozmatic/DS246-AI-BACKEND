import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
 
from templates import GRAPH_TEMPLATE
 
load_dotenv()
 
from Indexing import driver       # same driver
from Search import (
    extract_article_id,
    fulltext_search_sections,
    vector_search_sections,
    rrf_fuse,
    smart_semantic_search,  # NEW: bring in your smart semantic search
)
 
 
# ----------------- LLM -----------------
 
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",             # or gemini-1.5-pro
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
 
 
# ----------------- small helper for trimming text -----------------
 
def _trim(text: str | None, max_chars: int) -> str | None:
    if text is None:
        return None
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars] + " ...[TRUNCATED]..."
 
 
# ----------------- richer KG context fetch -----------------
def fetch_kg_context_for_sections(section_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch Act + Section + citations + roles + obligations + penalties
    for given section Neo4j ids.
 
    - Uses HAS_SECTION as the primary way to tie Acts to Sections (like before).
    - Optionally also collects APPEARS_IN_ACT, but does not rely on it.
    - Builds a human-readable formatted_citation like:
        "Section 10 of THE MADRAS CITY LAND REVENUE ACT, 1851"
    """
    if not section_ids:
        return []
 
    with driver.session() as session:
        recs = session.run(
            """
            MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
            WHERE id(s) IN $sids
 
            // Optional: extra link if you have APPEARS_IN_ACT too
            OPTIONAL MATCH (s)-[:APPEARS_IN_ACT]->(a2:Act)
 
            OPTIONAL MATCH (s)-[:CITES]->(cited:Section)
            OPTIONAL MATCH (s)-[:MENTIONS_ROLE]->(r:Role)
            OPTIONAL MATCH (s)-[:IMPOSES_OBLIGATION]->(o:Obligation)
            OPTIONAL MATCH (s)-[:PRESCRIBES_PENALTY]->(p:Penalty)
 
            WITH a, s,
                 collect(DISTINCT a2) AS appears_in_acts,
                 collect(DISTINCT cited) AS cited_sections,
                 collect(DISTINCT r) AS roles,
                 collect(DISTINCT o) AS obligations,
                 collect(DISTINCT p) AS penalties
 
            RETURN id(s) AS sid,
                   a,
                   s,
                   appears_in_acts,
                   cited_sections,
                   roles,
                   obligations,
                   penalties
            """,
            {"sids": section_ids},
        ).data()
 
    results: List[Dict[str, Any]] = []
 
    for row in recs:
        act = row["a"]
        s = row["s"]
 
        # ---- act info (same as before) ----
        act_title = act.get("title") or act.get("name")
        act_year = act.get("year")
        act_number = act.get("act_number")
 
        # ---- section-level citation string ----
        # node property is 'citation' (from your screenshot)
        raw_citation = s.get("citation") or s.get("Citation")
 
        if raw_citation and act_title and act_year:
            formatted_citation = f"{raw_citation} of {act_title}, {act_year}"
        elif raw_citation and act_title:
            formatted_citation = f"{raw_citation} of {act_title}"
        else:
            formatted_citation = raw_citation  # may be None if not set
 
        entry: Dict[str, Any] = {
            # keep this if you were using it elsewhere
            "section_neo4j_id": row["sid"],
 
            # NEW: what the LLM should use in the answer
            "citation": raw_citation,
            "formatted_citation": formatted_citation,
 
            "act_title": act_title,
            "act_year": act_year,
            "act_number": act_number,
 
            # optional list of other acts from APPEARS_IN_ACT, if any
            "appears_in_acts": [
                {
                    "act_title": a2.get("title") or a2.get("name"),
                    "act_year": a2.get("year"),
                    "act_number": a2.get("act_number"),
                }
                for a2 in (row["appears_in_acts"] or [])
                if a2
            ],
 
            "section_heading": s.get("heading"),
            "section_summary": _trim(s.get("summary"), 800),
            "section_text": _trim(s.get("text"), 1500),
            "severity_score": s.get("severity_score"),
 
            "roles": [role.get("name") for role in (row["roles"] or []) if role],
            "obligations": [
                {
                    "id": o.get("id"),
                    "action": o.get("action"),
                    "conditions": o.get("conditions"),
                    "source_span": o.get("source_span"),
                }
                for o in (row["obligations"] or [])
                if o
            ],
            "penalties": [
                {
                    "id": p.get("id"),
                    "description": p.get("description"),
                    "imprisonment": p.get("imprisonment"),
                    "fine_amount": p.get("fine_amount"),
                    "source_span": p.get("source_span"),
                }
                for p in (row["penalties"] or [])
                if p
            ],
            "cited_sections": [
                {
                    "citation": cs.get("citation") or cs.get("Citation"),
                    "heading": cs.get("heading"),
                    "text": _trim(cs.get("text"), 400),
                }
                for cs in (row["cited_sections"] or [])
                if cs
            ],
        }
 
        results.append(entry)
 
    return results
 
 
 
# ----------------- main GraphRAG tool -----------------
 
@tool("Graph_RAG_new_tool")
def Graph_RAG_new_tool(query: str) -> str:
    """
    Semantic + hybrid RAG over the legal KG.
 
    Flow:
      user query
        -> detect explicit Article/Section (24-A, 420, etc.)
        -> section-level vector search (Chroma)
        -> section-level lexical search (Neo4j)
        -> hybrid RRF fusion
        -> if weak: smart semantic search (acts + sections)
        -> expand act hits to sections
        -> fetch KG neighborhood
        -> LLM answer using GRAPH_TEMPLATE
    """
 
    # 1) Detect explicit Article/Section number
    explicit_article_id = extract_article_id(query)
 
    # 2) Semantic search (Chroma, sections)
    vec_results = vector_search_sections(query, article_id=explicit_article_id, limit=50)
 
    # 3) Lexical search (Neo4j full-text)
    lex_records = fulltext_search_sections(query, limit=50)
    lex_results = [{"sid": r["sid"], "score": r["score"]} for r in lex_records]
 
    # 4) Hybrid RRF fusion on sections
    fused_section_ids = rrf_fuse(vec_results, lex_results, k_rrf=60, top_k=10)
 
    # ----------------- FALLBACK 1: explicit section id match -----------------
    if not fused_section_ids and explicit_article_id:
        with driver.session() as session:
            direct = session.run(
                """
                MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
                WHERE toUpper(s.section_id) = $sid
                RETURN id(s) AS sid
                """,
                {"sid": explicit_article_id},
            ).data()
        fused_section_ids = [d["sid"] for d in direct]
 
    # ----------------- FALLBACK 2: smart semantic search (acts + sections) -----------------
    # Uses your smart_semantic_search() from Search.py, which combines:
    #   - section embeddings
    #   - act-title embeddings
    if not fused_section_ids:
        sem_results = smart_semantic_search(query, limit=50)
 
        # Sections directly from semantic search
        sec_ids_from_sem = [r["id"] for r in sem_results if r["type"] == "section"]
 
        # Acts from semantic search -> expand to sections
        act_ids = [r["id"] for r in sem_results if r["type"] == "act"]
 
        if act_ids:
            with driver.session() as session:
                act_sec_rows = session.run(
                    """
                    MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
                    WHERE id(a) IN $aids
                    RETURN id(s) AS sid
                    ORDER BY s.sectionNo ASC
                    LIMIT 40
                    """,
                    {"aids": act_ids},
                ).data()
            sec_ids_from_sem.extend([row["sid"] for row in act_sec_rows])
 
        # Deduplicate and cap
        if sec_ids_from_sem:
            fused_section_ids = list(dict.fromkeys(sec_ids_from_sem))[:10]
 
    # ----------------- FALLBACK 3: fuzzy match on Act title only -----------------
    # Handles plain title queries like "Indian Red Cross Act"
    if not fused_section_ids:
        with driver.session() as session:
            act_based_secs = session.run(
                """
                MATCH (a:Act)-[:HAS_SECTION]->(s:Section)
                WHERE toLower(a.title) CONTAINS toLower($q)
                RETURN id(s) AS sid
                LIMIT 20
                """,
                {"q": query},
            ).data()
        fused_section_ids = [row["sid"] for row in act_based_secs]
 
    # Final: if still nothing, bail gracefully
    if not fused_section_ids:
        return (
            "I could not find any relevant sections in the knowledge graph for this query. "
            "Please try rephrasing it, or mention an Act name or section number explicitly."
        )
 
    # 5) Fetch KG neighborhood for the chosen sections
    kg_results = fetch_kg_context_for_sections(fused_section_ids)
 
    graph_context = json.dumps(kg_results, ensure_ascii=False, indent=2)
 
    # 6) LLM answer using GRAPH_TEMPLATE
    prompt = PromptTemplate(
        template=GRAPH_TEMPLATE,
        input_variables=["graph_context", "query"],
    )
 
    response = llm.invoke(
        prompt.format(
            graph_context=graph_context,
            query=query,
        )
    )
 
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "text"):
        return response.text
    return str(response)
 