"""
rag_pipeline.py
All retrieval, prompt building, and generation logic.
Imported by app.py — keep this file free of any Streamlit code.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import certifi
from groq import Groq
from langdetect import detect
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

def _secret(key: str) -> str:
    """Read from Streamlit secrets in production, .env locally."""
    try:
        import streamlit as st
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

# ── Configuration ──────────────────────────────────────────────────────────────
MONGO_URI  = _secret("MONGO_URI")
DB_NAME    = "ibu_legal_database"
MODEL_NAME = "llama-3.3-70b-versatile"

# ── Clients (initialised once when the module is imported) ─────────────────────
groq_client     = Groq(api_key=_secret("GROQ_API_KEY"))
embedding_model = SentenceTransformer("BAAI/bge-m3")


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def search_chunks(query: str, n_results: int = 5) -> list[dict]:
    """Vector search + cross-reference fetching from MongoDB Atlas."""
    client     = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    chunks_col = client[DB_NAME]["chunks"]

    query_vector = embedding_model.encode(
        [query], normalize_embeddings=True
    ).tolist()[0]

    results = list(chunks_col.aggregate([
        {
            "$vectorSearch": {
                "index":         "vector_index",
                "path":          "embedding",
                "queryVector":   query_vector,
                "numCandidates": 100,
                "limit":         n_results
            }
        },
        {
            "$project": {
                "breadcrumb":       1,
                "context_window":   1,
                "raw_text":         1,
                "hierarchy":        1,
                "filename":         1,
                "source_id":        1,
                "cross_references": 1,
                "page":             1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]))

    # Cross-reference fetching
    already_fetched = {
        r["hierarchy"].get("clan", "").rstrip(".") for r in results
    }
    xref_names = set()
    for r in results:
        for xref in r.get("cross_references", []):
            if xref.rstrip(".") not in already_fetched:
                xref_names.add(xref.rstrip("."))

    if xref_names:
        source_ids    = {r["source_id"] for r in results}
        xref_patterns = [
            {"hierarchy.clan": {"$regex": f"^{name}\\.?$"}}
            for name in xref_names
        ]
        xref_chunks = list(chunks_col.find(
            {"$or": xref_patterns, "source_id": {"$in": list(source_ids)}},
            {"breadcrumb": 1, "context_window": 1, "raw_text": 1,
             "hierarchy": 1, "filename": 1, "cross_references": 1, "page": 1}
        ))
        for chunk in xref_chunks:
            chunk["score"]    = None
            chunk["via_xref"] = True
        results += xref_chunks

    client.close()
    return results


def build_context_for_llm(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    primary   = [r for r in results if not r.get("via_xref")]
    secondary = [r for r in results if r.get("via_xref")]

    lines = []
    for r in primary:
        lines.append(
            f"[Source: {r['filename']} | {r['breadcrumb']} | "
            f"Score: {r['score']:.3f}]\n{r['raw_text']}\n"
        )
    if secondary:
        lines.append("--- Cross-referenced articles ---")
        for r in secondary:
            lines.append(
                f"[Source: {r['filename']} | {r['breadcrumb']} | "
                f"Referenced article]\n{r['raw_text']}\n"
            )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """Returns 'bs' for Bosnian/Croatian/Serbian, 'en' for everything else."""
    try:
        lang = detect(text)
        return "bs" if lang in ("bs", "hr", "sr", "sh") else "en"
    except:
        return "en"


# ══════════════════════════════════════════════════════════════════════════════
# QUERY REWRITING
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_query_if_followup(query: str,
                               conversation_history: list[dict]) -> str:
    """
    Detects whether the query is a follow-up or a new topic.
    If follow-up: rewrites to a self-contained question.
    If new topic: returns original query unchanged.
    """
    if not conversation_history:
        return query

    last_user = next(
        (m["content"] for m in reversed(conversation_history)
         if m["role"] == "user"), ""
    )
    last_assistant = next(
        (m["content"] for m in reversed(conversation_history)
         if m["role"] == "assistant"), ""
    )

    # Step 1: classify as follow-up or new topic
    topic_check = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content":
            f'Previous question: "{last_user}"\n'
            f'New question: "{query}"\n\n'
            f'Is the new question a follow-up to the previous one (same topic, '
            f'asking for more detail), or is it a completely new topic?\n'
            f'Reply with only one word: FOLLOWUP or NEWTOPIC'
        }],
        temperature=0.0
    ).choices[0].message.content.strip().upper()

    if "NEWTOPIC" in topic_check:
        return query

    # Step 2: rewrite the follow-up
    rewritten = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content":
            f'Previous question: "{last_user}"\n'
            f'Previous answer summary: "{last_assistant[:300]}"\n'
            f'Follow-up question: "{query}"\n\n'
            f'Rewrite the follow-up as a fully self-contained question. '
            f'Do NOT include specific article numbers or document names.\n'
            f'Return ONLY the rewritten question, nothing else.'
        }],
        temperature=0.0
    ).choices[0].message.content.strip()

    return rewritten


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(query: str,
                 context_block: str,
                 language: str,
                 conversation_history: list[dict]) -> list[dict]:
    """Build the full message list for the LLM."""
    lang_instruction = (
        "CRITICAL: Your ENTIRE response must be written exclusively in Bosnian. Do NOT write any sentence in English. Translate all content into Bosnian before responding."
        if language == "bs"
        else "CRITICAL: Your ENTIRE response must be written exclusively in English. Do NOT write any sentence in Bosnian. Translate all content into English before responding."
    )

    system_prompt = f"""You are a helpful regulatory assistant for students at International Burch University (IBU).
Your sole purpose is to answer questions about IBU regulations.

You must follow these rules strictly:
1. Answer ONLY using information present in the provided regulatory context. Never use outside knowledge.
2. Always cite your source using the breadcrumb provided, in this format: (Član X, DocumentName).
3. If the answer is not found in the provided context, respond with exactly:
   - In English:  "This information was not found in the available IBU regulatory documents."
   - In Bosnian:  "Ova informacija nije pronađena u dostupnim propisima IBU-a."
4. {lang_instruction}
5. Be concise and precise. Do not add introductory phrases like "Based on the context..."
6. If multiple articles are relevant, cite all of them.
7. If the retrieved context contains the relevant article, summarise its content fully. Do not say the information is incomplete if it is present in the context."""

    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_history[-6:]  # sliding window: last 3 turns

    messages.append({"role": "user", "content":
        f"Regulatory context:\n{context_block}\n\nQuestion: {query}"
    })
    return messages


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RAG FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def ask(query: str,
        conversation_history: list[dict],
        n_results: int = 5) -> dict:
    """
    Full RAG pipeline for one user turn.
    Returns: answer, sources, language, rewritten_query
    Updates conversation_history in place.
    """
    language     = detect_language(query)
    search_query = rewrite_query_if_followup(query, conversation_history)
    results      = search_chunks(search_query, n_results=n_results)
    context      = build_context_for_llm(results)
    messages     = build_prompt(query, context, language, conversation_history)

    answer = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1
    ).choices[0].message.content

    conversation_history.append({"role": "user",      "content": query})
    conversation_history.append({"role": "assistant", "content": answer})

    sources = [
        {
            "breadcrumb": r["breadcrumb"],
            "filename":   r["filename"],
            "page":       r.get("page", ""),
            "raw_text":   r["raw_text"],
            "score":      r.get("score"),
            "via_xref":   r.get("via_xref", False)
        }
        for r in results
    ]

    return {
        "answer":          answer,
        "sources":         sources,
        "language":        language,
        "rewritten_query": search_query,
    }
