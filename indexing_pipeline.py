"""
indexing_pipeline.py
====================
Converts all PDFs in the docs/ folder into article-aware chunks,
saves them to MongoDB Atlas, and generates BGE-M3 embeddings.

Run once to (re)build the full knowledge base:
    python indexing_pipeline.py
"""

import os
import re
import fitz
import pdfplumber
import certifi
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# ── Load credentials from .env ─────────────────────────────────────────────────
load_dotenv()
MONGO_URI = os.environ["MONGO_URI"]
DB_NAME   = "ibu_legal_database"

# ── Document metadata ──────────────────────────────────────────────────────────
# Keys must exactly match the PDF filenames inside your docs/ folder.
DOCUMENTS = {
    "1-Pravilnik o upisu studenata i kriterijima za sticanje visokoskolskih kvalifikacija IBU.pdf":
        {"doc_id": "doc_001", "doc_type": "pravilnik", "language": "bs"},

    "2-Eticki kodeks studenata Internacionalnog Burc univerziteta.pdf":
        {"doc_id": "doc_002", "doc_type": "kodeks",    "language": "bs"},

    "3-Pravilnik o radu studentskih klubova.pdf":
        {"doc_id": "doc_003", "doc_type": "pravilnik", "language": "bs"},

    "4-Pravila studiranja za prvi ciklus.pdf":
        {"doc_id": "doc_004", "doc_type": "pravila",   "language": "bs"},

    "5-Pravila studiranja za drugi ciklus.pdf":
        {"doc_id": "doc_005", "doc_type": "pravila",   "language": "bs"},

    "6-Pravila studiranja za treci ciklus.pdf":
        {"doc_id": "doc_006", "doc_type": "pravila",   "language": "bs"},

    "7-Pravilnik o priznavanju inostranih visokoskolskih kvalifikacija.pdf":
        {"doc_id": "doc_007", "doc_type": "pravilnik", "language": "bs"},

    "8-Pravilnik o pokretanju, promjenama i evaluaciji studijskih programa.pdf":
        {"doc_id": "doc_008", "doc_type": "pravilnik", "language": "bs"},

    "9-Pravilnik o studentskoj evaluaciji kvaliteta nastave IBU.pdf":
        {"doc_id": "doc_009", "doc_type": "pravilnik", "language": "bs"},

    "10-Pravilnik o stipendiranju studenata IBU.pdf":
        {"doc_id": "doc_010", "doc_type": "pravilnik", "language": "bs"},

    "11-Pravilnik o disciplinskoj odgovornosti studenata IBU.pdf":
        {"doc_id": "doc_011", "doc_type": "pravilnik", "language": "bs"},

    "12-Statut Internacionalnog Burc univerziteta.pdf":
        {"doc_id": "doc_012", "doc_type": "statut",    "language": "bs"},
}

# ── Regex patterns ─────────────────────────────────────────────────────────────
CLAN_PATTERN  = re.compile(r"^(Član\s+\d+[a-z]?\.?)\s*(.*)$", re.IGNORECASE | re.MULTILINE)
GLAVA_PATTERN = re.compile(r"^(Glava\s+[IVXLC\d]+[^\n]*|Poglavlje\s+[IVXLC\d]+[^\n]*)", re.IGNORECASE | re.MULTILINE)
XREF_PATTERN  = re.compile(r"[Čč]lan(?:om|a|u|e)?\s+(\d+[a-z]?)", re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION & CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = re.sub(r"^\s*[-–]?\s*\d+\s*[-–]?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()

def detect_cross_references(text: str) -> list:
    return list(set(f"Član {m}" for m in XREF_PATTERN.findall(text)))

def extract_tables_by_page(pdf_path: str) -> dict:
    tables_by_page = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = [t for t in page.extract_tables() if t]
            if page_tables:
                tables_by_page[i] = page_tables
    return tables_by_page

def extract_text_by_page(pdf_path: str) -> list:
    doc   = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        raw     = doc[i].get_text("text")
        cleaned = clean_text(raw)
        if cleaned:
            pages.append((i, cleaned))
    doc.close()
    return pages

def build_blocks_for_page(text: str, page_tables: list) -> list:
    blocks        = []
    order         = 1
    table_strings = set()
    for table in page_tables:
        for row in table:
            row_text = " ".join(str(cell).strip() for cell in row if cell)
            if row_text:
                table_strings.add(row_text.strip())

    for para in text.split("\n\n"):
        para = para.strip()
        if not para or para in table_strings:
            continue
        blocks.append({"type": "paragraph", "order": order, "raw_text": para})
        order += 1

    for table in page_tables:
        if not table:
            continue
        flat = ". ".join(
            " ".join(str(cell).strip() for cell in row if cell)
            for row in table if any(cell for cell in row)
        )
        if flat.strip():
            blocks.append({"type": "table", "order": order, "raw_text": flat, "data": table})
            order += 1

    return blocks

def parse_into_chunks(pages: list, tables_by_page: dict) -> list:
    full_lines   = []
    page_map     = {}
    line_counter = 0
    for page_idx, text in pages:
        for line in text.splitlines():
            full_lines.append(line)
            page_map[line_counter] = page_idx
            line_counter += 1
    full_text = "\n".join(full_lines)

    clan_matches  = list(CLAN_PATTERN.finditer(full_text))
    glava_matches = list(GLAVA_PATTERN.finditer(full_text))

    def get_current_glava(pos: int) -> str:
        current = ""
        for m in glava_matches:
            if m.start() <= pos:
                current = m.group(0).strip()
            else:
                break
        return current

    chunks = []
    for i, match in enumerate(clan_matches):
        clan_name   = match.group(1).strip()
        clan_naslov = match.group(2).strip()
        start_pos   = match.start()
        end_pos     = clan_matches[i + 1].start() if i + 1 < len(clan_matches) else len(full_text)

        clan_text = full_text[start_pos:end_pos].strip()
        glava     = get_current_glava(start_pos)

        line_at_start = full_text[:start_pos].count("\n")
        page_idx      = page_map.get(min(line_at_start, len(page_map) - 1), 0)

        page_tables = tables_by_page.get(page_idx, [])
        blocks      = build_blocks_for_page(clan_text, page_tables)

        xrefs = detect_cross_references(clan_text)
        xrefs = [x for x in xrefs if x != clan_name.rstrip(".")]

        all_raw        = " ".join(b["raw_text"] for b in blocks)
        context_window = (
            f"{glava}, {clan_name} ({clan_naslov}): {all_raw}"
            if glava else
            f"{clan_name} ({clan_naslov}): {all_raw}"
        )
        breadcrumb = (
            f"{glava} > {clan_name} – {clan_naslov}" if glava and clan_naslov else
            f"{glava} > {clan_name}"                 if glava else
            f"{clan_name} – {clan_naslov}"           if clan_naslov else
            clan_name
        )

        chunks.append({
            "clan": clan_name, "clan_naslov": clan_naslov, "glava": glava,
            "page": page_idx + 1, "raw_text": clan_text, "blocks": blocks,
            "context_window": context_window, "breadcrumb": breadcrumb,
            "cross_references": xrefs,
        })

    if not chunks:
        full_combined = "\n\n".join(t for _, t in pages)
        chunks.append({
            "clan": None, "clan_naslov": None, "glava": None, "page": 1,
            "raw_text": full_combined,
            "blocks": [{"type": "paragraph", "order": 1, "raw_text": full_combined}],
            "context_window": full_combined, "breadcrumb": "",
            "cross_references": [],
        })

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# MONGODB
# ══════════════════════════════════════════════════════════════════════════════

def save_to_mongo(pdf_path: str, chunks: list, metadata: dict):
    client     = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    db         = client[DB_NAME]
    docs_col   = db["documents"]
    chunks_col = db["chunks"]

    filename = os.path.basename(pdf_path)
    doc_id   = metadata["doc_id"]

    # Skip if this document is already indexed
    if docs_col.find_one({"_id": doc_id}):
        print(f"    Already indexed, skipping.")
        client.close()
        return

    docs_col.insert_one({
        "_id":          doc_id,
        "filename":     filename,
        "doc_type":     metadata.get("doc_type", ""),
        "language":     metadata.get("language", "bs"),
        "acquired_at":  datetime.now(timezone.utc).isoformat(),
        "total_chunks": len(chunks),
    })

    chunk_docs = []
    for i, chunk in enumerate(chunks):
        chunk_docs.append({
            "_id":              f"{doc_id}_chunk_{str(i).zfill(3)}",
            "source_id":        doc_id,
            "filename":         filename,
            "chunk_index":      i,
            "page":             chunk["page"],
            "hierarchy": {
                "glava":       chunk["glava"],
                "clan":        chunk["clan"],
                "clan_naslov": chunk["clan_naslov"],
            },
            "breadcrumb":       chunk["breadcrumb"],
            "raw_text":         chunk["raw_text"],
            "blocks":           chunk["blocks"],
            "context_window":   chunk["context_window"],
            "cross_references": chunk["cross_references"],
            "embedding":        None,
        })

    if chunk_docs:
        chunks_col.insert_many(chunk_docs)

    client.close()
    print(f"    Saved {len(chunk_docs)} chunks for {doc_id} ({filename})")


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def embed_chunks(model: SentenceTransformer, batch_size: int = 32):
    client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    col    = client[DB_NAME]["chunks"]

    to_embed = list(col.find({"embedding": None}, {"_id": 1, "context_window": 1}))
    total    = len(to_embed)
    print(f"\n  Embedding {total} chunks...")

    for i in range(0, total, batch_size):
        batch      = to_embed[i : i + batch_size]
        texts      = [c["context_window"] for c in batch]
        ids        = [c["_id"]            for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        for chunk_id, vec in zip(ids, embeddings):
            col.update_one({"_id": chunk_id}, {"$set": {"embedding": vec}})

        print(f"    {min(i + batch_size, total)} / {total}", end="\r")

    print(f"    {total} / {total} — done.          ")
    client.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    docs_folder = Path(__file__).parent / "docs"

    if not docs_folder.exists():
        print(f"ERROR: docs/ folder not found at {docs_folder}")
        print("Create a 'docs' folder next to indexing_pipeline.py and put your PDFs in it.")
        return

    # ── Step 1: Extract & save ─────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Extracting articles → MongoDB 'chunks'")
    print("=" * 60)

    found = 0
    for filename, metadata in DOCUMENTS.items():
        pdf_path = docs_folder / filename
        if not pdf_path.exists():
            print(f"  [SKIP] {filename} — not found in docs/")
            continue

        print(f"\n  Processing: {filename}")
        pages          = extract_text_by_page(str(pdf_path))
        tables_by_page = extract_tables_by_page(str(pdf_path))
        chunks         = parse_into_chunks(pages, tables_by_page)
        print(f"    Articles found: {len(chunks)}")
        save_to_mongo(str(pdf_path), chunks, metadata)
        found += 1

    print(f"\n  {found} / {len(DOCUMENTS)} documents processed.")

    # ── Step 2: Embed ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Generating BGE-M3 embeddings")
    print("=" * 60)
    print("\n  Loading model BAAI/bge-m3 (downloads ~2.3 GB on first run)...")
    model = SentenceTransformer("BAAI/bge-m3")
    embed_chunks(model)

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print("""
Next step — create a Vector Search index in MongoDB Atlas:

  1. Go to your cluster → Atlas Search → Create Search Index
  2. Choose Atlas Vector Search → JSON Editor
  3. Select collection: ibu_legal_database.chunks
  4. Paste this config:
     {"fields":[{"type":"vector","path":"embedding",
      "numDimensions":1024,"similarity":"cosine"}]}
  5. Name the index: vector_index
  6. Click Create — it builds in ~2 minutes

Then run:  streamlit run app.py
""")


if __name__ == "__main__":
    main()