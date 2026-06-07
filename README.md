# IBU Regulatory Chatbot

A bilingual (Bosnian/English) Retrieval-Augmented Generation (RAG) chatbot for querying International Burch University regulatory documents. Students can ask natural language questions about university rules and regulations and receive accurate, source-cited answers.

Built with BGE-M3 embeddings, MongoDB Atlas Vector Search, Llama 3.3 70B via Groq, and Streamlit.

---

## Features

- Ask questions in **English or Bosnian** — language is detected automatically
- Answers are grounded exclusively in official IBU regulatory documents
- Every answer cites the exact article it came from
- Cross-referenced articles are automatically fetched and included
- Multi-turn conversation with follow-up question support
- Source panel shows retrieved articles with similarity scores

---

## Project Structure

```
IBU_LEGAL_CHATBOT/
├── docs/                    # IBU regulatory PDFs (12 documents)
├── app.py                   # Streamlit chat interface
├── rag_pipeline.py          # Retrieval, prompt building, generation
├── indexing_pipeline.py     # PDF extraction, chunking, embedding, upload
├── requirements.txt
├── .env                     # Your credentials (not committed)
└── .gitignore
```

---

## Prerequisites

- Python 3.10 or later
- Git
- A [MongoDB Atlas](https://cloud.mongodb.com) account (free tier is sufficient)
- A [Groq](https://console.groq.com) account (free tier is sufficient)

---

## Replication Guide

### 1. Clone the Repository

Open PowerShell and run:

```
git clone https://github.com/azrasosic/IBU-Regulatory-Chatbot.git
cd ibu-regulatory-chatbot
```

### 2. Create and Activate a Virtual Environment

```
C:\Python312\python.exe -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt. Verify Python is pointing to the venv:

```
get-command python
```

It should show `...\venv\Scripts\python.exe`.

### 3. Install Dependencies

```
pip install -r requirements.txt
```

This will take a few minutes — `torch` and `sentence-transformers` are large packages.

### 4. Configure API Keys

Create a `.env` file in the project root with the following contents:

```
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
GROQ_API_KEY=gsk_<your_key_here>
```

**Groq API key:** Log in at [console.groq.com](https://console.groq.com) → API Keys → Create API Key.

**MongoDB connection string:** Log in at [cloud.mongodb.com](https://cloud.mongodb.com) → connect to your cluster → Drivers → copy the connection string and replace `<username>` and `<password>` with your database user credentials.

Make sure your IP is whitelisted: MongoDB Atlas → Network Access → Add IP Address → Allow Access From Anywhere.

### 5. Populate the Knowledge Base

Run the indexing pipeline to extract articles from all PDFs, save them to MongoDB, and generate embeddings:

```
python indexing_pipeline.py
```

This takes 10–15 minutes on first run (the BGE-M3 model, ~2.3 GB, is downloaded automatically).

Once finished, create a Vector Search index in MongoDB Atlas:

1. Go to your cluster → **Atlas Search** → **Create Search Index**
2. Select **Atlas Vector Search** → **JSON Editor**
3. Select collection: `ibu_legal_database.chunks`
4. Name the index: `vector_index`
5. Paste this configuration:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    }
  ]
}
```

6. Click **Create Search Index** — it builds in ~2 minutes.

### 6. Run the Application

```
streamlit run app.py
```

The chatbot will open in your browser at `http://localhost:8501`.

---

## Adding New Documents

To add a new regulatory document to the knowledge base:

1. Place the PDF in the `docs/` folder
2. Add a new entry to the `DOCUMENTS` dictionary at the top of `indexing_pipeline.py`:

```python
"Your New Document Filename.pdf":
    {"doc_id": "doc_013", "doc_type": "pravilnik", "language": "bs"},
```

3. Re-run `python indexing_pipeline.py` — it skips already-indexed documents and only processes the new one.

---

## Technology Stack

| Component | Technology |
|---|---|
| Embedding model | BAAI/bge-m3 (via sentence-transformers) |
| Vector database | MongoDB Atlas Vector Search |
| Language model | Llama 3.3 70B (via Groq API) |
| PDF extraction | PyMuPDF + pdfplumber |
| Language detection | langdetect |
| Frontend | Streamlit |

---

## Notes

- Answers are based solely on the 12 IBU regulatory documents included in the knowledge base. Questions outside this scope will return a "not found" response.
- For official matters, always consult the IBU administration directly.