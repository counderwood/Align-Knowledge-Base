"""
Align Builders Knowledge Base — Cloud Server v3
================================================
Uses simple TF-IDF style embeddings instead of ML models.
No heavy downloads, no memory issues on Railway free tier.
"""

import hashlib
import io
import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter

import anthropic
import pdfplumber
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS", "")
DRIVE_FOLDER_ID    = os.environ.get("DRIVE_FOLDER_ID", "1BfpoW2xFO1byMc1AH5JIiplozYZr7LmU")
CLAUDE_MODEL       = "claude-sonnet-4-20250514"
TOP_K              = 6
PORT               = int(os.environ.get("PORT", 8000))
SYNC_INTERVAL      = 3600

SYSTEM_PROMPT = """You are a knowledgeable assistant for Align Builders, a commercial construction company.
You have access to a library of internal proposal documents, project narratives, and RFQ/RFP submittals.

Answer questions using only the provided context. Be specific. Mention which document your answer 
comes from when relevant. If the context doesn't contain enough to answer confidently, say so clearly.

Write in clean, direct prose. No filler, no hedging."""

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("align-kb")

# ---------------------------------------------------------------------------
# In-memory document store (no ChromaDB, no ML models)
# ---------------------------------------------------------------------------

# Simple in-memory store: list of {id, text, source, file_id, tokens}
_documents = []
_indexed_files = {}
_sync_status = {"state": "idle", "message": "Not started", "indexed": 0, "total": 0}


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    text = text.lower()
    words = re.findall(r'\b[a-z][a-z0-9]{2,}\b', text)
    # Remove common stop words
    stops = {'the','and','for','are','was','were','this','that','with','have',
             'from','they','will','been','their','said','each','which','there',
             'what','about','would','make','like','into','than','them','some',
             'these','could','other','more','also','but','not','can','its'}
    return [w for w in words if w not in stops]


def bm25_score(query_tokens: list[str], doc_tokens: list[str], avg_dl: float, k1=1.5, b=0.75) -> float:
    """BM25 relevance scoring — much better than simple keyword match."""
    if not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    doc_freq = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term in doc_freq:
            tf = doc_freq[term]
            idf = math.log(1 + (len(_documents) + 0.5) / (1 + sum(1 for d in _documents if term in d['token_set'])))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """Search documents using BM25 scoring."""
    if not _documents:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    avg_dl = sum(len(d['tokens']) for d in _documents) / len(_documents)

    scored = []
    for doc in _documents:
        score = bm25_score(query_tokens, doc['tokens'], avg_dl)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        results.append({
            "text": doc["text"],
            "source": doc["source"],
            "relevance": round(score, 3)
        })
    return results


def add_document(text: str, source: str, file_id: str, chunk_index: int):
    """Add a document chunk to the in-memory store."""
    tokens = tokenize(text)
    doc_id = f"{file_id}_chunk_{chunk_index}"

    # Remove existing chunks for this file if re-indexing
    global _documents
    _documents = [d for d in _documents if d['file_id'] != file_id or d['chunk_index'] != chunk_index]

    _documents.append({
        "id": doc_id,
        "text": text,
        "source": source,
        "file_id": file_id,
        "chunk_index": chunk_index,
        "tokens": tokens,
        "token_set": set(tokens)
    })


def remove_file_chunks(file_id: str):
    """Remove all chunks for a given file."""
    global _documents
    _documents = [d for d in _documents if d['file_id'] != file_id]


# ---------------------------------------------------------------------------
# Google Drive
# ---------------------------------------------------------------------------

def get_drive_service():
    if GOOGLE_CREDENTIALS:
        creds_dict = json.loads(GOOGLE_CREDENTIALS)
    else:
        with open("align-knowledge-base-ad25c3204d55.json") as f:
            creds_dict = json.load(f)

    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_pdfs_in_folder(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
        fields="files(id, name, modifiedTime)",
        pageSize=200
    ).execute()
    return results.get("files", [])


def download_pdf(service, file_id):
    req = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def extract_text(pdf_bytes, filename):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"[Page {i+1}]\n{page_text}")
            text = "\n\n".join(pages)
    except Exception as e:
        log.warning(f"pdfplumber failed on {filename}: {e}")

    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(f"[Page {i+1}]\n{page_text}")
            text = "\n\n".join(pages)
        except Exception as e:
            log.warning(f"pypdf failed on {filename}: {e}")

    return text.strip()


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def chunk_text(text, chunk_size=800, overlap=150):
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        bp = text.rfind('\n\n', start, end)
        if bp <= start:
            bp = text.rfind('. ', start, end)
        if bp <= start:
            bp = end
        chunk = text[start:bp].strip()
        if chunk:
            chunks.append(chunk)
        start = bp - overlap
    return [c for c in chunks if len(c) > 50]


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def sync_drive():
    global _sync_status, _indexed_files

    _sync_status = {"state": "syncing", "message": "Connecting to Google Drive...", "indexed": 0, "total": 0}

    try:
        service = get_drive_service()
        files = list_pdfs_in_folder(service, DRIVE_FOLDER_ID)

        if not files:
            _sync_status = {"state": "done", "message": "No PDFs found in Drive folder.", "indexed": 0, "total": 0}
            return

        new_or_updated = [
            f for f in files
            if f["id"] not in _indexed_files or _indexed_files[f["id"]] != f["modifiedTime"]
        ]

        _sync_status["total"] = len(new_or_updated)
        _sync_status["message"] = f"Found {len(new_or_updated)} PDFs to index..."
        log.info(_sync_status["message"])

        indexed = 0
        for file in new_or_updated:
            try:
                _sync_status["message"] = f"Indexing: {file['name']}"
                log.info(f"Indexing: {file['name']}")

                pdf_bytes = download_pdf(service, file["id"])
                raw_text = extract_text(pdf_bytes, file["name"])

                if not raw_text:
                    log.warning(f"No text from {file['name']} — may be scanned")
                    continue

                text = clean_text(raw_text)
                chunks = chunk_text(text)

                if not chunks:
                    continue

                remove_file_chunks(file["id"])

                for i, chunk in enumerate(chunks):
                    add_document(chunk, file["name"], file["id"], i)

                _indexed_files[file["id"]] = file["modifiedTime"]
                indexed += 1
                _sync_status["indexed"] = indexed
                log.info(f"Indexed {file['name']} — {len(chunks)} chunks")

            except Exception as e:
                log.error(f"Failed {file['name']}: {e}")

        _sync_status = {
            "state": "done",
            "message": f"Ready — {len(_documents):,} chunks from {len(_indexed_files)} PDFs",
            "indexed": indexed,
            "total": len(new_or_updated)
        }
        log.info(_sync_status["message"])

    except Exception as e:
        _sync_status = {"state": "error", "message": str(e), "indexed": 0, "total": 0}
        log.error(f"Sync failed: {e}")


def background_sync_loop():
    while True:
        sync_drive()
        time.sleep(SYNC_INTERVAL)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def answer_query(query, history):
    chunks = search(query)

    if not chunks:
        return {"answer": "No relevant content found for that question. Make sure your PDFs are uploaded to the Align KB folder in Google Drive and the sync has run.", "sources": []}

    context = "\n\n".join([
        f"--- SOURCE: {c['source']} ---\n{c['text']}"
        for c in chunks
    ])

    messages = []
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({
        "role": "user",
        "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"
    })

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    sources = list({c["source"] for c in chunks})
    return {"answer": response.content[0].text, "sources": sources}


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/query", methods=["POST"])
def query():
    data = request.json or {}
    user_query = (data.get("query") or "").strip()
    history = data.get("history") or []
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    try:
        result = answer_query(user_query, history)
        return jsonify(result)
    except Exception as e:
        log.error(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    return jsonify({
        "chunks": len(_documents),
        "files": len(_indexed_files),
        "sync": _sync_status
    })


@app.route("/api/sync", methods=["POST"])
def manual_sync():
    thread = threading.Thread(target=sync_drive, daemon=True)
    thread.start()
    return jsonify({"message": "Sync started"})


if __name__ == "__main__":
    sync_thread = threading.Thread(target=background_sync_loop, daemon=True)
    sync_thread.start()
    log.info(f"Starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)

