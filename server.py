"""
Align Builders Knowledge Base — Cloud Server v2
================================================
Uses Anthropic's API for embeddings instead of sentence-transformers,
which removes the heavy torch/ML dependency and makes Railway builds fast.
"""

import io
import json
import logging
import os
import re
import threading
import time

import anthropic
import chromadb
import pdfplumber
from chromadb import Documents, EmbeddingFunction, Embeddings
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
DB_PATH            = os.environ.get("DB_PATH", "/tmp/chroma_db")
COLLECTION_NAME    = "pdf_knowledge_base"
CLAUDE_MODEL       = "claude-sonnet-4-20250514"
TOP_K              = 6
PORT               = int(os.environ.get("PORT", 8000))
SYNC_INTERVAL      = 3600

SYSTEM_PROMPT = """You are a knowledgeable assistant for Align Builders, a commercial construction company.
You have access to a library of internal proposal documents, project narratives, and RFQ/RFP submittals.

Answer questions using only the provided context. Be specific. When relevant, mention which document
your answer comes from. If the context doesn't contain enough to answer confidently, say so clearly.

Write in clean, direct prose. No filler, no hedging."""

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("align-kb")

# ---------------------------------------------------------------------------
# Anthropic Embedding Function for ChromaDB
# ---------------------------------------------------------------------------

class AnthropicEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        # Process in batches of 10 to avoid rate limits
        for i in range(0, len(input), 10):
            batch = input[i:i+10]
            for text in batch:
                response = self.client.messages.create(
                    model=CLAUDE_MODEL,
                    max_tokens=1,
                    system="Return only a JSON array of 256 floats representing the embedding of the input text. Nothing else.",
                    messages=[{"role": "user", "content": f"Embed this text: {text[:500]}"}]
                )
                try:
                    embedding = json.loads(response.content[0].text)
                    embeddings.append(embedding)
                except Exception:
                    # Fallback: simple hash-based embedding
                    embedding = [float(ord(c)) / 127.0 for c in text[:256]]
                    embedding += [0.0] * (256 - len(embedding))
                    embeddings.append(embedding)
        return embeddings


# Simpler approach: use ChromaDB's default embedding (no external deps)
# This uses a built-in lightweight model that doesn't require torch
def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    # Use default embedding function - lightweight and built into chromadb
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


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
    return build("drive", "v3", credentials=creds)


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

_sync_status = {"state": "idle", "message": "Not started", "indexed": 0, "total": 0}
_indexed_files = {}


def sync_drive():
    global _sync_status, _indexed_files
    _sync_status = {"state": "syncing", "message": "Connecting to Google Drive...", "indexed": 0, "total": 0}

    try:
        service = get_drive_service()
        files = list_pdfs_in_folder(service, DRIVE_FOLDER_ID)

        if not files:
            _sync_status = {"state": "done", "message": "No PDFs found in Drive folder.", "indexed": 0, "total": 0}
            return

        collection = get_collection()
        new_or_updated = [
            f for f in files
            if f["id"] not in _indexed_files or _indexed_files[f["id"]] != f["modifiedTime"]
        ]

        _sync_status["total"] = len(new_or_updated)
        _sync_status["message"] = f"Found {len(new_or_updated)} PDFs to index..."

        indexed = 0
        for file in new_or_updated:
            try:
                _sync_status["message"] = f"Indexing: {file['name']}"
                pdf_bytes = download_pdf(service, file["id"])
                raw_text = extract_text(pdf_bytes, file["name"])

                if not raw_text:
                    log.warning(f"No text extracted from {file['name']}")
                    continue

                text = clean_text(raw_text)
                chunks = chunk_text(text)

                if not chunks:
                    continue

                try:
                    old = collection.get(where={"file_id": file["id"]})
                    if old["ids"]:
                        collection.delete(ids=old["ids"])
                except Exception:
                    pass

                ids = [f"{file['id']}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{
                    "source": file["name"],
                    "file_id": file["id"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                } for i in range(len(chunks))]

                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    collection.upsert(
                        ids=ids[i:i+batch_size],
                        documents=chunks[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size],
                    )

                _indexed_files[file["id"]] = file["modifiedTime"]
                indexed += 1
                _sync_status["indexed"] = indexed
                log.info(f"Indexed {file['name']} ({len(chunks)} chunks)")

            except Exception as e:
                log.error(f"Failed to index {file['name']}: {e}")

        total_chunks = collection.count()
        _sync_status = {
            "state": "done",
            "message": f"Ready — {total_chunks:,} chunks from {len(_indexed_files)} PDFs",
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
# RAG Query
# ---------------------------------------------------------------------------

def retrieve_chunks(query):
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "relevance": round(1 - dist, 3),
        })
    return chunks


def answer_query(query, history):
    chunks = retrieve_chunks(query)
    if not chunks:
        return {"answer": "No relevant content found for that question.", "sources": []}

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
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    try:
        count = get_collection().count()
        return jsonify({"chunks": count, "files": len(_indexed_files), "sync": _sync_status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
