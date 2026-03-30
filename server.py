"""
Align Builders Knowledge Base — Cloud Server v4
- No ML models, no ChromaDB
- BM25 in-memory search
- Caps each PDF at 60 pages
- Frees memory between each PDF with gc.collect()
"""

import gc
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

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
GOOGLE_CREDENTIALS = os.environ.get("GOOGLE_CREDENTIALS", "")
DRIVE_FOLDER_ID    = os.environ.get("DRIVE_FOLDER_ID", "1BfpoW2xFO1byMc1AH5JIiplozYZr7LmU")
CLAUDE_MODEL       = "claude-sonnet-4-20250514"
TOP_K              = 6
PORT               = int(os.environ.get("PORT", 8000))
SYNC_INTERVAL      = 3600
MAX_PAGES          = 60

SYSTEM_PROMPT = """You are a knowledgeable assistant for Align Builders, a commercial construction company.
You have access to a library of internal proposal documents, project narratives, and RFQ/RFP submittals.
Answer questions using only the provided context. Be specific. Mention which document your answer
comes from when relevant. If the context does not contain enough to answer confidently, say so clearly.
Write in clean, direct prose. No filler, no hedging."""

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("align-kb")

_documents = []
_indexed_files = {}
_sync_status = {"state": "idle", "message": "Not started", "indexed": 0, "total": 0}
_lock = threading.Lock()


def tokenize(text):
    text = text.lower()
    words = re.findall(r'\b[a-z][a-z0-9]{2,}\b', text)
    stops = {'the','and','for','are','was','were','this','that','with','have',
             'from','they','will','been','their','said','each','which','there',
             'what','about','would','make','like','into','than','them','some',
             'these','could','other','more','also','but','not','can','its','you',
             'all','has','her','his','him','she','who','our','your','may','any'}
    return [w for w in words if w not in stops]


def bm25_score(query_tokens, doc_tokens, avg_dl, k1=1.5, b=0.75):
    if not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    doc_freq = Counter(doc_tokens)
    score = 0.0
    n = max(len(_documents), 1)
    for term in query_tokens:
        if term in doc_freq:
            tf = doc_freq[term]
            df = sum(1 for d in _documents if term in d['token_set'])
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


def search(query, top_k=TOP_K):
    with _lock:
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
        return [{"text": d["text"], "source": d["source"], "relevance": round(s, 3)}
                for s, d in scored[:top_k]]


def add_chunks(chunks, source, file_id):
    with _lock:
        global _documents
        _documents = [d for d in _documents if d['file_id'] != file_id]
        for i, chunk in enumerate(chunks):
            tokens = tokenize(chunk)
            _documents.append({
                "id": f"{file_id}_{i}",
                "text": chunk,
                "source": source,
                "file_id": file_id,
                "chunk_index": i,
                "tokens": tokens,
                "token_set": set(tokens)
            })


def get_drive_service():
    if GOOGLE_CREDENTIALS:
        creds_dict = json.loads(GOOGLE_CREDENTIALS)
    else:
        with open("align-knowledge-base-ad25c3204d55.json") as f:
            creds_dict = json.load(f)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/drive.readonly"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_pdfs(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
        fields="files(id, name, modifiedTime)",
        pageSize=200
    ).execute()
    return results.get("files", [])


def download_pdf(service, file_id):
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue()


def extract_text(pdf_bytes, filename):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for i, page in enumerate(pdf.pages[:MAX_PAGES]):
                pt = page.extract_text() or ""
                if pt.strip():
                    pages.append(f"[Page {i+1}]\n{pt}")
            text = "\n\n".join(pages)
    except Exception as e:
        log.warning(f"pdfplumber failed on {filename}: {e}")
    if not text.strip():
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for i, page in enumerate(reader.pages[:MAX_PAGES]):
                pt = page.extract_text() or ""
                if pt.strip():
                    pages.append(f"[Page {i+1}]\n{pt}")
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


def sync_drive():
    global _sync_status, _indexed_files
    _sync_status = {"state": "syncing", "message": "Connecting to Google Drive...", "indexed": 0, "total": 0}
    try:
        service = get_drive_service()
        files = list_pdfs(service, DRIVE_FOLDER_ID)
        if not files:
            _sync_status = {"state": "done", "message": "No PDFs found in Drive folder.", "indexed": 0, "total": 0}
            return

        new_or_updated = [f for f in files
            if f["id"] not in _indexed_files or _indexed_files[f["id"]] != f["modifiedTime"]]

        if not new_or_updated:
            _sync_status = {"state": "done",
                "message": f"Up to date — {len(_documents):,} chunks from {len(_indexed_files)} PDFs",
                "indexed": 0, "total": 0}
            return

        _sync_status["total"] = len(new_or_updated)
        indexed = 0

        for file in new_or_updated:
            try:
                _sync_status["message"] = f"Indexing {indexed+1}/{len(new_or_updated)}: {file['name']}"
                log.info(f"Indexing: {file['name']}")

                pdf_bytes = download_pdf(service, file["id"])
                raw_text = extract_text(pdf_bytes, file["name"])
                del pdf_bytes
                gc.collect()

                if raw_text:
                    chunks = chunk_text(clean_text(raw_text))
                    del raw_text
                    gc.collect()
                    if chunks:
                        add_chunks(chunks, file["name"], file["id"])
                        log.info(f"Done: {file['name']} — {len(chunks)} chunks")
                else:
                    log.warning(f"No text extracted: {file['name']}")

                _indexed_files[file["id"]] = file["modifiedTime"]
                indexed += 1
                _sync_status["indexed"] = indexed
                time.sleep(2)
                gc.collect()

            except Exception as e:
                log.error(f"Failed {file['name']}: {e}")
                gc.collect()
                time.sleep(2)

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
    time.sleep(5)
    while True:
        sync_drive()
        time.sleep(SYNC_INTERVAL)


def answer_query(query, history):
    chunks = search(query)
    if not chunks:
        return {"answer": "No relevant content found. Make sure your PDFs are in the Align KB Google Drive folder and click Sync Drive.", "sources": []}
    context = "\n\n".join([f"--- SOURCE: {c['source']} ---\n{c['text']}" for c in chunks])
    messages = []
    for turn in history[-6:]:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"})
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(model=CLAUDE_MODEL, max_tokens=2048, system=SYSTEM_PROMPT, messages=messages)
    sources = list({c["source"] for c in chunks})
    return {"answer": response.content[0].text, "sources": sources}


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
        return jsonify(answer_query(user_query, history))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    return jsonify({"chunks": len(_documents), "files": len(_indexed_files), "sync": _sync_status})


@app.route("/api/sync", methods=["POST"])
def manual_sync():
    if _sync_status.get("state") == "syncing":
        return jsonify({"message": "Sync already running"})
    threading.Thread(target=sync_drive, daemon=True).start()
    return jsonify({"message": "Sync started"})


if __name__ == "__main__":
    threading.Thread(target=background_sync_loop, daemon=True).start()
    log.info(f"Starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
