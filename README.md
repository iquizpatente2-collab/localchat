# Localchat: Local PDF Recipe Intelligence

Localchat is a local web RAG app for old recipe PDFs.  
It is tuned for fast retrieval, typo tolerance, and grounded (source-faithful) answers.

---

## Current Project Scope

This repo runs a **web-first** workflow:

- Backend API: `web/app.py` (FastAPI)
- Frontend: `web/static/index.html`
- Retrieval modules: `web/rag/*`

Legacy voice-assistant files were removed from this project scope.

---

## Models We Use

Default Ollama models:

- Embeddings: `nomic-embed-text`
- Chat model: `qwen3.5:9b`
- Optional fallback: `qwen2.5:7b-instruct`

Configurable via:

- `OLLAMA_HOST`
- `OLLAMA_EMBED_MODEL`
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_CHAT_FALLBACK`

Inference stays on your machine (no cloud LLM API in this app). Browser **voice** (SpeechRecognition) may still use the browser vendor’s service unless you type instead.

---

## Web UI (`index.html`)

### Layout

- **Desktop / wide screens (about 900px+):** Chat on the left, **composer** (ask + PDF tools) in a **sticky side column**.
- **Tablet / narrow:** Stacked layout with a compact toolbar for PDF actions.
- **Phone (≤720px):** **Chat-first** layout: the page does not scroll as one long document; **only the conversation panel scrolls** so answers stay readable. The bottom **ask/search** block can be **collapsed** with **Hide search ▼** / **Show search ▲** to read long recipes.

### Chat

- **Conversation history** with user and assistant **bubbles**.
- **Thinking** animation while waiting for a reply.
- **Source chips** (page · score) under answers; recipe mode shows match titles where available.
- **Enter** sends; **Shift+Enter** new line (desktop hint; mobile shows “Tap Ask to send”).
- **Clear conversation** clears the on-screen thread only.

### PDF & index (same features everywhere)

- Upload + **Ingest manual**, **Use docs PDF** (rebuild from `docs/`), **Remove uploaded PDF** (`DELETE /api/upload`).
- On mobile, **PDF & index** starts **collapsed** in a `<details>` drawer; expand to use it.

### Voice (STT)

- **Start voice** / **Stop voice** (two clicks), languages **English** and **Italian**.

---

## Web Page Flow

### On page load

1. `GET /api/health`
2. `GET /api/status`
3. UI shows index / models / recipe catalog state

### On PDF upload

1. `POST /api/upload`
2. Backend extracts pages, builds `data/rag_store/` and `data/recipe_store/`
3. UI refreshes status

### Ask

- **Recipe checkbox OFF** → `POST /api/chat` (chunk RAG + sources).
- **Recipe checkbox ON** → `POST /api/recipe-chat` (hybrid retrieval + grounded or LLM mode).

---

## Retrieval Stack

`Query → Hybrid retrieval → Top candidates → Grounded output`

Hybrid ranking uses fuzzy matching (`rapidfuzz`), embeddings (FAISS or NumPy), token coverage, phrase-aware scoring, and small **intent bridges** (e.g. tomato sauce / fried chicken) where configured in `web/rag/recipe_catalog.py`.

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Ollama reachability |
| GET | `/api/status` | Chunks, models, recipe catalog meta |
| POST | `/api/upload` | Upload PDF → index |
| DELETE | `/api/upload` | Remove `data/manuals/current_manual.pdf` |
| POST | `/api/use-docs` | Clear indexes, rebuild from `docs/*.pdf` |
| POST | `/api/chat` | Manual RAG chat |
| POST | `/api/recipes/rank` | Raw hybrid recipe ranks (debug) |
| POST | `/api/recipe-chat` | Recipe hybrid + answer |

---

## Run

From repo root:

```bash
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8080
```

Open on the same machine:

- http://127.0.0.1:8080

Use **`--host 0.0.0.0`** if you want other devices on your **LAN** to open the app (e.g. phone on Wi‑Fi).

### Open from a phone on the same Wi‑Fi

1. On the PC, run `ipconfig` and note the **Wi‑Fi IPv4** (e.g. `10.x.x.x`).
2. On the phone: `http://<that-ip>:8080`
3. If the phone or `http://<ip>:8080` on the PC says the site cannot be reached, allow **inbound TCP 8080** (or Python) in **Windows Defender Firewall** for **Private** networks. Run **PowerShell or CMD as Administrator**, then:

```text
netsh advfirewall firewall add rule name="Localchat 8080" dir=in action=allow protocol=TCP localport=8080 profile=private,domain
```

Recommended Ollama pulls:

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:9b
```

---

## Useful Environment Flags

- `RAG_AUTO_DOCS`
- `RAG_DOCS_FILE`
- `RAG_RECIPE_NORMALIZE` (expensive if enabled)
- `RAG_RECIPE_NORMALIZE_MODE`
- `RAG_RECIPE_CONCURRENCY`
- `RAG_EMBED_CONCURRENCY`
- `RECIPE_W_EMBED`
- `RECIPE_W_FUZZY`
- `RECIPE_TOP_K`
- `OLLAMA_NUM_PARALLEL` (Ollama-side concurrency)

---

## Example Queries

- `what is salsa di pomidoro`
- `how to make tomato sauce`
- `how to prepare mutton cutlets`
- `filet of veal recipe`
- `eggs with tomatoes`
- `sauce for broiled fish`
- `risotto con pecorino`

---

## Troubleshooting

- **No recipe catalog**  
  Upload a PDF again or rebuild `data/rag_store` + `data/recipe_store` (e.g. **Use docs PDF**).

- **Slow startup**  
  Keep `RAG_RECIPE_NORMALIZE=0` unless you need LLM page normalization.

- **Embedding mismatch**  
  Rebuild indexes after changing `OLLAMA_EMBED_MODEL`.

- **Wrong recipe match**  
  Re-ingest after retrieval tweaks; use recipe mode for dish-style questions.

- **LAN / phone cannot connect**  
  Firewall rule above; same Wi‑Fi; use `http` not `https`.

---

## License

MIT
