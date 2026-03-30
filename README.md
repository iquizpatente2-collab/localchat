# Localchat: Local PDF Recipe Intelligence

Localchat is a local web RAG app for old recipe PDFs.  
It is tuned for fast retrieval, typo tolerance, and grounded (source-faithful) answers.

---

## Current Project Scope

This repo now runs a **web-first** workflow:

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

---

## Web Page Process (`index.html`)

### On page load

1. `GET /api/health`
2. `GET /api/status`
3. UI displays index/model/catalog state

### On PDF upload

1. Browser sends `POST /api/upload`
2. Backend extracts and cleans pages
3. Builds indexes:
   - `data/rag_store/` (chunk index)
   - `data/recipe_store/` (recipe index)
4. UI refreshes status

### On Ask button

- **Recipe checkbox OFF** -> `POST /api/chat`  
  Manual chunk RAG answer with source pages.

- **Recipe checkbox ON** -> `POST /api/recipe-chat`  
  Hybrid recipe retrieval + grounded response with match metadata.

### Voice STT button

- Click 1: `Start voice` (record)
- Click 2: `Stop voice` (transcribe into textarea)
- Language selector:
  - English (`en-US`)
  - Italian (`it-IT`)

---

## Retrieval Stack

Pipeline:

`Query -> Hybrid Retrieval -> Top Candidates -> Grounded Output`

Hybrid ranking uses:

- fuzzy similarity (`rapidfuzz`)
- semantic similarity (embeddings)
- token coverage (intent alignment)

This helps handle OCR noise and typos while staying aligned to the intended dish.

---

## API Endpoints

- `GET /api/health`
- `GET /api/status`
- `POST /api/upload`
- `POST /api/chat`
- `POST /api/recipes/rank`
- `POST /api/recipe-chat`

---

## Run

From repo root:

```bash
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8080
```

Open:

- http://127.0.0.1:8080

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
  - Upload PDF again or rebuild `data/rag_store` + `data/recipe_store`.

- **Slow startup**
  - Keep `RAG_RECIPE_NORMALIZE=0` unless you explicitly need normalization.

- **Embedding mismatch**
  - Rebuild indexes after changing `OLLAMA_EMBED_MODEL`.

- **Wrong fast match**
  - Re-ingest after retrieval updates; keep recipe mode enabled for dish-focused queries.

---

## License

MIT
