# Localchat: Local Manual Q&A (RAG)

Localchat is a local, web-first RAG app for querying PDF manuals with source-grounded answers.
It supports multilingual UI (English/Italian), theme switching, model switching, streaming chat, voice input, and community field notes.

## What This Repo Contains

- Backend API: `web/app.py` (FastAPI)
- Frontend UI: `web/static/index.html` + `web/static/themes.css`
- Retrieval and ranking logic: `web/rag/*`
- Docker runtime files: `Dockerfile`, `docker-compose.yml`

## Current Runtime Model Stack

By default this project uses Ollama as the model-serving backend.

- Chat model (default in compose): `qwen3.5:9b`
- Embedding model: `nomic-embed-text`
- Optional fallback model: `qwen3.5:27b`

Important: the app container does not include an Ollama server. It expects Ollama API to be reachable at `OLLAMA_HOST` (default `http://host.docker.internal:11434`).

## Features

### Core Q&A

- PDF ingestion and reindexing
- Streaming answer generation (`/api/chat-stream`)
- Source chips with page/score metadata
- Click-to-open manual page source links
- Conversation history for context-aware follow-up

### UI / UX

- Responsive layout (desktop, tablet, mobile)
- Collapsible composer on mobile
- Theme switcher:
  - Light
  - Dark
  - Blueprint (industrial style)
- Language switcher:
  - English
  - Italian
- Model picker in composer
- Current chat model indicator in chat toolbar

### Voice

- Browser speech-to-text (EN/IT)
- Whisper recording + server transcription (`/api/transcribe`) when available

### Community Notes

- Save field notes against answers
- List saved notes
- Delete saved notes
- Community relevance signals in response metadata

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Basic backend and Ollama connectivity check |
| GET | `/api/status` | Index, model, and feature status for UI |
| GET | `/api/models` | List configured/discovered chat model options |
| GET | `/api/manual` | Open active manual PDF |
| POST | `/api/transcribe` | Whisper transcription from uploaded audio |
| POST | `/api/upload` | Upload and index a PDF manual |
| DELETE | `/api/upload` | Remove uploaded manual |
| POST | `/api/use-docs` | Rebuild index from `docs/` PDFs |
| POST | `/api/chat` | Non-streaming chat response |
| POST | `/api/chat-stream` | Streaming chat response |
| POST | `/api/community-save` | Save a community note |
| GET | `/api/community` | List community notes |
| DELETE | `/api/community/{tip_id}` | Delete one community note |
| POST | `/api/recipes/rank` | Raw recipe ranking debug output |
| POST | `/api/recipe-progress` | Recipe progress pipeline endpoint |
| POST | `/api/recipe-chat` | Recipe-specific chat endpoint |

## Run Locally (Python)

From repo root:

```bash
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8080
```

Then open:

- [http://127.0.0.1:8080](http://127.0.0.1:8080)

## Run with Docker (Current Project Default)

From repo root:

```bash
docker compose up --build -d
```

Then open:

- [http://127.0.0.1:8082](http://127.0.0.1:8082)

Stop:

```bash
docker compose down
```

## Ollama Requirements

This project currently expects an external Ollama API.

1. Start Ollama on host machine
2. Pull required models
3. Keep Ollama backend running while using Localchat

Example model pulls:

```bash
ollama pull nomic-embed-text
ollama pull qwen3.5:9b
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:14b-instruct
```

Quick check:

```bash
ollama list
```

If `ollama list` works, backend is reachable.

## Environment Variables (Most Useful)

### Model / Backend

- `OLLAMA_HOST` (default: `http://host.docker.internal:11434` in Docker compose)
- `OLLAMA_CHAT_MODEL`
- `OLLAMA_EMBED_MODEL`
- `OLLAMA_CHAT_FALLBACK`
- `RAG_CHAT_TIMEOUT_S`

### RAG Behavior

- `RAG_MANUAL_FRONT_MODE`
- `RAG_AUTO_DOCS`
- `RAG_DOCS_FILE`
- `RAG_RECIPE_NORMALIZE`
- `RAG_RECIPE_NORMALIZE_MODE`
- `RAG_RECIPE_CONCURRENCY`
- `RAG_EMBED_CONCURRENCY`
- `RECIPE_W_EMBED`
- `RECIPE_W_FUZZY`
- `RECIPE_TOP_K`

### Voice / Community (runtime-dependent)

- Whisper availability depends on installed transcription backend and server capability
- Community features depend on backend status and storage state

## UI Notes

- Theme and language selections are persisted in browser local storage.
- Model switch affects chat calls immediately.
- Empty chat placeholders are localized and theme-aware.
- In Blueprint mode, animated watermark gears are rendered as background decoration.

## Common Troubleshooting

- **UI loads but chat fails**  
  Ollama backend is not reachable. Start Ollama and confirm with `ollama list`.

- **Model missing from selector**  
  Pull the model in Ollama and refresh.

- **Whisper button disabled**  
  Whisper backend/media support unavailable in current environment.

- **Sources look stale after model/embed changes**  
  Re-upload manual or run **Use docs PDF** to rebuild indexes.

- **Phone/LAN cannot connect**  
  Open correct host port and allow firewall inbound rule.

## Roadmap Direction

Planned/ongoing exploration includes optional OpenAI-compatible backend support (e.g., vLLM) in a separate project copy while keeping this repository stable on Ollama.

## License

MIT
