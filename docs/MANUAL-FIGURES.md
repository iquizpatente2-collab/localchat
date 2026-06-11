# Manual figures (Phase A)

Topic-linked images from PDFs are extracted at ingest and shown in chat when relevant chunks are retrieved.

## How it works

1. **Ingest** (`Use docs PDF` / upload): PyMuPDF extracts embedded images per page.
2. **Topic link**: Each figure gets a `topic_id` from the nearest section heading above it on the page (layout bbox).
3. **Chunks**: RAG chunks store `topic_id` and `image_ids` (matched by page + section).
4. **Chat**: Retrieved chunks resolve to figure URLs; the UI shows up to 6 thumbnails under the answer.

## Storage

- `data/manual_assets/catalog.json` — figure metadata
- `data/manual_assets/files/*.png` — extracted images

## Config

| Env | Default | Meaning |
|-----|---------|---------|
| `RAG_EXTRACT_FIGURES` | `1` | Set `0` to disable figure extraction |

## Re-ingest required

Existing indexes built before this feature do not include figures. Admin → **Use docs PDF** once to rebuild text + figure index.

## API

- `GET /api/manual-asset/{id}` — serve PNG
- Chat/stream `done` payload includes `images: [{ id, page, section, caption, url, ... }]`

## Next phases (not implemented yet)

- GLM-OCR / Chandra text inside figures
- Vision captions for semantic image search
- OCR for scanned pages pdfplumber marks empty
