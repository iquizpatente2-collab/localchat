# Set up Localchat on another machine

Repo: `https://github.com/iquizpatente2-collab/localchat.git`

You need: **Docker**, **Git**, your **PDF** in `docs/`, and **Ollama reachable** (on this machine or a remote DGX — not inside the Localchat image).

---

## Linux — same as Windows (Ollama on this PC + Localchat in Docker)

This matches **Windows `lc-local`**: Ollama runs on the machine; Docker only runs Localchat.

```bash
cd ~/localchat
git pull origin main
chmod +x scripts/setup-linux-full.sh
./scripts/setup-linux-full.sh
```

That script installs Ollama (if needed), pulls `nomic-embed-text` + chat models, writes `.env`, and starts Docker.

Custom models:

```bash
CHAT_MODEL=qwen3.6:latest EMBED_MODEL=nomic-embed-text ./scripts/setup-linux-full.sh
```

Then open **http://127.0.0.1:8082** → admin **admin** / **admin** → **Use docs PDF**.

---

## Linux (Ubuntu / Debian / etc.) — manual steps

### 1. Install Docker

```bash
# Example (Docker’s official guide may differ by distro):
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git curl
sudo usermod -aG docker "$USER"
# Log out and back in so group docker applies
```

### 2. Clone and add PDF

```bash
cd ~
git clone https://github.com/iquizpatente2-collab/localchat.git
cd localchat
mkdir -p docs
cp /path/to/your-manual.pdf docs/
chmod +x scripts/setup-other-pc.sh scripts/switch-mode.sh
```

### 3. Setup

**Ollama on this Linux PC:**

```bash
# Install Ollama: https://ollama.com/download/linux
ollama pull nomic-embed-text
ollama pull qwen3.5:9b
./scripts/setup-other-pc.sh
```

**Ollama on remote DGX only (no Ollama on this PC):**

```bash
cp env.dgx.example .env.dgx
nano .env.dgx   # set OLLAMA_HOST=http://YOUR-DGX-IP:11434
cp .env.dgx .env
./scripts/setup-other-pc.sh dgx
```

### 4. Use the app

1. Open **http://127.0.0.1:8082** (or `http://<server-ip>:8082` from another browser on the LAN).
2. Admin login: **admin** / **admin**
3. **Use docs PDF** (indexes `docs/*.pdf`)
4. Ask questions in chat

### Linux commands

| Task | Command |
|------|---------|
| Start | `docker compose up -d` |
| Rebuild after `git pull` | `docker compose up --build -d` |
| Stop | `docker compose down` |
| Switch Ollama target | `./scripts/switch-mode.sh local` or `dgx` |
| Health | `curl http://127.0.0.1:8082/api/health` |

### Linux troubleshooting

| Symptom | Fix |
|---------|-----|
| Health shows `0.0.0.0:11434` | Your shell has `OLLAMA_HOST=0.0.0.0:11434` (Ollama **bind**). Run `unset OLLAMA_HOST` then `env -u OLLAMA_HOST docker compose up -d`. Use `http://172.17.0.1:11434` in `.env`, not `0.0.0.0`. |
| `172.17.0.1` connection refused | Ollama not on docker bridge. Run `ss -tlnp \| grep 11434` — use that IP in `.env`, e.g. `OLLAMA_HOST=http://192.168.0.99:11434` |
| `host.docker.internal` fails | Set in `.env`: `OLLAMA_HOST=http://172.17.0.1:11434` or your host’s LAN IP from `ss` |
| PDF upload / **Use docs PDF** fails with HTTP 503 | Ollama queue full. In `.env`: `RAG_EMBED_CONCURRENCY=1`, `RAG_BUILD_RECIPE_INDEX=0`. On Ollama host: `OLLAMA_NUM_PARALLEL=2`, `OLLAMA_MAX_QUEUE=1024`. Rebuild Docker (`docker compose up --build -d`), click **Use docs PDF** once, wait 30–60 min. |
| PDF upload / **Use docs PDF** fails (other) | Usually Ollama unreachable from container — fix health first (`curl http://127.0.0.1:8082/api/health` → `"ok":true`) |
| No PDF in UI | Check `ls docs/*.pdf` on host; repo includes `docs/manuale_uso.pdf`. Click **Use docs PDF** (admin), not only upload. |
| Permission denied on Docker | `sudo usermod -aG docker $USER` and re-login |
| Chat fails | Ollama running; `ss -tlnp \| grep 11434` shows `0.0.0.0:11434` or `*:11434` |
| Remote DGX | `OLLAMA_HOST=http://dgx-ip:port`; Ollama must listen on `0.0.0.0`, not only `127.0.0.1` |

---

## Windows

### Quick setup

```powershell
git clone https://github.com/iquizpatente2-collab/localchat.git
cd localchat
mkdir docs -Force
copy "D:\path\to\your-manual.pdf" docs\
.\scripts\setup-other-pc.ps1
```

Remote DGX:

```powershell
.\scripts\setup-other-pc.ps1 -Mode dgx
```

### Ollama on Windows (local)

```powershell
ollama pull nomic-embed-text
ollama pull qwen3.5:9b
```

### Windows commands

| Task | Command |
|------|---------|
| Switch Ollama | `.\scripts\switch-mode.ps1 -Mode local` or `dgx` |
| Health | `curl http://127.0.0.1:8082/api/health` |

---

## PDF in `docs/`

No code changes for the filename. Put any `.pdf` in `docs/`, then admin → **Use docs PDF**. If several PDFs exist, the **newest** file is used (or set `RAG_DOCS_FILE=docs/name.pdf` in `.env`).
