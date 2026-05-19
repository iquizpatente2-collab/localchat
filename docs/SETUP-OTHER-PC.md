# Set up Localchat on another PC (Windows + Docker)

## What you need on the new PC

1. **Docker Desktop** — installed and running  
2. **Git** — to clone the repo  
3. **Ollama** — only if AI runs on *this* PC (skip if you use a remote DGX server)  
4. **Your manual PDF** — one file (not in Git)

Repo: `https://github.com/iquizpatente2-collab/localchat.git`

---

## Quick setup (recommended)

Open **PowerShell** and run:

```powershell
cd C:\
git clone https://github.com/iquizpatente2-collab/localchat.git
cd localchat

# Copy your PDF (change path to your file)
mkdir docs -Force
copy "D:\path\to\your-manual.pdf" docs\

# Automated setup (local Ollama on this PC)
.\scripts\setup-other-pc.ps1
```

If Ollama is on a **remote server** (DGX), edit `env.dgx.example` (copy to `.env.dgx`) with the server IP, then:

```powershell
.\scripts\setup-other-pc.ps1 -Mode dgx
```

---

## Ollama on this PC (first time only)

Install Ollama from https://ollama.com, then:

```powershell
ollama pull nomic-embed-text
ollama pull qwen3.5:9b
```

Keep Ollama running while you use Localchat.

---

## Use the app

1. Open **http://127.0.0.1:8082**  
2. **Admin login:** username `admin`, password `admin`  
3. Open **Gestisci note community** / PDF section → click **Usa PDF docs** (indexes `docs\` PDF)  
4. Wait until status shows index ready  
5. Ask questions in the chat box  

---

## Useful commands

| Task | Command |
|------|---------|
| Start | `docker compose up -d` |
| Rebuild after `git pull` | `docker compose up --build -d` |
| Stop | `docker compose down` |
| Switch local / DGX Ollama | `.\scripts\switch-mode.ps1 -Mode local` or `-Mode dgx` |
| Check health | `curl http://127.0.0.1:8082/api/health` |

---

## Problems

| Symptom | Fix |
|---------|-----|
| Chat fails | Start Ollama; check `OLLAMA_HOST` in `.env` |
| No index | Admin → **Use docs PDF** after PDF is in `docs\` |
| `host.docker.internal` fails on Linux | Use host IP in `.env`: `OLLAMA_HOST=http://192.168.x.x:11434` |

No code changes are needed when you change the PDF filename — any `.pdf` in `docs\` works.
