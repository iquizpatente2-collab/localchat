#!/usr/bin/env bash
# Full Linux deploy — same idea as Windows: Ollama on the host + Localchat in Docker.
#   chmod +x scripts/setup-linux-full.sh
#   ./scripts/setup-linux-full.sh
#
# Optional env overrides before running:
#   CHAT_MODEL=qwen3.6:latest EMBED_MODEL=nomic-embed-text ./scripts/setup-linux-full.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

CHAT_MODEL="${CHAT_MODEL:-qwen3.6:latest}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
CHAT_FALLBACK="${CHAT_FALLBACK:-gemma4:latest}"

echo "=== Localchat full setup (Linux, local Ollama) ==="
echo "Repo: $REPO"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Install: https://docs.docker.com/engine/install/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker not running. Try: sudo systemctl start docker"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Installing Ollama…"
  curl -fsSL https://ollama.com/install.sh | sh
fi

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl enable ollama 2>/dev/null || true
  sudo systemctl start ollama 2>/dev/null || true
fi

echo "Waiting for Ollama API…"
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama not responding on :11434. Start it: ollama serve"
  exit 1
fi

echo "Pulling models (this can take a long time)…"
ollama pull "$EMBED_MODEL"
ollama pull "$CHAT_MODEL"
if [[ -n "$CHAT_FALLBACK" && "$CHAT_FALLBACK" != "$CHAT_MODEL" ]]; then
  ollama pull "$CHAT_FALLBACK" || true
fi

mkdir -p docs data

# Docker bridge IP (container → host Ollama). Avoid host.docker.internal on some Linux setups.
DOCKER_HOST_IP="${DOCKER_HOST_IP:-172.17.0.1}"
if command -v ip >/dev/null 2>&1; then
  GW="$(ip route show default 2>/dev/null | awk '{print $3; exit}')"
  BR="$(ip -4 route show dev docker0 2>/dev/null | awk '/proto kernel/ {print $1; exit}')"
  if [[ -n "$BR" ]]; then
    DOCKER_HOST_IP="${BR%/*}"
  elif [[ -n "$GW" ]]; then
    DOCKER_HOST_IP="$GW"
  fi
fi

cat > .env.local <<EOF
# Client URL for Localchat container (NOT Ollama's server bind address 0.0.0.0).
OLLAMA_HOST=http://${DOCKER_HOST_IP}:11434
OLLAMA_CHAT_MODEL=$CHAT_MODEL
OLLAMA_CHAT_FALLBACK=$CHAT_FALLBACK
OLLAMA_EMBED_MODEL=$EMBED_MODEL
RAG_AUTO_DOCS=0
EOF

cp .env.local .env
echo "Wrote .env.local and .env (OLLAMA_HOST=http://${DOCKER_HOST_IP}:11434)"

if [[ -n "${OLLAMA_HOST:-}" && "${OLLAMA_HOST}" != http://* ]]; then
  echo "Note: shell OLLAMA_HOST=${OLLAMA_HOST} is Ollama's bind address — ignored for Docker (use .env)."
fi

if compgen -G "docs/*.pdf" >/dev/null; then
  echo "PDF: $(ls -1 docs/*.pdf | head -1)"
else
  echo "Note: no PDF in docs/ — git clone includes docs/manuale_uso.pdf if you pulled latest"
fi

echo "Starting Localchat (Docker)…"
# Shell OLLAMA_HOST=0.0.0.0:11434 (common on DGX) must not override .env for compose.
env -u OLLAMA_HOST docker compose up --build -d

sleep 5
echo ""
echo "Ollama on host:"
curl -fsS http://127.0.0.1:11434/api/tags | head -c 200 || true
echo ""
echo ""
echo "Localchat health:"
curl -fsS http://127.0.0.1:8082/api/health || echo "(wait a few seconds and retry)"
echo ""
cat <<EOF

=== Done (same layout as Windows lc-local) ===
  Browser:  http://127.0.0.1:8082
  Admin:    admin / admin  →  Use docs PDF
  Models:   $CHAT_MODEL  +  $EMBED_MODEL

Switch to remote DGX later:  ./scripts/switch-mode.sh dgx
EOF
