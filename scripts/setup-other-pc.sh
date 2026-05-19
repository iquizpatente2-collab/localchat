#!/usr/bin/env bash
# First-time setup on a new Linux machine (Docker + Localchat).
#   chmod +x scripts/setup-other-pc.sh
#   ./scripts/setup-other-pc.sh
#   ./scripts/setup-other-pc.sh dgx

set -euo pipefail

MODE="${1:-local}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "Localchat setup — repo: $REPO"

command -v docker >/dev/null 2>&1 || { echo "Install Docker first: https://docs.docker.com/engine/install/"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker not running. Start docker service, then retry."; exit 1; }

mkdir -p docs data

if [[ "$MODE" == "dgx" ]]; then
  ENV_EXAMPLE="env.dgx.example"
else
  ENV_EXAMPLE="env.local.example"
fi

[[ -f "$ENV_EXAMPLE" ]] || { echo "Missing $ENV_EXAMPLE"; exit 1; }

if [[ ! -f .env ]]; then
  cp "$ENV_EXAMPLE" .env
  echo "Created .env from $ENV_EXAMPLE"
else
  echo "Using existing .env"
fi

[[ -f .env.local ]] || [[ ! -f env.local.example ]] || cp env.local.example .env.local
[[ -f .env.dgx ]] || [[ ! -f env.dgx.example ]] || cp env.dgx.example .env.dgx

if compgen -G "docs/*.pdf" >/dev/null; then
  echo "Found PDF in docs/: $(ls -1 docs/*.pdf | head -1)"
else
  echo "No PDF in docs/ yet — copy your manual to docs/ before indexing."
fi

if [[ "$MODE" == "local" ]]; then
  if command -v ollama >/dev/null 2>&1; then
    echo "Ollama models (local mode):"
    ollama list || true
    echo "If missing: ollama pull nomic-embed-text && ollama pull qwen3.5:9b"
  else
    echo "Ollama not installed — install it or run: ./scripts/setup-other-pc.sh dgx"
  fi
else
  echo "DGX mode: set OLLAMA_HOST in .env to your server IP."
fi

echo "Building and starting Docker…"
docker compose up --build -d

sleep 4
echo "Health check:"
curl -fsS http://127.0.0.1:8082/api/health || echo "Wait ~30s, then open http://127.0.0.1:8082"

cat <<'EOF'

=== Next steps ===
1. Open  http://127.0.0.1:8082
2. Admin login:  admin / admin
3. Put PDF in docs/ if needed
4. Click  Use docs PDF  and wait for indexing
5. Ask a question in the chat

Switch Ollama:  ./scripts/switch-mode.sh local|dgx
EOF
