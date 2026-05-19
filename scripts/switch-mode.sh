#!/usr/bin/env bash
# Switch Ollama target and restart Localchat (Linux).
#   ./scripts/switch-mode.sh local
#   ./scripts/switch-mode.sh dgx

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "local" && "$MODE" != "dgx" ]]; then
  echo "Usage: $0 local|dgx"
  exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV_FILE=".env.${MODE}"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE in $REPO"; exit 1; }

cp "$ENV_FILE" .env
echo "Switched to $MODE mode using $ENV_FILE"

docker compose up -d --force-recreate

grep OLLAMA_HOST .env || true
echo "Health:"
curl -fsS http://127.0.0.1:8082/api/health || true
echo
