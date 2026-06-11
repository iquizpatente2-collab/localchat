# Run Localchat on Windows WITHOUT Docker (good for quick local testing).
# Requires: Ollama running on this PC, .venv with pip install -r requirements.txt
# Usage:  .\scripts\run-local-native.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Write-Host "Create venv first:  python -m venv .venv" -ForegroundColor Yellow
  Write-Host "Then:  .\.venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
  exit 1
}

$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:OLLAMA_CHAT_MODEL = if ($env:OLLAMA_CHAT_MODEL) { $env:OLLAMA_CHAT_MODEL } else { "qwen3.5:9b" }
$env:OLLAMA_CHAT_FALLBACK = if ($env:OLLAMA_CHAT_FALLBACK) { $env:OLLAMA_CHAT_FALLBACK } else { "qwen2.5:7b-instruct" }
$env:OLLAMA_EMBED_MODEL = "nomic-embed-text"
$env:RAG_EMBED_CONCURRENCY = "2"
$env:RAG_BUILD_RECIPE_INDEX = "0"
$env:RAG_EXTRACT_FIGURES = "1"

Write-Host "Localchat (native) — http://127.0.0.1:8082" -ForegroundColor Cyan
Write-Host "Ollama: $env:OLLAMA_HOST  chat=$env:OLLAMA_CHAT_MODEL" -ForegroundColor Gray
& $venvPy -m uvicorn web.app:app --host 127.0.0.1 --port 8082
