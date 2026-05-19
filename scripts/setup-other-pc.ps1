# First-time setup on a new Windows PC (Docker + Localchat).
# Run from repo root:  .\scripts\setup-other-pc.ps1
# Or:                 .\scripts\setup-other-pc.ps1 -Mode dgx

param(
  [ValidateSet("local", "dgx")]
  [string]$Mode = "local"
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo

function Test-Cmd($name) {
  $null -ne (Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "Localchat setup — repo: $repo" -ForegroundColor Cyan

if (-not (Test-Cmd docker)) {
  throw "Docker not found. Install Docker Desktop and restart PowerShell."
}
docker info 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
  throw "Docker is installed but not running. Start Docker Desktop, then run this script again."
}

New-Item -ItemType Directory -Force -Path (Join-Path $repo "docs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $repo "data") | Out-Null

$envExample = if ($Mode -eq "dgx") { "env.dgx.example" } else { "env.local.example" }
$envTarget = ".env"
if (-not (Test-Path $envExample)) {
  throw "Missing $envExample in repo root"
}
if (-not (Test-Path $envTarget)) {
  Copy-Item $envExample $envTarget -Force
  Write-Host "Created $envTarget from $envExample" -ForegroundColor Green
} else {
  Write-Host "Using existing $envTarget" -ForegroundColor Yellow
}
# Templates for switch-mode.ps1 (gitignored copies you can edit)
if (-not (Test-Path ".env.local") -and (Test-Path "env.local.example")) {
  Copy-Item "env.local.example" ".env.local" -Force
}
if (-not (Test-Path ".env.dgx") -and (Test-Path "env.dgx.example")) {
  Copy-Item "env.dgx.example" ".env.dgx" -Force
  if ($Mode -eq "dgx") {
    Write-Host "Edit .env.dgx — set OLLAMA_HOST to your DGX IP, then run switch-mode -Mode dgx" -ForegroundColor Yellow
  }
}

$pdfs = @(Get-ChildItem -Path (Join-Path $repo "docs") -Filter "*.pdf" -ErrorAction SilentlyContinue)
if ($pdfs.Count -eq 0) {
  Write-Host "No PDF in docs/ yet — copy your manual to docs\ before indexing." -ForegroundColor Yellow
} else {
  Write-Host "Found PDF in docs/: $($pdfs[0].Name)" -ForegroundColor Green
}

if ($Mode -eq "local") {
  if (Test-Cmd ollama) {
    Write-Host "Checking Ollama models (local mode)…" -ForegroundColor Cyan
    ollama list 2>&1 | Out-Host
    Write-Host "If models are missing, run:" -ForegroundColor Yellow
    Write-Host "  ollama pull nomic-embed-text" -ForegroundColor Gray
    Write-Host "  ollama pull qwen3.5:9b" -ForegroundColor Gray
  } else {
    Write-Host "Ollama CLI not found — install Ollama on this PC or use -Mode dgx" -ForegroundColor Yellow
  }
} else {
  Write-Host "DGX mode: ensure Ollama is running on the server in .env (OLLAMA_HOST)." -ForegroundColor Yellow
}

Write-Host "Building and starting Docker…" -ForegroundColor Cyan
docker compose up --build -d
if ($LASTEXITCODE -ne 0) { throw "docker compose failed" }

Start-Sleep -Seconds 4
Write-Host "Health check:" -ForegroundColor Cyan
try {
  $health = curl.exe -sS http://127.0.0.1:8082/api/health
  Write-Host $health
} catch {
  Write-Host "App not ready yet — wait 30s and open http://127.0.0.1:8082" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Next steps ===" -ForegroundColor Cyan
Write-Host "1. Open  http://127.0.0.1:8082" -ForegroundColor White
Write-Host "2. Admin login (PDF tools):  admin / admin" -ForegroundColor White
Write-Host "3. Put PDF in docs\ if not already there" -ForegroundColor White
Write-Host "4. Click  Use docs PDF  (or upload) and wait for indexing" -ForegroundColor White
Write-Host "5. Ask a question in the chat box" -ForegroundColor White
Write-Host ""
Write-Host "Switch Ollama target later:  .\scripts\switch-mode.ps1 -Mode local|dgx" -ForegroundColor Gray
