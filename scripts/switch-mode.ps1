param(
  [ValidateSet("local","dgx")]
  [string]$Mode
)

$repo = "C:\Users\Usama\Localchat"
Set-Location $repo

$envFile = if ($Mode -eq "dgx") { ".env.dgx" } else { ".env.local" }
if (-not (Test-Path $envFile)) {
  throw "Missing $envFile in $repo"
}

Copy-Item $envFile .env -Force
Write-Host "Switched to $Mode mode using $envFile" -ForegroundColor Cyan

docker compose up -d --force-recreate | Out-Host

$hostLine = docker compose config | Select-String "OLLAMA_HOST" | Select-Object -First 1
if ($hostLine) {
  Write-Host $hostLine.ToString().Trim() -ForegroundColor Yellow
}

Write-Host "Health:" -ForegroundColor Cyan
curl.exe -sS http://127.0.0.1:8082/api/health | Out-Host