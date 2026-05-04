$repo = "C:\Users\Usama\Localchat"
Set-Location $repo

$hostLine = docker compose config | Select-String "OLLAMA_HOST" | Select-Object -First 1
$envLine = docker inspect localchat-app --format "{{range .Config.Env}}{{println .}}{{end}}" 2>$null | Select-String "OLLAMA_HOST" | Select-Object -First 1

if ($hostLine) { Write-Host "Compose: $($hostLine.ToString().Trim())" -ForegroundColor Yellow }
if ($envLine) { Write-Host "Container: $($envLine.ToString().Trim())" -ForegroundColor Yellow }

Write-Host "Health:" -ForegroundColor Cyan
curl.exe -sS http://127.0.0.1:8082/api/health | Out-Host