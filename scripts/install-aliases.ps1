$profilePath = $PROFILE.CurrentUserAllHosts
$profileDir = Split-Path -Parent $profilePath
if (-not (Test-Path $profileDir)) {
  New-Item -ItemType Directory -Path $profileDir | Out-Null
}
if (-not (Test-Path $profilePath)) {
  New-Item -ItemType File -Path $profilePath | Out-Null
}

$snippet = @"
# Localchat mode aliases
function lc-local { & "C:\Users\Usama\Localchat\scripts\switch-mode.ps1" -Mode local }
function lc-dgx   { & "C:\Users\Usama\Localchat\scripts\switch-mode.ps1" -Mode dgx }
function lc-mode  { & "C:\Users\Usama\Localchat\scripts\check-mode.ps1" }
"@

$existing = Get-Content $profilePath -Raw
if ($existing -notmatch "function lc-local") {
  Add-Content -Path $profilePath -Value "`n$snippet"
  Write-Host "Aliases added to $profilePath" -ForegroundColor Green
} else {
  Write-Host "Aliases already present in $profilePath" -ForegroundColor Yellow
}

Write-Host "Open new PowerShell, then use: lc-local | lc-dgx | lc-mode" -ForegroundColor Cyan