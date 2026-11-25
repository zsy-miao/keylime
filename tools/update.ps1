<# keylime: get the latest version from GitHub and build it.
   tools\update.ps1 #>
$root = Split-Path $PSScriptRoot -Parent
$Host.UI.RawUI.WindowTitle = 'keylime update'
Set-Location $root
if (Get-Process keylime -ErrorAction SilentlyContinue) {
    Write-Host 'Closing the running keylime (the build replaces its exe)...' -ForegroundColor Yellow
    Stop-Process -Name keylime -Force
    Start-Sleep -Seconds 3
}
$before = (git rev-parse --short HEAD)
Write-Host "keylime ${before}: fetching from GitHub..." -ForegroundColor Cyan
git fetch origin
if ($LASTEXITCODE -ne 0) { Write-Host 'Could not reach GitHub.' -ForegroundColor Red; exit 1 }
if (git status --porcelain --untracked-files=no) {
    Write-Host 'This copy has local changes; they are put aside (git stash) before updating.' -ForegroundColor Yellow
    git stash push -m "local changes before update $(Get-Date -Format s)"
}
git pull --ff-only origin main
if ($LASTEXITCODE -ne 0) { Write-Host 'Update failed (the history diverged).' -ForegroundColor Red; exit 1 }
$after = (git rev-parse --short HEAD)
if ($before -eq $after) { Write-Host "Already the latest ($after)." -ForegroundColor Green }
else {
    Write-Host "Updated ${before} -> ${after}:" -ForegroundColor Green
    git log --oneline "$before..$after"
}
Write-Host 'Building...' -ForegroundColor Cyan
& (Join-Path $PSScriptRoot 'build.ps1')
if ($LASTEXITCODE -ne 0) { Write-Host 'Build failed.' -ForegroundColor Red; exit 1 }
Write-Host "keylime $after is built." -ForegroundColor Green
exit 0
