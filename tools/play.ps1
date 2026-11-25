<# keylime: run GTA VI with a live log console. Logs also go to D:\keylime\logs\keylime.log.
   tools\play.ps1                 normal run
   tools\play.ps1 -Trace          very detailed logging (KEYLIME_TRACE=1)
   tools\play.ps1 -Update         get the latest version from GitHub and build it first #>
param([switch]$Trace, [switch]$Update, [string]$Game = 'D:\GTA6.Xbox.v1.0.2.1-CYBERLEEK')
$root = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $root 'build\keylime.exe'
$Host.UI.RawUI.WindowTitle = 'keylime log'
if ($Update) {
    & (Join-Path $PSScriptRoot 'update.ps1')
    if ($LASTEXITCODE -ne 0) { Read-Host 'Update failed. Press Enter to play the version already built'; }
}
if (-not (Test-Path $exe)) {
    Write-Host 'Building keylime first...' -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot 'build.ps1')
    if ($LASTEXITCODE -ne 0) { Write-Host 'Build failed.' -ForegroundColor Red; Read-Host 'Press Enter'; exit 1 }
}
if ($Trace) { $env:KEYLIME_TRACE = '1' }
Set-Location $root
Write-Host ("keylime " + (git -C $root rev-parse --short HEAD) + " - GTA VI") -ForegroundColor Green
Write-Host "Keyboard: Enter=A  Backspace=B  Esc=Menu  Tab=View  arrows=D-pad  WASD=left stick  IJKL=right stick" -ForegroundColor DarkGray
Write-Host "Status lines (cyan) every 2 s show frames, GPU work, streaming, audio and input.`n" -ForegroundColor DarkGray
& $exe run (Join-Path $Game 'game_scarlett_master.exe') --execute
$rc = $LASTEXITCODE
Write-Host "`nkeylime exited with code $rc (log: $root\logs\keylime.log)" -ForegroundColor Yellow
Read-Host 'Press Enter to close'
