<# keylime: GTA VI save data (%LOCALAPPDATA%\keylime\data\7952987A).
   tools\saves.ps1 -Clear     back up the saves, then delete them (a fresh start)
   tools\saves.ps1 -Backup    copy the saves to D:\keylime\save-backups\<date>
   tools\saves.ps1 -Restore   put the newest backup back
   tools\saves.ps1 -Open      open the save folder #>
param([switch]$Clear, [switch]$Backup, [switch]$Restore, [switch]$Open)
$data = Join-Path $env:LOCALAPPDATA 'keylime\data\7952987A'
$backups = Join-Path (Split-Path $PSScriptRoot -Parent) 'save-backups'
$parts = 'saves', 'storage'
$Host.UI.RawUI.WindowTitle = 'keylime saves'

function Save-Copy {
    $have = $parts | Where-Object { Test-Path (Join-Path $data $_) }
    if (-not $have) { Write-Host 'There is no save data to back up.' -ForegroundColor DarkGray; return $null }
    $to = Join-Path $backups (Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')
    New-Item -ItemType Directory -Force $to | Out-Null
    foreach ($p in $have) { Copy-Item -Recurse -Force (Join-Path $data $p) $to }
    Write-Host "Backed up to $to" -ForegroundColor Green
    return $to
}
function Stop-Game {
    if (Get-Process keylime -ErrorAction SilentlyContinue) {
        Write-Host 'Closing the running game first...' -ForegroundColor Yellow
        Stop-Process -Name keylime -Force
        Start-Sleep -Seconds 3
    }
}

if ($Open) { New-Item -ItemType Directory -Force $data | Out-Null; Start-Process explorer.exe $data; exit 0 }
if ($Backup) { Save-Copy | Out-Null; exit 0 }
if ($Clear) {
    $answer = Read-Host 'Delete the GTA VI save data (a backup is made first)? Type yes'
    if ($answer -ne 'yes') { Write-Host 'Nothing deleted.'; exit 0 }
    Stop-Game
    Save-Copy | Out-Null
    foreach ($p in $parts + 'temp') { Remove-Item -Recurse -Force (Join-Path $data $p) -ErrorAction SilentlyContinue }
    Write-Host 'Save data cleared: the next start is a new game.' -ForegroundColor Green
    exit 0
}
if ($Restore) {
    $last = Get-ChildItem $backups -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
    if (-not $last) { Write-Host 'No backups yet.' -ForegroundColor Red; exit 1 }
    $answer = Read-Host "Replace the current saves with the backup from $($last.Name)? Type yes"
    if ($answer -ne 'yes') { Write-Host 'Nothing changed.'; exit 0 }
    Stop-Game
    Save-Copy | Out-Null
    foreach ($p in $parts) {
        Remove-Item -Recurse -Force (Join-Path $data $p) -ErrorAction SilentlyContinue
        if (Test-Path (Join-Path $last.FullName $p)) { Copy-Item -Recurse -Force (Join-Path $last.FullName $p) $data }
    }
    Write-Host "Restored the saves from $($last.Name)." -ForegroundColor Green
    exit 0
}
Get-Help $PSCommandPath
