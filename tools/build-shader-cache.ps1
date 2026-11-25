<# keylime: translate and compile every compute shader the game ships, once, into cache\.
   Run it once, and again only after an NVIDIA driver update or a TRANSLATOR_VERSION change
   (src/shader/translate.cpp); rebuilding keylime keeps the cache.
   tools\build-shader-cache.ps1 [-Workers 8] [-Game D:\GTA6...] #>
param([int]$Workers = 8, [string]$Game = 'D:\GTA6.Xbox.v1.0.2.1-CYBERLEEK')
$root = Split-Path $PSScriptRoot -Parent
$Host.UI.RawUI.WindowTitle = 'keylime shader cache'
Set-Location $root
if (Get-Process keylime -ErrorAction SilentlyContinue) {
    Write-Host 'Close the game first (both write the same cache).' -ForegroundColor Red
    exit 1
}
if (-not (Test-Path 'build\keylime.exe')) {
    & (Join-Path $PSScriptRoot 'build.ps1')
    if ($LASTEXITCODE -ne 0) { exit 1 }
}
# The game's shaders: its shader pack, and those inside its executable and DLLs.
if (-not (Test-Path 'shaders-pack\names.txt')) {
    Write-Host 'Extracting the shader pack (about a minute)...' -ForegroundColor Cyan
    python -I tools\extract_shaders.py (Join-Path $Game 'shaders.rpf') shaders-pack
}
if (-not (Test-Path 'shaders-embedded')) {
    Write-Host 'Extracting shaders inside the executable and DLLs...' -ForegroundColor Cyan
    python -I tools\extract_embedded_shaders.py $Game shaders-embedded
}
# A copy of the exe compiles, so keylime can be rebuilt meanwhile.
New-Item -ItemType Directory -Force cache | Out-Null
Copy-Item -Force build\keylime.exe cache\keylime-precompile.exe
$env:KEYLIME_PRECOMPILE_WORKERS = "$Workers"
foreach ($dir in 'shaders-pack', 'shaders-embedded') {
    Write-Host "Compiling $dir (shaders already in the cache are skipped)..." -ForegroundColor Cyan
    & .\cache\keylime-precompile.exe precompile (Join-Path $root $dir) | Select-String 'precompile:' | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -ne 0) { Write-Host "Stopped with code $LASTEXITCODE (run it again to go on)." -ForegroundColor Red; exit 1 }
}
Write-Host 'Shader cache built.' -ForegroundColor Green
exit 0
