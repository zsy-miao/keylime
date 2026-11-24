#!/bin/bash
# Build, run the title for N seconds (default 25), show what happened.
#   tools/bringup.sh [secs]
cd /d/keylime || exit 1
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build.ps1 > logs/build.txt 2>&1 || { grep -E "error|FAILED" logs/build.txt | head -30; exit 2; }
taskkill //F //IM keylime.exe >/dev/null 2>&1
timeout ${1:-25} ./build/keylime.exe run 'D:\GTA6.Xbox.v1.0.2.1-CYBERLEEK\game_scarlett_master.exe' --execute > logs/run.txt 2>&1
echo "exit=$?"
taskkill //F //IM keylime.exe >/dev/null 2>&1
grep -vE "loader +(  missing|bound|mapped|attaching)|static TLS" logs/keylime.log | cut -c1-230 | tail -${2:-40}
