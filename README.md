# keylime

Runs the Xbox Series build of GTA VI on Windows. C++20, from scratch, reusing
ideas/layouts from VIctory (GPL-3.0) and others where noted in each file.

    tools\build.ps1                      build (VS 2022 Build Tools, CMake + Ninja)
    build\keylime.exe imports <exe>      map + bind only, list what is missing
    build\keylime.exe run <exe> --execute
    tools/bringup.sh [secs]              build, run, show the log tail

Layout: src/loader (PE mapping, TLS redirect, import binding, stubs),
src/hle (kernel, files, xmem, gameinput, dstorage, ...), src/gpu (to come).
Logs: logs\keylime.log. Cache: %LOCALAPPDATA%\keylime.
