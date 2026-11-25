# keylime

**keylime** is an experimental runtime, written from scratch in C++20, that runs
Xbox Series (Scarlett) game code natively on Windows PCs. Its first target is
GTA VI's Xbox build.

It is not an emulator of the CPU: the console's x86-64 code runs directly on
the host CPU. keylime provides everything around it: the loader, the console
system libraries, and a GPU layer that drives **Vulkan**, with a translator
from the console's RDNA 2 machine code to **SPIR-V**.

> keylime contains **no game files, keys or console SDK code**. You need your
> own copy of the game. Nothing here downloads or distributes game content.

## Status

| Area | State |
|---|---|
| Loader (PE mapping, static TLS redirect, import binding) | working |
| Kernel / files (`G:` package, `T:` temp) / memory (`xmem`, physical pages) | working |
| DirectStorage (worker pool, zlib) | working (BCPack textures: not yet) |
| GameInput (XInput pads + keyboard as a pad) | working |
| Audio: AcpHal XMA decoder, Opus (`xapu`), DSP (`chad`) | working (reverb is dry) |
| XG texture layouts (AMD addrlib) | working |
| GPU objects: device, DXGI, queues, fences, lists, heaps, resources, views, pipelines | working; the game's main loop runs at 30 fps |
| Shader containers + GFX10 decoder | working (320/320 sampled shaders decode) |
| GFX10 → SPIR-V translator | in progress |
| Drawing the game's frames | not yet (the window shows keylime's live status) |

## How keylime differs from other approaches

| | keylime |
|---|---|
| Graphics API | Vulkan 1.3/1.4 |
| Shader output | SPIR-V emitted directly (no DXC / DXIL round trip, fast driver compiles) |
| Queues | asynchronous: timeline semaphores, no wait after each submission |
| Memory | a GPU-side page table maps console addresses to wherever a page lives (host memory or VRAM / Resizable BAR), which is also how pages migrate |
| Descriptors | real RDNA 2 descriptor words in descriptor memory; images and samplers are found through a GPU-side hash table into bindless arrays, with no per-frame mirroring |
| wave64 on 32-lane GPUs | compute: one invocation runs two lanes exactly (planned) |
| Frame pacing | high-resolution waitable timers |

See [`docs/translator.md`](docs/translator.md) and
[`docs/rdna2-descriptors.md`](docs/rdna2-descriptors.md).

## Building

Requirements: Windows 10/11 x64, Visual Studio 2022 (or Build Tools) with the C++ workload
(CMake and Ninja ship with it), and a GPU driver with Vulkan 1.3.

    powershell -File tools\build.ps1

## Running

    tools\play.ps1                      opens a live log console and starts the game
    build\keylime.exe run <dir>\game_scarlett_master.exe --execute
    build\keylime.exe imports <exe>     map and bind only; lists unimplemented functions
    build\keylime.exe shader <dir|file> [--dis]   parse/disassemble console shader containers

Logs go to `logs\keylime.log`; set `KEYLIME_TRACE=1` for detailed logging.
Keyboard stands in for a controller: Enter = A, Backspace = B, Esc = Menu, Tab = View,
arrows = D-pad, WASD / IJKL = sticks, Q / E = shoulders, Z / C = triggers.

## Layout

    src/loader.*, src/pe.*     mapping and binding the game's images
    src/hle/                   console system libraries (kernel, files, xmem, dstorage, gameinput, xg)
    src/audio/                 AcpHal (XMA), xapu (Opus), chad (DSP)
    src/gpu/                   D3D12.X objects on Vulkan
    src/shader/                containers, GFX10 decoder, SPIR-V builder and translator
    third_party/               miniz, libopus, AMD addrlib, Vulkan and SPIR-V headers

## Credits and licence

keylime is licensed under the **GNU GPL v3.0 or later** (see `LICENSE`).

- Many console interface layouts (object slot maps, structure shapes, file
  formats) were reverse engineered by the [VIctory](https://github.com/Marvisak/VIctory)
  project (GPL-3.0); keylime reuses that knowledge and ports some of its code
  (noted in the files concerned).
- The XMA decoder follows FFmpeg's `wmaprodec.c` (LGPL-2.1-or-later) via VIctory.
- Third-party code keeps its own licence: miniz (MIT), libopus (BSD-3-Clause),
  AMD addrlib from Mesa (MIT), Khronos Vulkan / SPIR-V headers (Apache-2.0 / MIT),
  GFX10 opcode tables from Mesa's ACO (MIT) and register layouts from Mesa (MIT).

This project is not affiliated with Microsoft, Rockstar Games or Take-Two Interactive.
