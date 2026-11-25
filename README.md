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

## Progress

The real retail GTA VI exe boots natively, reaches the main menu with all
audio working, and keylime renders its frames on Vulkan.

**Current blocker:** about 48 s in, when the game world starts loading after
the menu, the game sends thousands of new shaders at once and crashed or froze
shortly after. What changed, waiting for a test run:

- The NVIDIA driver could take 2 to 26 s to compile one translated compute
  shader. Done on the title's submit thread, that stalled the GPU long enough
  for the game's own GPU watchdog to abort ("GPU Loc Sig", a deliberate write
  to address 0 with code `0x4D8BD421`). Skipping compute dispatches instead
  made the game read a bogus address and hit an `int3`
  (`game_scarlett_master.exe+0x4da6315`).
- The submit thread never compiles. A first use queues the pipeline ahead of
  the rest and waits for it within a budget of 0.5 s per frame over all
  threads; past it, draws and dispatches whose pipeline isn't ready are
  dropped. Compute pipelines get a quick unoptimized build first.
- Pipelines a run used are listed in `cache\used-pipelines.txt`; the next run
  starts building those as soon as the game creates them, so the first run
  after a change still drops some work and later runs should drop less.
- Translated shaders are cheaper for the driver: structured loops and
  branches instead of a dispatcher loop, and only the user SGPRs in use are
  loaded. Driver compiles are kept across runs in `cache\pipelines.bin`.
- `ExecuteIndirect` runs for Draw, DrawIndexed and Dispatch signatures
  (counted too), and GPU waits on memory hold later work until the value
  matches (polled on the CPU, 250 ms timeout). Indirect arguments are
  checked on the GPU first: a command past the count or with sizes no frame
  would use (memory a dropped dispatch never wrote) becomes empty.
- The bindless texture table has 1M entries and a better hash (it ran out of
  room at about 42k images), and up to 262144 image slots.
- Mesh shader dispatches and multi-argument `ExecuteIndirect` signatures are
  still skipped.

- [x] Retail exe boots natively: archives stream through DirectStorage
      (thousands of requests, 260+ MB), main loop holds 30 fps
- [x] Gets through the intro and the main menu
- [x] All audio works (XMA, Opus, DSP)
- [x] All 320 sampled shaders translate GFX10 → SPIR-V directly, pass
      `spirv-val` and build as NVIDIA pipelines
- [x] The game's command lists run on the host GPU: about 650 draws and 60
      dispatches every 2 s with none skipped, 2,052 images, 600+ frames
      presented to the swap chain (one test run)
- [x] Title memory imported zero-copy behind a GPU page table; textures as
      Vulkan images (addrlib detiling, bindless hash table); rect/quad lists
      through a geometry shader
- [x] The game's frames render on Vulkan and present to the window
- [x] BCPack-compressed textures decode in the DirectStorage path
- [x] `xgameruntime` areas (user, store, save, async, networking, package,
      storage, system and more) emulated well enough to reach the menu
- [ ] Get through world loading after the menu (pipeline compile stalls, see above)
- [x] `ExecuteIndirect` (single-argument signatures) and GPU memory waits
- [ ] Mesh shader dispatches, multi-argument `ExecuteIndirect`

## Status

| Area | State |
|---|---|
| Loader (PE mapping, static TLS redirect, import binding) | working |
| Kernel / files (`G:` package, `T:` temp) / memory (`xmem`, physical pages) | working |
| DirectStorage (worker pool, zlib, BCPack) | working |
| GameInput (XInput pads + keyboard as a pad) | working |
| `xgameruntime` (XUser, XStore, XGameSave, XAsync, other areas) | working far enough for the menus; some methods are stubs |
| Audio: AcpHal XMA decoder, Opus (`xapu`), DSP (`chad`) | working: all audio plays in the menus (reverb is dry) |
| XG texture layouts (AMD addrlib) | working |
| GPU objects: device, DXGI, queues, fences, lists, heaps, resources, views, pipelines | working; the game's main loop runs at 30 fps |
| Shader containers + GFX10 decoder | working (320/320 sampled shaders decode) |
| GFX10 → SPIR-V translator | working (320/320 sampled shaders pass `spirv-val`) |
| Executing the game's command lists and rendering its frames on Vulkan | working through the main menu; `ExecuteIndirect` not run yet |

## How keylime differs from other approaches

Several projects are trying to get GTA VI's Xbox build running on PC. keylime
sits in the same family as VIctory and XCL (the console's own x86-64 code runs
natively, with the system libraries and GPU replaced around it), but it takes
a different graphics path: everything else here that says what it targets goes
through Direct3D 12 and DXIL, while keylime drives Vulkan and emits SPIR-V
itself.

| Project | Approach | Scope | Graphics path |
|---|---|---|---|
| **keylime** | native runtime, C++20, written from scratch | GTA VI first | Vulkan; RDNA 2 machine code → SPIR-V directly |
| [VIctory](https://github.com/Marvisak/VIctory) | compatibility layer, Rust | GTA VI only | RDNA 2 machine code → DXIL, compiled by DXC |
| [hyphennn's fork](https://github.com/fxceless/victory-fork) | VIctory plus its own rendering and speed fixes | GTA VI only | same as VIctory |
| [XCL](https://github.com/xcl-dev/xcl) | Wine-style compatibility layer | any Xbox Series game | RDNA 2 programs → HLSL → DXIL |
| [Xbox2Windows](https://github.com/R0Y00/Xbox2Windows) | static recompilation: code and shaders turned into Windows C++ ahead of time | any Xbox Series game | DXIL |
| [TuxBox](https://github.com/brcly/TuxBox) | native-first layer for unpacked games, mostly tested on Linux via Proton | any Xbox Series game | not stated |
| [xbox-emulator](https://github.com/omonkeyz/xbox-emulator) | Rust runtime running dumped builds as Windows processes (continues TomBox) | any Xbox Series game | not stated |
| [ScarlettX](https://github.com/Freminet/ScarlettX) | clean-room HLE emulator of the Scarlett OS and GDK runtime | any Xbox Series game | not stated |
| [OpenScarlett CE](https://github.com/OpenCommunityEdition/OpenScarlett) | AGPL C++ layer that swaps console DLLs for PC ones | any Xbox Series game | Direct3D 12 |

What that means in practice:

- **No DXC round trip.** Only 52 of 320 sampled GTA VI shaders carry DXIL
  bitcode, so every layer has to translate the GFX10 machine code. keylime
  writes SPIR-V straight from it instead of going through HLSL or DXIL, which
  keeps driver compiles fast (VIctory's DXIL path takes 30-325 s for big
  compute shaders).
- **The console's memory model, kept.** Shaders read raw 64-bit pointers and
  real RDNA 2 descriptor words, as on the hardware, instead of VIctory's
  descriptor heaps mirrored to host heaps every frame.
- **GTA VI first.** Like VIctory, keylime targets one game rather than every
  Xbox Series title. It builds on VIctory's reverse-engineered interface
  layouts (see [Credits](#credits-and-licence)) but not on its GPU path.

Where it stands: keylime is behind the leading projects. The game boots, its
frames render on Vulkan and it reaches the main menu with audio, but it crashes
after that and isn't playable yet (see
[Progress](#progress)). The other
projects' progress is their own reporting, mostly unverified.

### Design choices

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
A debug panel in the top-left corner shows fps, a frame-time graph, GPU work,
shader compiles, VRAM and streaming; F3 toggles it (`KEYLIME_OVERLAY=0` starts it hidden).
Crash diagnostics: every distinct fault is logged once, an unhandled crash writes
`logs\crash.dmp`, and if no frame is shown for 6 s every thread's stack goes to
`logs\stacks.txt` (also `KEYLIME_STACKS_AT=<seconds>`).
Keyboard stands in for a controller: Enter = A, Backspace = B, Esc = Menu, Tab = View,
arrows = D-pad, WASD / IJKL = sticks, Q / E = shoulders, Z / C = triggers.

## Layout

    src/loader.*, src/pe.*     mapping and binding the game's images
    src/hle/                   console system libraries (kernel, files, xmem, dstorage, gameinput, xg)
    src/hle/xgr/               xgameruntime (user, store, saves, async, other areas)
    src/formats/               BCPack decoder
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

keylime is written with the help of Claude, Anthropic's AI model.

This project is not affiliated with Microsoft, Rockstar Games or Take-Two Interactive.
