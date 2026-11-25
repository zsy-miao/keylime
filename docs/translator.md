# keylime shader translator (GFX10 → SPIR-V)

Why not the embedded DXIL: only 52 of 320 shaders sampled from GTA VI carry
bitcode; the rest exist only as GFX10 machine code. So keylime translates the
machine code, like VIctory, but differently:

| | VIctory | keylime |
|---|---|---|
| Output | DXIL (LLVM bitcode) → DXC + driver compile (30-325 s for big CS) | SPIR-V emitted directly (fast driver compile on NVIDIA) |
| Memory | descriptor heaps mirrored to host heaps every frame | raw 64-bit pointers (buffer device address), like the hardware |
| wave64 on 32-wide GPUs | approximated | compute: one 32-wide invocation runs two lanes (L, L+32) exactly |

## Pipeline
1. `shader/container` — DXBC → XDAT → XBCH (zlib) → FSSM records: code, wave size,
   register counts, root layout (user SGPRs), used resources, stage I/O.
2. `shader/gfx10` — decoder for every RDNA 2 encoding (Mesa opcode tables).
3. `shader/translate` — GFX10 → SPIR-V:
   - SGPRs / VGPRs / VCC / EXEC / SCC / M0 are Function-storage variables.
   - Control flow: blocks in program order, each guarded by `pc == block`, so
     forward branches cost a compare and no loop; backward branches form loop
     regions (crossing ones merged) emitted as structured loops that repeat
     while `pc` stays inside. Only the user SGPRs the root layout fills are
     loaded from the draw record.
   - Wave-wide masks (VCC, EXEC, SGPR results of VOPC) come from subgroup ballots;
     VALU ops are predicated on the lane's EXEC bit.
   - Graphics stages run one lane per invocation (the rasterizer fixes them);
     compute runs wave64 as two lanes per invocation.
   - Scalar/global memory: PhysicalStorageBuffer pointers built from SGPR pairs /
     V# descriptor base addresses, with V# NUM_RECORDS bounds checks.
   - Images / samplers: the FSSM "used resources" table says which root
     parameter / table offset each T#/S# comes from, so descriptors are resolved
     at submission into Vulkan image/sampler slots of a bindless table; the
     shader indexes that table instead of reading T# words. Dynamic heap
     indexing (SM 6.6) goes through a global table keyed by heap slot.
   - VS inputs (fetch shader), exports (pos/param/mrt) and PS interpolation
     (v_interp_p1/p2) map to SPIR-V Input/Output variables by the FSSM I/O tables.
4. `gpu/translate` (later) — command streams → Vulkan command buffers, pipelines
   built from translated modules, dynamic rendering, push constants for user SGPRs.

UI first: the UI pipelines are VS + PS with few resources.
