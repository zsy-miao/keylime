// keylime: reaching title memory: the flat view, the page walk, loads and stores (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    // -------------------------------------------------------- memory ----
    // title address -> device address through the 2-level page table (1 GiB, 64 KiB pages).
    // The page table's address and the sink page's (loaded once, in the entry block).
    Id pt_base = 0, sink_base = 0, zero_base = 0, flat_base = 0;
    bool flat() const { return opt && opt->flat; }
    // Flat view: title address a is at slots[a >> 30] + a (a sparse buffer the host binds page by
    // page, one slot per title GiB; unbound pages read zeros and drop stores: no walk or guard).
    // flat_base holds the slot table's address.
    Id flat_addr(Id addr) {
        addr = b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64((1ull << 47) - 1)});
        Id at = b.op(spv::OpIAdd, t_u64, {flat_base, b.op(spv::OpShiftLeftLogical, t_u64, {b.op(spv::OpShiftRightLogical, t_u64, {addr, b.c_u32(30)}), b.c_u32(3)})});
        Id e = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {at}), uint32_t(spv::MemoryAccessAlignedMask), 8});
        return b.op(spv::OpIAdd, t_u64, {e, addr});
    }
    Id page_table() { return pt_base ? pt_base : load(b.op(spv::OpAccessChain, p_pc_u64, {push, u(1)}), t_u64); }
    void memory_bases() {
        pt_base = page_table();
        Id at = b.op(spv::OpIAdd, t_u64, {pt_base, b.c_u64((1ull << 17) * 8 + 16)});
        sink_base = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {at}), uint32_t(spv::MemoryAccessAlignedMask), 8});
        Id at2 = b.op(spv::OpIAdd, t_u64, {pt_base, b.c_u64((1ull << 17) * 8 + 24)});
        zero_base = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {at2}), uint32_t(spv::MemoryAccessAlignedMask), 8});
        if (flat()) {
            Id at3 = b.op(spv::OpIAdd, t_u64, {pt_base, b.c_u64((1ull << 17) * 8 + 32)});
            flat_base = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {at3}), uint32_t(spv::MemoryAccessAlignedMask), 8});
        }
    }
    Id translate_addr(Id addr) { return flat() ? flat_addr(addr) : dev_of(page_entry(addr), addr); }
    Id dev_of(Id entry, Id addr) {
        Id page = b.op(spv::OpBitwiseAnd, t_u64, {entry, b.c_u64(~(3ull << 62))});  // bits 63 guard, 62 next-page-follows
        return b.op(spv::OpIAdd, t_u64, {page, b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(0xffff)})});
    }
    // Entries of imported pages that no GPU resource or heap covers carry bit 63: stores there
    // are skipped, counted and the first one's address kept for the host's log.
    Id guarded(Id entry) { return b.op(spv::OpINotEqual, t_bool, {b.op(spv::OpShiftRightLogical, t_u64, {entry, b.c_u32(63)}), b.c_u64(0)}); }
    void report_guard(Id bad, Id addr) {
        Id yes = b.id(), after = b.id(), first = b.id(), done = b.id();
        b.op_void(spv::OpSelectionMerge, {after, 0});
        b.op_void(spv::OpBranchConditional, {bad, yes, after});
        b.label(yes);
        Id rec = b.op(spv::OpIAdd, t_u64, {page_table(), b.c_u64((1ull << 17) * 8)});
        auto at = [&](uint32_t k) { return b.op(spv::OpConvertUToPtr, p_psb_u32, {b.op(spv::OpIAdd, t_u64, {rec, b.c_u64(4 * k)})}); };
        Id old = b.op(spv::OpAtomicIAdd, t_u32, {at(0), u(spv::ScopeDevice), u(0), u(1)});
        Id is_first = b.op(spv::OpIEqual, t_bool, {old, u(0)});
        b.op_void(spv::OpSelectionMerge, {done, 0});
        b.op_void(spv::OpBranchConditional, {is_first, first, done});
        b.label(first);
        auto put = [&](uint32_t k, Id v) { b.op_void(spv::OpStore, {at(k), v, uint32_t(spv::MemoryAccessAlignedMask), 4}); };
        put(1, u(code_id(p)));
        put(2, b.op(spv::OpUConvert, t_u32, {addr}));
        put(3, b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {addr, b.c_u32(32)})}));
        b.op_void(spv::OpBranch, {done});
        b.label(done);
        b.op_void(spv::OpBranch, {after});
        b.label(after);
    }
    Id page_entry(Id addr) {
        addr = b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64((1ull << 47) - 1)});
        Id pt = page_table();
        Id i1 = b.op(spv::OpShiftRightLogical, t_u64, {addr, b.c_u32(30)});
        Id a1 = b.op(spv::OpIAdd, t_u64, {pt, b.op(spv::OpShiftLeftLogical, t_u64, {i1, b.c_u32(3)})});
        Id l2 = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {a1}), uint32_t(spv::MemoryAccessAlignedMask), 8});
        Id i2 = b.op(spv::OpBitwiseAnd, t_u64, {b.op(spv::OpShiftRightLogical, t_u64, {addr, b.c_u32(16)}), b.c_u64(0x3fff)});
        Id a2 = b.op(spv::OpIAdd, t_u64, {l2, b.op(spv::OpShiftLeftLogical, t_u64, {i2, b.c_u32(3)})});
        return b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {a2}), uint32_t(spv::MemoryAccessAlignedMask), 8});
    }
    // `count` consecutive dwords from `addr64`: one instruction touches at most two 64 KiB pages,
    // so two page-table walks serve all of them (the walks are half of a shader's driver
    // compile time when done per dword).
    // `count` consecutive dwords from `addr64` with one page walk, at contiguous device addresses
    // the driver can fetch as one load (per-dword page selects cost a third of a big shader's
    // compile time). Spanning into the next page is fine where the host marked it as following
    // on the device (bit 62, true inside every import); an access spanning two separate imports
    // reads zeros.
    std::vector<Id> mem_load_n(Id addr64, int count) {
        std::vector<Id> out;
        auto word = [&](Id dev) {
            return b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {dev}), uint32_t(spv::MemoryAccessAlignedMask), 4});
        };
        if (flat()) {
            Id base = flat_addr(addr64);
            for (int k = 0; k < count; ++k) out.push_back(word(k ? b.op(spv::OpIAdd, t_u64, {base, b.c_u64(uint64_t(4 * k))}) : base));
            return out;
        }
        Id e0 = page_entry(addr64);
        Id dev0 = dev_of(e0, addr64);
        if (count == 1) return {word(dev0)};
        Id in_page = band(b.op(spv::OpUConvert, t_u32, {addr64}), u(0xffff));
        Id fits = b.op(spv::OpULessThanEqual, t_bool, {in_page, u(uint32_t(0x10000 - 4 * count))});
        Id next = b.op(spv::OpINotEqual, t_bool, {b.op(spv::OpBitwiseAnd, t_u64, {e0, b.c_u64(1ull << 62)}), b.c_u64(0)});
        Id base = b.op(spv::OpSelect, t_u64, {b.op(spv::OpLogicalOr, t_bool, {fits, next}), dev0, zero_base});
        for (int k = 0; k < count; ++k) out.push_back(word(b.op(spv::OpIAdd, t_u64, {base, b.c_u64(uint64_t(4 * k))})));
        return out;
    }
    Id mem_load(Id addr64) {
        Id dev = translate_addr(addr64);
        return b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {dev}), uint32_t(spv::MemoryAccessAlignedMask), 4});
    }
    // Branch-free: lanes that must not store (inactive, address 0, or a page no GPU resource or
    // heap covers) store to the sink page instead (a branch per store cost the driver much of
    // its compile time).
    void mem_store_n(Id addr64, Id go, const std::vector<Id>& vals) {
        int count = int(vals.size());
        go = b.op(spv::OpLogicalAnd, t_bool, {go, b.op(spv::OpINotEqual, t_bool, {addr64, b.c_u64(0)})});
        if (flat()) {  // lanes that must not store write the sink page (still branch-free)
            Id base = b.op(spv::OpSelect, t_u64, {go, flat_addr(addr64), sink_base});
            for (int k = 0; k < count; ++k) {
                Id dev = k ? b.op(spv::OpIAdd, t_u64, {base, b.c_u64(uint64_t(4 * k))}) : base;
                b.op_void(spv::OpStore, {b.op(spv::OpConvertUToPtr, p_psb_u32, {dev}), vals[size_t(k)], uint32_t(spv::MemoryAccessAlignedMask), 4});
            }
            return;
        }
        Id e0 = page_entry(addr64);
        Id ok = b.op(spv::OpLogicalAnd, t_bool, {go, b.op(spv::OpLogicalNot, t_bool, {guarded(e0)})});
        if (count > 1) {  // one walk, contiguous on the device (see mem_load_n); else the sink
            Id in_page = band(b.op(spv::OpUConvert, t_u32, {addr64}), u(0xffff));
            Id fits = b.op(spv::OpULessThanEqual, t_bool, {in_page, u(uint32_t(0x10000 - 4 * count))});
            Id next = b.op(spv::OpINotEqual, t_bool, {b.op(spv::OpBitwiseAnd, t_u64, {e0, b.c_u64(1ull << 62)}), b.c_u64(0)});
            ok = b.op(spv::OpLogicalAnd, t_bool, {ok, b.op(spv::OpLogicalOr, t_bool, {fits, next})});
        }
        Id base = b.op(spv::OpSelect, t_u64, {ok, dev_of(e0, addr64), sink_base});
        for (int k = 0; k < count; ++k) {
            Id dev = b.op(spv::OpIAdd, t_u64, {base, b.c_u64(uint64_t(4 * k))});
            b.op_void(spv::OpStore, {b.op(spv::OpConvertUToPtr, p_psb_u32, {dev}), vals[size_t(k)], uint32_t(spv::MemoryAccessAlignedMask), 4});
        }
    }
    void mem_store(Id addr64, Id val) { mem_store_n(addr64, b.c_bool(true), {val}); }
    Id add64(Id a64, Id off32) { return b.op(spv::OpIAdd, t_u64, {a64, b.op(spv::OpUConvert, t_u64, {off32})}); }
    Id add64s(Id a64, Id off32) { return b.op(spv::OpIAdd, t_u64, {a64, b.op(spv::OpSConvert, t_u64, {bc(t_i32, off32)})}); }
    // V# base (48 bits) from four SGPRs.
    Id vsharp_base(uint16_t s0) {
        Id lo = b.op(spv::OpUConvert, t_u64, {get_s(s0)});
        Id hi = b.op(spv::OpUConvert, t_u64, {band(get_s(uint16_t(s0 + 1)), u(0xffff))});
        return b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
    }

