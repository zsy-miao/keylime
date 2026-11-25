// keylime: atomics on title memory (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    void buffer_atomic(const Inst& i, std::string_view n, Id addr, Id in_bounds, uint16_t data) {
        Id go = b.op(spv::OpLogicalAnd, t_bool, {in_bounds, active()});
        Id a = b.op(spv::OpSelect, t_u64, {go, addr, b.c_u64(0)});
        Id dev;
        if (flat()) dev = b.op(spv::OpSelect, t_u64, {go, flat_addr(a), sink_base});
        else dev = translate_addr(b.op(spv::OpSelect, t_u64, {guarded(page_entry(a)), b.c_u64(0), a}));
        Id ptr = b.op(spv::OpConvertUToPtr, p_psb_u32, {dev});
        Id v_ = get_v(data);
        Id old;
        auto is = [&](const char* k) { return n.find(k) != std::string_view::npos; };
        if (is("cmpswap")) {
            // data = new value, data + 1 = the value memory must hold; returns the old value.
            Id cmp = get_v(uint16_t(data + 1));
            Id res = b.local(p_fn_u32, u(0));
            if_then(go, [&] { store(res, b.op(spv::OpAtomicCompareExchange, t_u32, {ptr, u(spv::ScopeDevice), u(0), u(0), v_, cmp})); });
            if (i.glc) set_v(data, load(res, t_u32));
            return;
        }
        if (is("fmin") || is("fmax")) {
            // Float min/max: compare-and-swap loop.
            bool mn = is("fmin");
            Id hdr = b.id(), body_ = b.id(), cont_ = b.id(), merge_ = b.id();
            Id prev_var = b.local(p_fn_u32, u(0)), tries_var = b.local(p_fn_u32, u(0));
            store(prev_var, b.op(spv::OpAtomicLoad, t_u32, {ptr, u(spv::ScopeDevice), u(0)}));
            store(tries_var, u(0));
            b.op_void(spv::OpBranch, {hdr});
            b.label(hdr);
            b.op_void(spv::OpLoopMerge, {merge_, cont_, 0});
            b.op_void(spv::OpBranch, {body_});
            b.label(body_);
            Id prev = load(prev_var, t_u32);
            Id want = tou(ext(mn ? GLSLstd450NMin : GLSLstd450NMax, t_f32, {tof(prev), tof(v_)}));
            want = sel(t_u32, go, want, prev);
            Id got = b.op(spv::OpAtomicCompareExchange, t_u32, {ptr, u(spv::ScopeDevice), u(0), u(0), want, prev});
            store(prev_var, got);
            Id tries = b.op(spv::OpIAdd, t_u32, {load(tries_var, t_u32), u(1)});
            store(tries_var, tries);
            b.op_void(spv::OpBranch, {cont_});
            b.label(cont_);
            Id done = b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpIEqual, t_bool, {got, prev}), b.op(spv::OpUGreaterThanEqual, t_bool, {tries, u(256)})});
            b.op_void(spv::OpBranchConditional, {done, merge_, hdr});
            b.label(merge_);
            old = load(prev_var, t_u32);
        } else {
            spv::Op op = is("_add") ? spv::OpAtomicIAdd : is("_sub") ? spv::OpAtomicISub : is("_or") ? spv::OpAtomicOr
                       : is("_and") ? spv::OpAtomicAnd : is("_xor") ? spv::OpAtomicXor : is("_smax") ? spv::OpAtomicSMax
                       : is("_smin") ? spv::OpAtomicSMin : is("_umax") ? spv::OpAtomicUMax : is("_umin") ? spv::OpAtomicUMin
                       : is("_swap") ? spv::OpAtomicExchange : spv::OpNop;
            if (op == spv::OpNop) return unsupported(i);
            Id neutral = is("_and") || is("_umin") ? u(~0u) : is("_smin") ? u(0x7fffffff) : is("_smax") ? u(0x80000000) : u(0);
            if (op == spv::OpAtomicExchange) {
                old = b.op(spv::OpAtomicLoad, t_u32, {ptr, u(spv::ScopeDevice), u(0)});
                Id yes = b.id(), after = b.id();
                b.op_void(spv::OpSelectionMerge, {after, 0});
                b.op_void(spv::OpBranchConditional, {go, yes, after});
                b.label(yes);
                b.op(spv::OpAtomicExchange, t_u32, {ptr, u(spv::ScopeDevice), u(0), v_});
                b.op_void(spv::OpBranch, {after});
                b.label(after);
            } else {
                old = b.op(op, t_u32, {ptr, u(spv::ScopeDevice), u(0), sel(t_u32, go, v_, neutral)});
            }
        }
        if (i.glc) set_v(data, old);  // glc: return the pre-op value
    }

    // Images: T# (8 SGPRs) / S# (4 SGPRs) -> bindless slots through the texture table.
    //   table: u32 mask, then entries of {u32 key_lo, u32 key_hi, u32 slot, u32 pad}
    // A loop that stops at the first hit (usually the first probe): the unrolled form cost the
    // driver seconds per shader.
    struct SlotMemo { Id lo, hi, slot; uint32_t block; };
    std::vector<SlotMemo> slot_memo;
    Id slot_of(Id key_lo, Id key_hi) {
        for (auto& m : slot_memo)
            if (m.lo == key_lo && m.hi == key_hi && m.block == b.labels) return m.slot;
        Id tt = load(b.op(spv::OpAccessChain, p_pc_u64, {push, u(2)}), t_u64);
        Id mask = b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {tt}), uint32_t(spv::MemoryAccessAlignedMask), 4});
        // texture_hash, in SPIR-V.
        Id hx = bxor(b.op(spv::OpIMul, t_u32, {key_lo, u(0x9e3779b1u)}), b.op(spv::OpIMul, t_u32, {key_hi, u(0x85ebca6bu)}));
        hx = b.op(spv::OpIMul, t_u32, {bxor(hx, shr(hx, u(16))), u(0x7feb352du)});
        Id h = band(bxor(hx, shr(hx, u(15))), mask);
        Id probe = b.local(p_fn_u32, u(0)), result = b.local(p_fn_u32, u(0));
        b.op_void(spv::OpStore, {probe, u(0)});
        b.op_void(spv::OpStore, {result, u(0)});
        Id header = b.id(), body = b.id(), cont = b.id(), merge = b.id();
        b.op_void(spv::OpBranch, {header});
        b.label(header);
        b.op_void(spv::OpLoopMerge, {merge, cont, 0});
        b.op_void(spv::OpBranch, {body});
        b.label(body);
        Id pv = b.op(spv::OpLoad, t_u32, {probe});
        Id idx = band(iadd(h, pv), mask);
        Id e = b.op(spv::OpIAdd, t_u64, {tt, b.op(spv::OpUConvert, t_u64, {iadd(u(16), shl(idx, u(4)))})});
        auto word = [&](uint32_t k) {
            return b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {b.op(spv::OpIAdd, t_u64, {e, b.c_u64(4 * k)})}),
                                             uint32_t(spv::MemoryAccessAlignedMask), 4});
        };
        Id hit = b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {word(0), key_lo}), b.op(spv::OpIEqual, t_bool, {word(1), key_hi})});
        b.op_void(spv::OpStore, {result, sel(t_u32, hit, word(2), u(0))});
        Id last = b.op(spv::OpUGreaterThanEqual, t_bool, {pv, u(TEXTURE_TABLE_PROBES - 1)});
        b.op_void(spv::OpBranchConditional, {b.op(spv::OpLogicalOr, t_bool, {hit, last}), merge, cont});
        b.label(cont);
        b.op_void(spv::OpStore, {probe, iadd(pv, u(1))});
        b.op_void(spv::OpBranch, {header});
        b.label(merge);
        Id slot = b.op(spv::OpLoad, t_u32, {result});
        slot_memo.push_back({key_lo, key_hi, slot, b.labels});
        return slot;
    }
