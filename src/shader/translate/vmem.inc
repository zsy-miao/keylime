// keylime: buffer, global and LDS memory instructions (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    void vmem(const Inst& i) {
        std::string_view n = i.name;
        int count = 1;
        if (n.find("x2") != std::string_view::npos || n.find("_xy") != std::string_view::npos) count = 2;
        if (n.find("x3") != std::string_view::npos || n.find("_xyz") != std::string_view::npos) count = 3;
        if (n.find("x4") != std::string_view::npos || n.find("_xyzw") != std::string_view::npos) count = 4;
        if (i.enc == Enc::MUBUF || i.enc == Enc::MTBUF) {
            bool load_op = n.find("load") != std::string_view::npos, store_op = n.find("store") != std::string_view::npos;
            bool dword = n.find("dword") != std::string_view::npos || n.find("format") != std::string_view::npos;
            bool atomic = n.rfind("buffer_atomic_", 0) == 0;
            bool small = n.find("byte") != std::string_view::npos || n.find("short") != std::string_view::npos;
            if (!(load_op || store_op || atomic) || !(dword || small || atomic)) return unsupported(i);
            Id base = vsharp_base(i.srsrc);
            Id stride = band(shr(get_s(uint16_t(i.srsrc + 1)), u(16)), u(0x3fff));
            Id records = get_s(uint16_t(i.srsrc + 2));
            Id off = u(uint32_t(i.mem_offset));
            if (i.soffset != SNULL) off = iadd(off, raw(Operand{i.soffset, 0, false, false, true}));
            uint8_t va = i.vaddr;
            Id index = u(0);
            if (i.idxen) index = get_v(va++);
            if (i.offen) off = iadd(off, get_v(va));
            Id byte = iadd(b.op(spv::OpIMul, t_u32, {index, stride}), off);
            // Raw buffers (stride 0): bound by bytes; structured: by index.
            Id in_bounds = sel(t_bool, b.op(spv::OpIEqual, t_bool, {stride, u(0)}), b.op(spv::OpULessThan, t_bool, {off, records}),
                               b.op(spv::OpULessThan, t_bool, {index, records}));
            Id addr = add64(base, byte);
            if (atomic) return buffer_atomic(i, n, addr, in_bounds, i.vdata);
            if (small) {
                bool is_byte = n.find("byte") != std::string_view::npos, sext = n.find("sbyte") != std::string_view::npos || n.find("sshort") != std::string_view::npos;
                uint32_t width = is_byte ? 8 : 16;
                Id aligned = b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(~3ull)});
                Id shift = shl(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(3)})}), u(3));
                Id safe = b.op(spv::OpSelect, t_u64, {in_bounds, aligned, b.c_u64(0)});
                Id word = mem_load(safe);
                if (load_op) {
                    Id v_ = sext ? tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, word), shift, u(width)}))
                                 : b.op(spv::OpBitFieldUExtract, t_u32, {word, shift, u(width)});
                    set_v(i.vdata, sel(t_u32, in_bounds, v_, u(0)));
                } else {
                    Id merged = b.op(spv::OpBitFieldInsert, t_u32, {word, get_v(i.vdata), shift, u(width)});
                    Id wsafe = b.op(spv::OpSelect, t_u64, {b.op(spv::OpLogicalAnd, t_bool, {in_bounds, active()}), aligned, b.c_u64(0)});
                    mem_store(wsafe, merged);
                }
                return;
            }
            if (load_op) {
                // Out of bounds reads zero: guard by an address of our zero page (page table entry 0).
                auto words = mem_load_n(b.op(spv::OpSelect, t_u64, {in_bounds, addr, b.c_u64(0)}), count);
                for (int k = 0; k < count; ++k) set_v(uint16_t(i.vdata + k), sel(t_u32, in_bounds, words[k], u(0)));
                return;
            }
            std::vector<Id> vals;
            for (int k = 0; k < count; ++k) vals.push_back(get_v(uint16_t(i.vdata + k)));
            mem_store_n(addr, b.op(spv::OpLogicalAnd, t_bool, {in_bounds, active()}), vals);
            return;
        }
        if (i.enc == Enc::GLOBAL || i.enc == Enc::FLAT) {
            bool load_op = n.find("load") != std::string_view::npos, store_op = n.find("store") != std::string_view::npos;
            bool atomic = n.find("_atomic_") != std::string_view::npos;
            bool small = n.find("byte") != std::string_view::npos || n.find("short") != std::string_view::npos;
            if (!atomic && (!(load_op || store_op) || (n.find("dword") == std::string_view::npos && !small))) return unsupported(i);
            Id addr;
            if (i.enc == Enc::GLOBAL && i.saddr != 0x7d && i.saddr != 0x7f) {
                Operand sa;
                sa.code = i.saddr;
                addr = add64(raw64(sa), get_v(i.vaddr));
            } else {
                Id lo = b.op(spv::OpUConvert, t_u64, {get_v(i.vaddr)});
                Id hi = b.op(spv::OpUConvert, t_u64, {get_v(uint16_t(i.vaddr + 1))});
                addr = b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
            }
            addr = add64s(addr, u(uint32_t(i.mem_offset)));
            if (atomic) return buffer_atomic(i, n, addr, b.c_bool(true), i.vdata);
            if (small) {  // 8 / 16 bits: through the aligned dword
                bool is_byte = n.find("byte") != std::string_view::npos;
                bool sext = n.find("sbyte") != std::string_view::npos || n.find("sshort") != std::string_view::npos;
                uint32_t width = is_byte ? 8 : 16;
                Id aligned = b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(~3ull)});
                Id shift = shl(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(3)})}), u(3));
                Id word = mem_load(aligned);
                if (load_op) {
                    Id v_ = sext ? tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, word), shift, u(width)}))
                                 : b.op(spv::OpBitFieldUExtract, t_u32, {word, shift, u(width)});
                    return set_v(i.dst.vgpr_index(), v_);
                }
                Id merged = b.op(spv::OpBitFieldInsert, t_u32, {word, get_v(i.vdata), shift, u(width)});
                mem_store_n(aligned, active(), {merged});
                return;
            }
            uint16_t data = load_op ? i.dst.vgpr_index() : i.vdata;
            if (load_op) {
                auto words = mem_load_n(addr, count);
                for (int k = 0; k < count; ++k) set_v(uint16_t(data + k), words[k]);
                return;
            }
            std::vector<Id> vals;
            for (int k = 0; k < count; ++k) vals.push_back(get_v(uint16_t(data + k)));
            mem_store_n(addr, active(), vals);
            return;
        }
        if (i.enc == Enc::DS && lds) {
            const uint32_t last = (p.lds_bytes + 3) / 4 - 1;
            auto at = [&](Id byte) {
                return b.op(spv::OpAccessChain, lds_ptr, {lds, ext(GLSLstd450UMin, t_u32, {shr(byte, u(2)), u(last)})});
            };
            Id addr = iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff)));
            if (n == "ds_read_b32" || n == "ds_read_b64" || n == "ds_read_b128" || n == "ds_read_b96") {
                int c = n == "ds_read_b32" ? 1 : n == "ds_read_b64" ? 2 : n == "ds_read_b96" ? 3 : 4;
                for (int k = 0; k < c; ++k) set_v(uint16_t(i.dst.vgpr_index() + k), load(at(iadd(addr, u(4 * k))), t_u32));
                return;
            }
            if (n == "ds_write_b32" || n == "ds_write_b64" || n == "ds_write_b128" || n == "ds_write_b96") {
                int c = n == "ds_write_b32" ? 1 : n == "ds_write_b64" ? 2 : n == "ds_write_b96" ? 3 : 4;
                for (int k = 0; k < c; ++k) {
                    Id p = at(iadd(addr, u(4 * k)));
                    store(p, sel(t_u32, active(), get_v(uint16_t(i.data0 + k)), load(p, t_u32)));
                }
                return;
            }
            if (n == "ds_read2_b32" || n == "ds_write2_b32" || n == "ds_read2st64_b32" || n == "ds_write2st64_b32") {
                Id base = get_v(i.vaddr);
                uint32_t scale = n.find("st64") != std::string_view::npos ? 256 : 4;
                Id a0 = iadd(base, u(uint32_t(i.offset0) * scale)), a1 = iadd(base, u(uint32_t(i.offset1) * scale));
                if (n.rfind("ds_read2", 0) == 0) {
                    set_v(i.dst.vgpr_index(), load(at(a0), t_u32));
                    set_v(uint16_t(i.dst.vgpr_index() + 1), load(at(a1), t_u32));
                } else {
                    Id p0 = at(a0), p1 = at(a1);
                    store(p0, sel(t_u32, active(), get_v(i.data0), load(p0, t_u32)));
                    store(p1, sel(t_u32, active(), get_v(i.data1), load(p1, t_u32)));
                }
                return;
            }
        }
        if (i.enc == Enc::DS && lds && (n == "ds_write_b16" || n == "ds_write_b8")) {
            Id addr = iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff)));
            Id ptr = b.op(spv::OpAccessChain, lds_ptr, {lds, ext(GLSLstd450UMin, t_u32, {shr(addr, u(2)), u((p.lds_bytes + 3) / 4 - 1)})});
            Id shift = shl(band(addr, u(3)), u(3));
            Id merged = b.op(spv::OpBitFieldInsert, t_u32, {load(ptr, t_u32), get_v(i.data0), shift, u(n == "ds_write_b16" ? 16 : 8)});
            store(ptr, sel(t_u32, active(), merged, load(ptr, t_u32)));
            return;
        }
        // LDS atomics: ds_<op>[_rtn]_<u32|i32|b32>; inactive lanes leave memory alone.
        if (i.enc == Enc::DS && lds && n.size() > 7 && (n.substr(n.size() - 4) == "_u32" || n.substr(n.size() - 4) == "_i32" || n.substr(n.size() - 4) == "_b32")) {
            std::string_view body = n.substr(3, n.size() - 7);
            bool rtn = body.size() > 4 && body.substr(body.size() - 4) == "_rtn";
            if (rtn) body = body.substr(0, body.size() - 4);
            bool sg = n.substr(n.size() - 4) == "_i32";
            spv::Op op = body == "add" ? spv::OpAtomicIAdd : body == "sub" ? spv::OpAtomicISub : body == "or" ? spv::OpAtomicOr
                       : body == "and" ? spv::OpAtomicAnd : body == "xor" ? spv::OpAtomicXor
                       : body == "max" ? (sg ? spv::OpAtomicSMax : spv::OpAtomicUMax) : body == "min" ? (sg ? spv::OpAtomicSMin : spv::OpAtomicUMin)
                       : body == "wrxchg" ? spv::OpAtomicExchange : body == "inc" ? spv::OpAtomicIIncrement : spv::OpNop;
            if (op != spv::OpNop && op != spv::OpAtomicIIncrement) {
                Id addr = iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff)));
                Id ptr = b.op(spv::OpAccessChain, lds_ptr, {lds, ext(GLSLstd450UMin, t_u32, {shr(addr, u(2)), u((p.lds_bytes + 3) / 4 - 1)})});
                Id act = active(), val = get_v(i.data0), old;
                if (op == spv::OpAtomicExchange) {
                    Id res = b.local(p_fn_u32, u(0));
                    store(res, load(ptr, t_u32));
                    if_then(act, [&] { store(res, b.op(spv::OpAtomicExchange, t_u32, {ptr, u(spv::ScopeWorkgroup), u(0), val})); });
                    old = load(res, t_u32);
                } else {
                    Id neutral = body == "and" || (body == "min" && !sg) ? u(~0u) : body == "min" ? u(0x7fffffff) : body == "max" && sg ? u(0x80000000) : u(0);
                    old = b.op(op, t_u32, {ptr, u(spv::ScopeWorkgroup), u(0), sel(t_u32, act, val, neutral)});
                }
                if (rtn) set_v(i.dst.vgpr_index(), old);
                return;
            }
        }
        if (i.enc == Enc::DS && n == "ds_bpermute_b32") {
            auto shuf = [&](Id x, Id from) { return b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), x, from}); };
            if (!wide) {
                Id from = band(shr(iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff))), u(2)), u(31));
                return set_v(i.dst.vgpr_index(), shuf(get_v(i.data0), from));
            }
            // Wide (emitted once): every lane's result from both halves' sources, then the writes.
            Id src[2] = {as_half(0, [&] { return get_v(i.data0); }), as_half(1, [&] { return get_v(i.data0); })};
            Id res[2];
            for (int hh = 0; hh < 2; ++hh) {
                Id idx = as_half(hh, [&] { return band(shr(iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff))), u(2)), u(63)); });
                Id from = band(idx, u(31));
                res[hh] = sel(t_u32, ne0(band(idx, u(32))), shuf(src[1], from), shuf(src[0], from));
            }
            for (int hh = 0; hh < 2; ++hh) as_half(hh, [&] { set_v(i.dst.vgpr_index(), res[hh]); return 0; });
            return;
        }
        if (i.enc == Enc::DS && n == "ds_swizzle_b32") {
            // offset bit 15: quad-perm mode (offset[7:0] = 4 x 2-bit lanes); else bitmask mode (and/or/xor on 5 bits).
            uint32_t off = uint32_t(i.mem_offset) & 0xffff;
            Id from;
            if (off & 0x8000) {
                Id q = band(lane, u(3));
                Id selbits = shr(u(off & 0xff), shl(q, u(1)));
                from = bor(band(lane, u(~3u)), band(selbits, u(3)));
            } else {
                uint32_t and_m = off & 0x1f, or_m = (off >> 5) & 0x1f, xor_m = (off >> 10) & 0x1f;
                from = bxor(bor(band(lane, u(and_m | ~31u)), u(or_m)), u(xor_m));
            }
            return set_v(i.dst.vgpr_index(), b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), get_v(i.data0), band(from, u(31))}));
        }
        if (i.enc == Enc::MIMG) return image(i);
        return unsupported(i);
    }

    // Atomics on title memory (buffer_atomic_* / global_atomic_*). Inactive lanes and
    // out-of-bounds ones operate on the zero page with a neutral value.
