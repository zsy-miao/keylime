// keylime: registers, constants, lane masks and small helpers (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    // ---------------------------------------------------------- helpers ----
    Id u(uint32_t x) {
        Id c = b.c_u32(x);
        consts[c] = x;
        if (x == ~0u && !no_lane_facts) mine_ids.insert(c);
        return c;
    }
    Id shift_amount(Id n) {
        auto it = consts.find(n);
        return it != consts.end() ? u(it->second & 31) : band(n, u(31));
    }
    Id f(float x) { return b.c_f32(x); }
    // Register variables keep their last value per basic block: a load after a store or load of
    // the same register in the same block reuses the value (far less SPIR-V for the driver).
    struct Known { Id value; uint32_t block; };
    std::unordered_map<Id, Known> known;
    std::unordered_set<Id> registers;
    bool cache_loads = false;  // measured: reusing loaded values made driver compiles slightly slower
    Id load(Id var, Id type) {
        if (cache_loads && registers.count(var)) {
            auto it = known.find(var);
            if (it != known.end() && it->second.block == b.labels) return it->second.value;
            Id x = b.op(spv::OpLoad, type, {var});
            known[var] = {x, b.labels};
            return x;
        }
        return b.op(spv::OpLoad, type, {var});
    }
    void store(Id var, Id val) {
        b.op_void(spv::OpStore, {var, val});
        if (registers.count(var)) known[var] = {val, b.labels};
    }
    Id bc(Id type, Id x) { return b.op(spv::OpBitcast, type, {x}); }
    Id ext(uint32_t inst, Id type, std::initializer_list<Id> args) {
        std::vector<uint32_t> w = {glsl, inst};
        w.insert(w.end(), args);
        return b.op(spv::OpExtInst, type, w);
    }
    Id sel(Id type, Id c, Id a, Id z) { return b.op(spv::OpSelect, type, {c, a, z}); }
    Id iadd(Id a, Id z) { return b.op(spv::OpIAdd, t_u32, {a, z}); }
    Id isub(Id a, Id z) { return b.op(spv::OpISub, t_u32, {a, z}); }
    Id band(Id a, Id z) { return mine_if(b.op(spv::OpBitwiseAnd, t_u32, {a, z}), mine_ids.count(a) && mine_ids.count(z)); }
    Id bor(Id a, Id z) { return mine_if(b.op(spv::OpBitwiseOr, t_u32, {a, z}), mine_ids.count(a) || mine_ids.count(z)); }
    Id mine_if(Id x, bool yes) {
        if (yes) mine_ids.insert(x);
        return x;
    }
    Id bxor(Id a, Id z) { return b.op(spv::OpBitwiseXor, t_u32, {a, z}); }
    Id bnot(Id a) { return b.op(spv::OpNot, t_u32, {a}); }
    Id shl(Id a, Id n) { return b.op(spv::OpShiftLeftLogical, t_u32, {a, shift_amount(n)}); }
    Id shr(Id a, Id n) { return b.op(spv::OpShiftRightLogical, t_u32, {a, shift_amount(n)}); }
    Id sar(Id a, Id n) { return b.op(spv::OpShiftRightArithmetic, t_u32, {a, shift_amount(n)}); }
    Id ne0(Id a) { return b.op(spv::OpINotEqual, t_bool, {a, u(0)}); }
    Id b2u(Id c) { return sel(t_u32, c, u(1), u(0)); }
    Id tof(Id x) { return bc(t_f32, x); }
    Id tou(Id x) { return bc(t_u32, x); }

    Id sreg(uint16_t i) {
        if (i >= SGPRS) i = SNULL;
        if (!s[i]) s[i] = b.local(p_fn_u32, u(0)), registers.insert(s[i]);
        return s[i];
    }
    Id vreg(uint16_t i) { return vreg_of(h, i); }
    Id vreg_of(int half, uint16_t i) {
        Id& r = v[half & 1][i & 0xff];
        if (!r) r = b.local(p_fn_u32, u(0)), registers.insert(r);
        return r;
    }
    // A lane-mask SGPR as this half sees it: the pair's upper dword for the upper half.
    Id mask_reg(uint16_t lo) { return get_s(wide && lo != SNULL ? uint16_t(lo + h) : lo); }
    Id mask_src(const Operand& o) {
        if (wide && o.valid && !o.vgpr() && (o.code <= 105 || o.code == VCC_LO || o.code == EXEC_LO)) return get_s(uint16_t(o.code + h));
        return raw(o);
    }
    // Runs `f` as if emitting for half `half` (wide mode reads of the other half's lanes).
    template <class F>
    auto as_half(int half, F f) {
        int was = h;
        h = half;
        auto r = f();
        h = was;
        return r;
    }
    // SGPRs whose value has this invocation's lane bit set, and values known to: an EXEC_LO with
    // the bit makes every lane active, so writes skip the select against the old value.
    std::bitset<SGPRS> mine_reg;
    bool no_lane_facts = false;
    std::unordered_set<Id> mine_ids;
    Id get_s(uint16_t i) {
        if (i == SNULL) return u(0);
        Id x = load(sreg(i), t_u32);
        if (i < SGPRS && mine_reg[i]) mine_ids.insert(x);
        return x;
    }
    void set_s(uint16_t i, Id x) {
        if (i == EXEC_LO || i == EXEC_HI) ++exec_gen;
        if (i != SNULL && i < SGPRS) store(sreg(i), x), mine_reg[i] = mine_ids.count(x) > 0;
    }
    // Both dwords of a lane mask ORed: zero when no lane of the 64 is set (the upper one stays 0 for wave32).
    // (wave32 code may use vcc_hi as a plain SGPR, so only wave64 programs read the upper dword.)
    Id mask64(uint16_t lo) { return p.wave64 ? bor(get_s(lo), get_s(uint16_t(lo + 1))) : get_s(lo); }
    Id get_scc() { return load(scc, t_bool); }
    void set_scc(Id c) { store(scc, c); }
    Id get_v(uint16_t i) { return load(vreg(i), t_u32); }

    // A 9-bit source as u32 (no float modifiers applied).
    Id raw(const Operand& o) {
        uint16_t c = o.code;
        if (c >= 256) return get_v(uint16_t(c - 256));
        if (c <= 105 || c == VCC_LO || c == VCC_HI || c == M0 || c == EXEC_LO || c == EXEC_HI || (c >= 108 && c <= 123))
            return get_s(c);
        if (c == SNULL) return u(0);
        if (c >= 128 && c <= 192) return u(c - 128);
        if (c >= 193 && c <= 208) return u(uint32_t(-(int(c) - 192)));
        static const float fl[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.15915494f};
        if (c >= 240 && c <= 248) {
            uint32_t w;
            memcpy(&w, &fl[c - 240], 4);
            return u(w);
        }
        if (c == 255) return u(o.literal);
        if (c == 253) return b2u(get_scc());
        if (c == 251) return b2u(b.op(spv::OpIEqual, t_bool, {mask64(VCC_LO), u(0)}));  // vccz
        if (c == 252) return b2u(b.op(spv::OpIEqual, t_bool, {mask64(EXEC_LO), u(0)}));  // execz
        return u(0);
    }
    Id fmods(Id x, const Operand& o) {
        if (o.abs) x = ext(GLSLstd450FAbs, t_f32, {x});
        if (o.neg) x = b.op(spv::OpFNegate, t_f32, {x});
        return x;
    }
    Id fsrc(const Operand& o) {
        Id x = tof(raw(o));
        if (o.abs) x = ext(GLSLstd450FAbs, t_f32, {x});
        if (o.neg) x = b.op(spv::OpFNegate, t_f32, {x});
        return x;
    }
    // 64-bit scalar pair as u64.
    Id raw64(const Operand& o) {
        if (o.code <= 105 || o.code == VCC_LO || o.code == EXEC_LO || (o.code >= 108 && o.code <= 123)) {
            Id lo = b.op(spv::OpUConvert, t_u64, {get_s(o.code)});
            Id hi = b.op(spv::OpUConvert, t_u64, {get_s(uint16_t(o.code + 1))});
            return b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
        }
        Id x = b.op(spv::OpUConvert, t_u64, {raw(o)});
        if (o.code >= 193 && o.code <= 208) x = b.op(spv::OpSConvert, t_u64, {bc(t_i32, raw(o))});
        return x;
    }
    // A 64-bit scalar source as its two dwords (as raw64 reads it).
    std::pair<Id, Id> dwords(const Operand& o) {
        if (o.code <= 105 || o.code == VCC_LO || o.code == EXEC_LO || (o.code >= 108 && o.code <= 123)) return {get_s(o.code), get_s(uint16_t(o.code + 1))};
        return {raw(o), o.code >= 193 && o.code <= 208 ? u(~0u) : u(0)};
    }
    void set_s64(uint16_t i, Id x64) {
        set_s(i, b.op(spv::OpUConvert, t_u32, {x64}));
        set_s(uint16_t(i + 1), b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {x64, b.c_u32(32)})}));
    }

    // 64-bit vector operand (VGPR pair, SGPR pair or constant) as u64.
    Id vraw64(const Operand& o) {
        if (o.vgpr()) {
            Id lo = b.op(spv::OpUConvert, t_u64, {get_v(o.vgpr_index())});
            Id hi = b.op(spv::OpUConvert, t_u64, {get_v(uint16_t(o.vgpr_index() + 1))});
            return b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
        }
        return raw64(o);
    }
    void set_v64(uint16_t i, Id x64) {
        set_v(i, b.op(spv::OpUConvert, t_u32, {x64}));
        set_v(uint16_t(i + 1), b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {x64, b.c_u32(32)})}));
    }
    Id half(Id x, bool hi) {  // f16 half of a dword as f32
        Id h = hi ? shr(x, u(16)) : band(x, u(0xffff));
        return b.op(spv::OpCompositeExtract, t_f32, {ext(GLSLstd450UnpackHalf2x16, t_v2f, {h}), 0});
    }
    Id pack_half(Id lo, Id hi) { return ext(GLSLstd450PackHalf2x16, t_u32, {b.op(spv::OpCompositeConstruct, t_v2f, {lo, hi})}); }

    // Lane state.
    Id act_value[2] = {};
    uint32_t act_block[2] = {~0u, ~0u}, act_gen[2] = {~0u, ~0u}, exec_gen = 0;
    Id active() {
        if (!wide && mine_reg[EXEC_LO]) return b.c_bool(true);
        if (act_gen[h] == exec_gen && act_block[h] == b.labels) return act_value[h];
        act_gen[h] = exec_gen, act_block[h] = b.labels;
        return act_value[h] = mask_bit(mask_reg(EXEC_LO));
    }
    void set_v(uint16_t i, Id x) {
        if (!wide && mine_reg[EXEC_LO] && !dpp_write && !(sdwa && sdwa->sdwa_dst_sel != 6)) return store(vreg(i), x);
        Id old = get_v(i);
        Id act = active();
        if (dpp_write) act = b.op(spv::OpLogicalAnd, t_bool, {act, dpp_write});
        if (sdwa && sdwa->sdwa_dst_sel != 6) {
            uint32_t s_ = sdwa->sdwa_dst_sel;
            uint32_t shift = s_ <= 3 ? 8 * s_ : 16 * (s_ - 4), width = s_ <= 3 ? 8 : 16;
            Id part = b.op(spv::OpBitFieldUExtract, t_u32, {x, u(0), u(width)});
            Id placed = shl(part, u(shift));
            Id mask = u(((width == 32 ? 0xffffffffu : ((1u << width) - 1)) << shift));
            Id rest = sdwa->sdwa_dst_unused == 2 ? band(old, bnot(mask)) : u(0);  // 2 = preserve, else zero
            if (sdwa->sdwa_dst_unused == 1 && shift + width < 32)  // sign-extend upward
                rest = band(bnot(mask), sel(t_u32, ne0(band(part, u(1u << (width - 1)))), u(~0u << (shift + width)), u(0)));
            x = bor(rest, placed);
        }
        store(vreg(i), sel(t_u32, act, x, old));
    }
    Id ballot(Id cond) {
        Id m = b.op(spv::OpGroupNonUniformBallot, t_v4u, {u(spv::ScopeSubgroup), cond});
        return b.op(spv::OpCompositeExtract, t_u32, {m, 0});
    }
    // A per-lane condition into a wave mask register (VOPC semantics: inactive lanes read 0).
    void set_mask(uint16_t sdst, Id cond) {
        Id m = ballot(!wide && mine_reg[EXEC_LO] ? cond : b.op(spv::OpLogicalAnd, t_bool, {cond, active()}));
        if (wide) return set_s(sdst == SNULL ? sdst : uint16_t(sdst + h), m);
        set_s(sdst, m);
        if (sdst != SNULL && p.wave64) set_s(uint16_t(sdst + 1), u(0));
    }
    Id mask_bit(Id mask) { return ne0(band(mask, lane_bit)); }
    Id fclamp(Id x, bool clamp) { return clamp ? ext(GLSLstd450FClamp, t_f32, {x, f(0), f(1)}) : x; }
    Id omod(Id x, uint8_t m) {
        if (m == 1) return b.op(spv::OpFMul, t_f32, {x, f(2)});
        if (m == 2) return b.op(spv::OpFMul, t_f32, {x, f(4)});
        if (m == 3) return b.op(spv::OpFMul, t_f32, {x, f(0.5f)});
        return x;
    }

