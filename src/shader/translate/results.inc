// keylime: instruction results and shared operand handling (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    // ------------------------------------------------------- the code ----
    bool decode_all() {
        auto code = p.patched_code();
        uint32_t at = 0;
        std::set<uint32_t> leaders = {0};
        bool ended = false;
        while (at < code.size()) {
            Inst i;
            if (!gfx10::decode(code.data(), code.size(), at, i)) {
                if (ended) break;  // data after the program's end
                char e[64];
                snprintf(e, sizeof e, "undecodable instruction at 0x%x", at);
                error = e;
                return false;
            }
            insts.push_back(i);
            uint32_t target;
            if (gfx10::branch_target(i, target)) leaders.insert(target), leaders.insert(at + i.size);
            if (gfx10::ends_program(i)) {
                ended = true;
                leaders.insert(at + i.size);
                // Code after the end may be data or unreachable padding; keep decoding while it decodes.
            }
            at += i.size;
        }
        uint32_t n = 0;
        for (uint32_t l : leaders) block_of[l] = n++;
        return true;
    }

    void unsupported(const Inst& i) { ++out.unsupported[i.name]; }

    // Writes a VALU result (VOP1/2/3), honouring clamp / omod for floats.
    void vres_f(const Inst& i, Id x) { set_v(i.dst.vgpr_index(), tou(fclamp(omod(x, i.omod), i.clamp))); }
    void vres_u(const Inst& i, Id x) { set_v(i.dst.vgpr_index(), x); }

    // v_cmp_class: whether the float's class is one the mask names (bit 0 signaling NaN, 1 quiet
    // NaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf).
    Id fclass(Id bits, Id mask, const Operand& o) {
        if (o.abs) bits = band(bits, u(0x7fffffff));
        if (o.neg) bits = bxor(bits, u(0x80000000));
        Id e = band(shr(bits, u(23)), u(0xff)), m = band(bits, u(0x7fffff));
        Id neg = ne0(shr(bits, u(31)));
        auto eq = [&](Id a, uint32_t k) { return b.op(spv::OpIEqual, t_bool, {a, u(k)}); };
        Id e_max = eq(e, 255), e_zero = eq(e, 0), m_zero = eq(m, 0);
        auto side = [&](uint32_t nk, uint32_t pk) { return sel(t_u32, neg, u(nk), u(pk)); };
        Id idx = sel(t_u32, e_max, sel(t_u32, m_zero, side(2, 9), sel(t_u32, ne0(band(m, u(0x400000))), u(1), u(0))),
                     sel(t_u32, e_zero, sel(t_u32, m_zero, side(5, 6), side(4, 7)), side(3, 8)));
        return ne0(band(shr(mask, idx), u(1)));
    }

    bool vcmp(const Inst& i, std::string_view n) {
        // v_cmp{x}_<op>_<type>
        bool x = n.rfind("v_cmpx_", 0) == 0;
        std::string_view rest = n.substr(x ? 7 : 6);
        size_t us = rest.rfind('_');
        if (us == std::string_view::npos) return false;
        std::string_view op = rest.substr(0, us), ty = rest.substr(us + 1);
        Id c = 0;
        if (ty == "f32") {
            Id a = fsrc(i.src[0]), z = fsrc(i.src[1]);
            static const std::pair<const char*, spv::Op> ops[] = {
                {"lt", spv::OpFOrdLessThan}, {"eq", spv::OpFOrdEqual}, {"le", spv::OpFOrdLessThanEqual},
                {"gt", spv::OpFOrdGreaterThan}, {"lg", spv::OpFOrdNotEqual}, {"ge", spv::OpFOrdGreaterThanEqual},
                {"o", spv::OpOrdered}, {"u", spv::OpUnordered}, {"nge", spv::OpFUnordLessThan}, {"nlg", spv::OpFUnordEqual},
                {"ngt", spv::OpFUnordLessThanEqual}, {"nle", spv::OpFUnordGreaterThan}, {"neq", spv::OpFUnordNotEqual},
                {"nlt", spv::OpFUnordGreaterThanEqual}};
            if (op == "f") c = b.c_bool(false);
            else if (op == "tru" || op == "t") c = b.c_bool(true);
            else
                for (auto& [nm, o] : ops)
                    if (op == nm) c = b.op(o, t_bool, {a, z});
            if (op == "class") c = fclass(raw(i.src[0]), raw(i.src[1]), i.src[0]);
        } else if (ty == "u32" || ty == "i32") {
            Id a = raw(i.src[0]), z = raw(i.src[1]);
            bool s = ty == "i32";
            if (s) a = bc(t_i32, a), z = bc(t_i32, z);
            if (op == "lt") c = b.op(s ? spv::OpSLessThan : spv::OpULessThan, t_bool, {a, z});
            else if (op == "le") c = b.op(s ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, {a, z});
            else if (op == "gt") c = b.op(s ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, {a, z});
            else if (op == "ge") c = b.op(s ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, {a, z});
            else if (op == "eq") c = b.op(spv::OpIEqual, t_bool, {a, z});
            else if (op == "ne" || op == "lg") c = b.op(spv::OpINotEqual, t_bool, {a, z});
            else if (op == "f") c = b.c_bool(false);
            else if (op == "t") c = b.c_bool(true);
        } else if (ty == "u64" || ty == "i64" || ty == "u16" || ty == "i16" || ty == "f16") {
            bool s = ty[0] == 'i';
            Id a, z, T;
            if (ty == "f16") {
                Id fa = half(raw(i.src[0]), i.opsel & 1), fz = half(raw(i.src[1]), i.opsel & 2);
                if (op == "lt") c = b.op(spv::OpFOrdLessThan, t_bool, {fa, fz});
                else if (op == "gt") c = b.op(spv::OpFOrdGreaterThan, t_bool, {fa, fz});
                else if (op == "le") c = b.op(spv::OpFOrdLessThanEqual, t_bool, {fa, fz});
                else if (op == "ge") c = b.op(spv::OpFOrdGreaterThanEqual, t_bool, {fa, fz});
                else if (op == "eq") c = b.op(spv::OpFOrdEqual, t_bool, {fa, fz});
                else if (op == "neq" || op == "lg") c = b.op(spv::OpFUnordNotEqual, t_bool, {fa, fz});
            } else {
                if (ty == "u64" || ty == "i64") {
                    a = vraw64(i.src[0]), z = vraw64(i.src[1]);
                    T = s ? b.t_int(64, true) : t_u64;
                    if (s) a = bc(T, a), z = bc(T, z);
                } else {
                    a = b.op(spv::OpBitFieldUExtract, t_u32, {raw(i.src[0]), u(0), u(16)});
                    z = b.op(spv::OpBitFieldUExtract, t_u32, {raw(i.src[1]), u(0), u(16)});
                    if (s) a = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, a), u(0), u(16)}), z = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, z), u(0), u(16)});
                }
                if (op == "lt") c = b.op(s ? spv::OpSLessThan : spv::OpULessThan, t_bool, {a, z});
                else if (op == "le") c = b.op(s ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, {a, z});
                else if (op == "gt") c = b.op(s ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, {a, z});
                else if (op == "ge") c = b.op(s ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, {a, z});
                else if (op == "eq") c = b.op(spv::OpIEqual, t_bool, {a, z});
                else if (op == "ne" || op == "lg") c = b.op(spv::OpINotEqual, t_bool, {a, z});
            }
        } else if (ty == "f64") {
            return false;
        }
        if (!c) return false;
        if (x) {
            Id m = ballot(b.op(spv::OpLogicalAnd, t_bool, {c, active()}));
            set_s(uint16_t(EXEC_LO + (wide ? h : 0)), m);
            if (i.enc == Enc::VOP3 && i.sdst.valid) set_s(uint16_t(i.sdst.code + (wide ? h : 0)), m);
        } else {
            uint16_t sd = i.enc == Enc::VOP3 ? i.sdst.code : VCC_LO;
            set_mask(sd, c);
        }
        return true;
    }

    // SDWA source selection: byte 0-3, word 0-1, dword; zero- or sign-extended.
    Id sdwa_src(Id x, uint32_t s_, bool sext) {
        uint32_t shift = s_ <= 3 ? 8 * s_ : 16 * (s_ - 4), width = s_ <= 3 ? 8 : 16;
        if (sext) return tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, x), u(shift), u(width)}));
        return b.op(spv::OpBitFieldUExtract, t_u32, {x, u(shift), u(width)});
    }
    // DPP: src0 read from another lane of its row (16 lanes); sets dpp_write for lanes that write.
    Id dpp_src(const Inst& i) {
        Id src = i.src[0].vgpr() ? get_v(i.src[0].vgpr_index()) : raw(i.src[0]);
        Id from, valid = b.c_bool(true);
        Id row_lane = band(lane, u(15)), row_base = band(lane, u(~15u));
        if (i.dpp8) {
            Id sh = b.op(spv::OpIMul, t_u32, {band(lane, u(7)), u(3)});
            from = bor(band(lane, u(~7u)), band(shr(u(i.dpp_sel8), sh), u(7)));
        } else {
            uint32_t c = i.dpp_ctrl;
            if (c <= 0xff) {
                Id sh = shl(band(lane, u(3)), u(1));
                from = bor(band(lane, u(~3u)), band(shr(u(c), sh), u(3)));
            } else if (c >= 0x101 && c <= 0x10f) {  // row_shl: lane i reads i + n
                uint32_t n = c & 0xf;
                from = iadd(lane, u(n));
                valid = b.op(spv::OpULessThanEqual, t_bool, {iadd(row_lane, u(n)), u(15)});
            } else if (c >= 0x111 && c <= 0x11f) {  // row_shr: lane i reads i - n
                uint32_t n = c & 0xf;
                from = isub(lane, u(n));
                valid = b.op(spv::OpUGreaterThanEqual, t_bool, {row_lane, u(n)});
            } else if (c >= 0x121 && c <= 0x12f) {  // row_ror
                uint32_t n = c & 0xf;
                from = bor(row_base, band(isub(iadd(row_lane, u(16)), u(n)), u(15)));
            } else if (c == 0x140) {
                from = bor(row_base, isub(u(15), row_lane));
            } else if (c == 0x141) {
                from = bor(band(lane, u(~7u)), isub(u(7), band(lane, u(7))));
            } else if (c >= 0x150 && c <= 0x15f) {  // row_share
                from = bor(row_base, u(c & 0xf));
            } else if (c >= 0x160 && c <= 0x16f) {  // row_xmask
                from = bor(row_base, bxor(row_lane, u(c & 0xf)));
            } else {
                from = lane;
            }
            // Row / bank masks: lanes of masked-off rows or banks keep their destination.
            if (i.dpp_row_mask != 0xf || i.dpp_bank_mask != 0xf) {
                Id row_on = ne0(band(shr(u(i.dpp_row_mask), iadd(shr(lane, u(4)), u(uint32_t(2 * h)))), u(1)));
                Id bank_on = ne0(band(shr(u(i.dpp_bank_mask), band(shr(lane, u(2)), u(3))), u(1)));
                dpp_write = b.op(spv::OpLogicalAnd, t_bool, {row_on, bank_on});
            }
        }
        Id moved = b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), src, band(from, u(31))});
        if (!i.dpp8) {
            // An invalid source reads 0 with bound_ctrl, else the lane keeps its destination.
            if (i.dpp_bound_ctrl) moved = sel(t_u32, valid, moved, u(0));
            else dpp_write = dpp_write ? b.op(spv::OpLogicalAnd, t_bool, {dpp_write, valid}) : valid;
        }
        return moved;
    }

    // One instruction inside a block. Returns false only on a hard error.
