// keylime: scalar and vector ALU instructions (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    void alu(const Inst& i) {
        std::string_view n = i.name;
        auto S0 = [&] { return ov[0] ? ov[0] : raw(i.src[0]); };
        auto S1 = [&] { return ov[1] ? ov[1] : raw(i.src[1]); };
        auto S2 = [&] { return ov[2] ? ov[2] : raw(i.src[2]); };
        auto F0 = [&] { return ov[0] ? fmods(tof(ov[0]), i.src[0]) : fsrc(i.src[0]); };
        auto F1 = [&] { return ov[1] ? fmods(tof(ov[1]), i.src[1]) : fsrc(i.src[1]); };
        auto F2 = [&] { return ov[2] ? fmods(tof(ov[2]), i.src[2]) : fsrc(i.src[2]); };
        auto fop = [&](spv::Op o) { vres_f(i, b.op(o, t_f32, {F0(), F1()})); };
        auto iop = [&](spv::Op o) { vres_u(i, b.op(o, t_u32, {S0(), S1()})); };
        uint16_t sd = i.dst.code;
        auto sres = [&](Id x) {  // SALU 32-bit result; SCC = result != 0 for logic ops
            set_s(sd, x);
        };
        auto slogic = [&](spv::Op o, bool b64, bool inv0 = false, bool inv1 = false, bool invr = false) {
            if (!b64) {
                Id a = S0(), z = S1();
                if (inv0) a = bnot(a);
                if (inv1) z = bnot(z);
                Id r = o == spv::OpBitwiseAnd ? band(a, z) : o == spv::OpBitwiseOr ? bor(a, z) : b.op(o, t_u32, {a, z});
                if (invr) r = bnot(r);
                sres(r);
                set_scc(ne0(r));
            } else {
                auto [a0, a1] = dwords(i.src[0]);
                auto [z0, z1] = dwords(i.src[1]);
                auto one = [&](Id a, Id z) {
                    if (inv0) a = bnot(a);
                    if (inv1) z = bnot(z);
                    Id r = o == spv::OpBitwiseAnd ? band(a, z) : o == spv::OpBitwiseOr ? bor(a, z) : b.op(o, t_u32, {a, z});
                    return invr ? bnot(r) : r;
                };
                Id r0 = one(a0, z0), r1 = one(a1, z1);
                set_s(sd, r0);
                set_s(uint16_t(sd + 1), r1);
                set_scc(ne0(bor(r0, r1)));
            }
        };
        // ---- scalar ALU ----
        if (i.enc == Enc::SOP1 || i.enc == Enc::SOP2 || i.enc == Enc::SOPK || i.enc == Enc::SOPC) {
            if (n.rfind("s_waitcnt", 0) == 0 || n == "s_version" || n == "s_subvector_loop_begin" || n == "s_subvector_loop_end") return;
            if (n == "s_mov_b32") return sres(S0());
            if (n == "s_mov_b64") {
                auto [lo, hi] = dwords(i.src[0]);
                set_s(sd, lo);
                return set_s(uint16_t(sd + 1), hi);
            }
            if (n == "s_movk_i32") return sres(u(uint32_t(int32_t(int16_t(i.simm16)))));
            if (n == "s_cmov_b32") return sres(sel(t_u32, get_scc(), S0(), get_s(sd)));
            if (n == "s_not_b32") { Id r = bnot(S0()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_and_b32") return slogic(spv::OpBitwiseAnd, false);
            if (n == "s_and_b64") return slogic(spv::OpBitwiseAnd, true);
            if (n == "s_or_b32") return slogic(spv::OpBitwiseOr, false);
            if (n == "s_or_b64") return slogic(spv::OpBitwiseOr, true);
            if (n == "s_xor_b32") return slogic(spv::OpBitwiseXor, false);
            if (n == "s_xor_b64") return slogic(spv::OpBitwiseXor, true);
            if (n == "s_andn2_b32") return slogic(spv::OpBitwiseAnd, false, false, true);
            if (n == "s_andn2_b64") return slogic(spv::OpBitwiseAnd, true, false, true);
            if (n == "s_orn2_b32") return slogic(spv::OpBitwiseOr, false, false, true);
            if (n == "s_orn2_b64") return slogic(spv::OpBitwiseOr, true, false, true);
            if (n == "s_nand_b32") return slogic(spv::OpBitwiseAnd, false, false, false, true);
            if (n == "s_nor_b32") return slogic(spv::OpBitwiseOr, false, false, false, true);
            if (n == "s_xnor_b32") return slogic(spv::OpBitwiseXor, false, false, false, true);
            if (n == "s_add_u32" || n == "s_add_i32") {
                Id a = S0(), z = S1(), r = iadd(a, z);
                sres(r);
                return set_scc(n == "s_add_u32" ? b.op(spv::OpULessThan, t_bool, {r, a}) : b.c_bool(false));
            }
            if (n == "s_addc_u32") {
                Id a = S0(), z = S1(), c = b2u(get_scc());
                Id r = iadd(iadd(a, z), c);
                sres(r);
                return set_scc(b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpULessThan, t_bool, {r, a}),
                                                               b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {r, a}), ne0(c)})}));
            }
            if (n == "s_sub_u32" || n == "s_sub_i32") {
                Id a = S0(), z = S1();
                sres(isub(a, z));
                return set_scc(n == "s_sub_u32" ? b.op(spv::OpULessThan, t_bool, {a, z}) : b.c_bool(false));
            }
            if (n == "s_subb_u32") return sres(isub(isub(S0(), S1()), b2u(get_scc())));
            if (n == "s_mul_i32") return sres(b.op(spv::OpIMul, t_u32, {S0(), S1()}));
            if (n == "s_addk_i32") return sres(iadd(get_s(sd), u(uint32_t(int32_t(int16_t(i.simm16))))));
            if (n == "s_mulk_i32") return sres(b.op(spv::OpIMul, t_u32, {get_s(sd), u(uint32_t(int32_t(int16_t(i.simm16))))}));
            if (n == "s_lshl_b32") { Id r = shl(S0(), S1()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_lshr_b32") { Id r = shr(S0(), S1()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_ashr_i32") { Id r = sar(S0(), S1()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_lshl_b64") {
                Id r = b.op(spv::OpShiftLeftLogical, t_u64, {raw64(i.src[0]), band(S1(), u(63))});
                set_s64(sd, r);
                return set_scc(b.op(spv::OpINotEqual, t_bool, {r, b.c_u64(0)}));
            }
            if (n == "s_lshr_b64") {
                Id r = b.op(spv::OpShiftRightLogical, t_u64, {raw64(i.src[0]), band(S1(), u(63))});
                set_s64(sd, r);
                return set_scc(b.op(spv::OpINotEqual, t_bool, {r, b.c_u64(0)}));
            }
            if (n.rfind("s_lshl", 0) == 0 && n.find("_add_u32") != std::string_view::npos) {  // s_lshl1..4_add_u32
                Id r = iadd(shl(S0(), u(uint32_t(n[6] - '0'))), S1());
                sres(r);
                return set_scc(b.c_bool(false));
            }
            if (n == "s_min_u32" || n == "s_max_u32" || n == "s_min_i32" || n == "s_max_i32") {
                bool mn = n[3] == 'i', sg = n.back() == '2' && n[n.size() - 3] == 'i';
                uint32_t op = sg ? (mn ? GLSLstd450SMin : GLSLstd450SMax) : (mn ? GLSLstd450UMin : GLSLstd450UMax);
                Id r = ext(op, t_u32, {S0(), S1()});
                sres(r);
                return set_scc(b.op(spv::OpIEqual, t_bool, {r, S0()}));
            }
            if (n == "s_cselect_b32") return sres(sel(t_u32, get_scc(), S0(), S1()));
            if (n == "s_cselect_b64") return set_s64(sd, sel(t_u64, get_scc(), raw64(i.src[0]), raw64(i.src[1])));
            if (n == "s_bfe_u32" || n == "s_bfe_i32") {
                Id off = band(S1(), u(31)), w = band(shr(S1(), u(16)), u(0x7f));
                Id r = n == "s_bfe_u32" ? b.op(spv::OpBitFieldUExtract, t_u32, {S0(), off, w})
                                        : tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), off, w}));
                sres(r);
                return set_scc(ne0(r));
            }
            if (n == "s_bfm_b32") return sres(shl(isub(shl(u(1), S0()), u(1)), S1()));
            if (n == "s_brev_b32") return sres(b.op(spv::OpBitReverse, t_u32, {S0()}));
            if (n == "s_bcnt1_i32_b32") { Id r = b.op(spv::OpBitCount, t_u32, {S0()}); sres(r); return set_scc(ne0(r)); }
            if (n == "s_ff1_i32_b32") return sres(tou(ext(GLSLstd450FindILsb, t_i32, {S0()})));
            if (n == "s_ff1_i32_b64") {
                Id lo = get_s(i.src[0].code), hi = get_s(uint16_t(i.src[0].code + 1));
                Id lsb_lo = tou(ext(GLSLstd450FindILsb, t_i32, {lo})), lsb_hi = tou(ext(GLSLstd450FindILsb, t_i32, {hi}));
                return sres(sel(t_u32, ne0(lo), lsb_lo, sel(t_u32, ne0(hi), iadd(lsb_hi, u(32)), u(~0u))));
            }
            if (n == "s_flbit_i32_b32") {
                Id m = ext(GLSLstd450FindUMsb, t_i32, {S0()});
                return sres(sel(t_u32, b.op(spv::OpSLessThan, t_bool, {m, b.c_i32(0)}), u(~0u), isub(u(31), tou(m))));
            }
            if (n == "s_sext_i32_i16") return sres(tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(16)})));
            if (n == "s_sext_i32_i8") return sres(tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(8)})));
            if (n == "s_pack_ll_b32_b16") return sres(bor(band(S0(), u(0xffff)), shl(S1(), u(16))));
            if (n == "s_pack_lh_b32_b16") return sres(bor(band(S0(), u(0xffff)), band(S1(), u(0xffff0000))));
            if (n == "s_pack_hh_b32_b16") return sres(bor(shr(S0(), u(16)), band(S1(), u(0xffff0000))));
            if (n.find("_saveexec_b32") != std::string_view::npos || n.find("_saveexec_b64") != std::string_view::npos ||
                n.find("_wrexec_b32") != std::string_view::npos || n.find("_wrexec_b64") != std::string_view::npos) {
                bool wrexec = n.find("wrexec") != std::string_view::npos;
                // 64-bit forms on a wave64 program work on both dwords (EXEC_HI is real there).
                bool both = n.find("_b64") != std::string_view::npos && p.wave64;
                auto combine = [&](Id s0v, Id e) -> Id {
                    if (n.rfind("s_and_", 0) == 0) return band(s0v, e);
                    if (n.rfind("s_or_", 0) == 0) return bor(s0v, e);
                    if (n.rfind("s_xor_", 0) == 0) return bxor(s0v, e);
                    if (n.rfind("s_andn2_", 0) == 0) return band(s0v, bnot(e));
                    if (n.rfind("s_orn2_", 0) == 0) return bor(s0v, bnot(e));
                    if (n.rfind("s_andn1_", 0) == 0) return band(bnot(s0v), e);
                    if (n.rfind("s_nand_", 0) == 0) return bnot(band(s0v, e));
                    if (n.rfind("s_nor_", 0) == 0) return bnot(bor(s0v, e));
                    if (n.rfind("s_xnor_", 0) == 0) return bnot(bxor(s0v, e));
                    if (n.rfind("s_orn1_", 0) == 0) return bor(bnot(s0v), e);
                    return 0;
                };
                Id e_lo = get_s(EXEC_LO), e_hi = both ? get_s(EXEC_HI) : u(0);
                Id s_lo = S0(), s_hi = both ? (i.src[0].code <= 105 || i.src[0].code == VCC_LO || i.src[0].code == EXEC_LO
                                                   ? get_s(uint16_t(i.src[0].code + 1))
                                                   : (i.src[0].code >= 193 && i.src[0].code <= 208 ? u(~0u) : u(0)))
                                            : u(0);
                Id r_lo = combine(s_lo, e_lo);
                if (!r_lo) return unsupported(i);
                Id r_hi = both ? combine(s_hi, e_hi) : u(0);
                if (!wrexec) {
                    sres(e_lo);
                    if (n.find("_b64") != std::string_view::npos) set_s(uint16_t(sd + 1), e_hi);
                } else {
                    sres(r_lo);
                    if (both) set_s(uint16_t(sd + 1), r_hi);
                }
                set_s(EXEC_LO, r_lo);
                if (both) set_s(EXEC_HI, r_hi);
                return set_scc(ne0(bor(r_lo, r_hi)));
            }
            if (n == "s_wqm_b32" || n == "s_wqm_b64") {
                // Any lane of a quad -> the whole quad.
                auto wqm = [&](Id x) {
                    Id q = bor(bor(x, shr(x, u(1))), bor(shr(x, u(2)), shr(x, u(3))));
                    return b.op(spv::OpIMul, t_u32, {band(q, u(0x11111111)), u(0xf)});
                };
                Id q = wqm(S0()), qh = u(0);
                if (n == "s_wqm_b64" && p.wave64) qh = wqm(get_s(uint16_t(i.src[0].code + 1)));
                sres(q);
                if (n == "s_wqm_b64") set_s(uint16_t(sd + 1), qh);
                return set_scc(ne0(bor(q, qh)));
            }
            if (n == "s_getpc_b64") return set_s64(sd, b.c_u64(uint64_t(i.offset + 4)));
            if (n == "s_swappc_b64" && p.stage == HwStage::Vs) {
                // The fetch shader call that opens every vertex shader: attributes are already in
                // their VGPRs (prologue), so the call is a no-op; the return address is what it saves.
                return set_s64(sd, b.c_u64(uint64_t(i.offset + 4)));
            }
            if (n == "s_bcnt1_i32_b64") {
                Id r = iadd(b.op(spv::OpBitCount, t_u32, {get_s(i.src[0].code)}), b.op(spv::OpBitCount, t_u32, {get_s(uint16_t(i.src[0].code + 1))}));
                sres(r);
                return set_scc(ne0(r));
            }
            if (n == "s_bitset1_b32") return sres(bor(get_s(sd), shl(u(1), S0())));
            if (n == "s_bitset0_b32") return sres(band(get_s(sd), bnot(shl(u(1), S0()))));
            if (n == "s_mul_hi_u32") {
                Id m = b.op(spv::OpIMul, t_u64, {b.op(spv::OpUConvert, t_u64, {S0()}), b.op(spv::OpUConvert, t_u64, {S1()})});
                return sres(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {m, b.c_u32(32)})}));
            }
            if (n == "s_getreg_b32") return sres(u(0));
            if (n == "s_setreg_b32" || n == "s_setreg_imm32_b32") return;
            if (i.enc == Enc::SOPC || n.rfind("s_cmpk_", 0) == 0) {
                Id a = S0(), z = i.enc == Enc::SOPK ? u(uint32_t(uint16_t(i.simm16))) : S1();
                if (i.enc == Enc::SOPK) a = get_s(sd);
                std::string_view op = n.substr(n.rfind("s_cmp", 0) == 0 ? (n[5] == 'k' ? 7 : 6) : 0);
                bool s = op.find("_i32") != std::string_view::npos;
                if (s && i.enc == Enc::SOPK) z = u(uint32_t(int32_t(int16_t(i.simm16))));
                Id A = s ? bc(t_i32, a) : a, Z = s ? bc(t_i32, z) : z;
                Id c = 0;
                if (op.rfind("eq", 0) == 0) c = b.op(spv::OpIEqual, t_bool, {a, z});
                else if (op.rfind("lg", 0) == 0) c = b.op(spv::OpINotEqual, t_bool, {a, z});
                else if (op.rfind("gt", 0) == 0) c = b.op(s ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, {A, Z});
                else if (op.rfind("ge", 0) == 0) c = b.op(s ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, {A, Z});
                else if (op.rfind("lt", 0) == 0) c = b.op(s ? spv::OpSLessThan : spv::OpULessThan, t_bool, {A, Z});
                else if (op.rfind("le", 0) == 0) c = b.op(s ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, {A, Z});
                else if (n == "s_bitcmp0_b32") c = b.op(spv::OpIEqual, t_bool, {band(shr(a, z), u(1)), u(0)});
                else if (n == "s_bitcmp1_b32") c = ne0(band(shr(a, z), u(1)));
                if (c) return set_scc(c);
            }
            return unsupported(i);
        }
        if (i.enc == Enc::SOPP) {
            if (n == "s_nop" || n == "s_waitcnt" || n == "s_inst_prefetch" || n == "s_sendmsg" || n == "s_setprio" ||
                n == "s_ttracedata" || n == "s_sleep" || n == "s_icache_inv" || n == "s_clause" || n == "s_round_mode" ||
                n == "s_denorm_mode" || n == "s_waitcnt_depctr" || n == "s_waitcnt_vscnt" || n == "s_waitcnt_vmcnt" ||
                n == "s_waitcnt_lgkmcnt" || n == "s_waitcnt_expcnt" || n == "s_code_end" || n == "s_sethalt" || n == "s_trap")
                return;
            if (n == "s_sendmsg" && p.stage == HwStage::Ms && (i.simm16 & 0xf) == 9) return store(ms_count, get_s(M0));
            if (n == "s_barrier") {
                if (p.stage == HwStage::Cs || p.stage == HwStage::Ms)
                    b.op_void(spv::OpControlBarrier, {u(spv::ScopeWorkgroup), u(spv::ScopeWorkgroup),
                                                      u(spv::MemorySemanticsWorkgroupMemoryMask | spv::MemorySemanticsAcquireReleaseMask)});
                return;
            }
            return unsupported(i);
        }
        // ---- scalar memory ----
        if (i.enc == Enc::SMEM) {
            int count = 1;
            if (n.find("x2") != std::string_view::npos) count = 2;
            else if (n.find("x4") != std::string_view::npos) count = 4;
            else if (n.find("x8") != std::string_view::npos) count = 8;
            else if (n.find("x16") != std::string_view::npos) count = 16;
            bool buffer = n.rfind("s_buffer_load", 0) == 0;
            if (n.rfind("s_load_dword", 0) == 0 || buffer) {
                Operand base;
                base.code = i.sbase;
                Id addr = buffer ? vsharp_base(i.sbase) : raw64(base);
                Id off = u(uint32_t(i.smem_offset));
                if (i.soffset != SNULL) off = iadd(off, get_s(i.soffset));
                addr = add64(addr, off);
                auto words = mem_load_n(addr, count);
                for (int k = 0; k < count; ++k) set_s(uint16_t(i.sdata + k), words[k]);
                return;
            }
            if (n.rfind("s_dcache", 0) == 0 || n == "s_gl1_inv") return;
            if (n == "s_memtime" || n == "s_memrealtime") return set_s64(i.sdata, b.c_u64(0));
            return unsupported(i);
        }
        // ---- vector ALU ----
        if (i.enc == Enc::VOPC || (i.enc == Enc::VOP3 && n.rfind("v_cmp", 0) == 0)) {
            if (!vcmp(i, n)) unsupported(i);
            return;
        }
        if (i.dpp || i.sdwa) {
            Inst plain = i;
            plain.dpp = plain.sdwa = false;
            if (i.dpp) {
                ov[0] = dpp_src(i);
            } else {
                for (int k = 0; k < 2; ++k)
                    if (i.src[k].valid && i.sdwa_src_sel[k] != 6) ov[k] = sdwa_src(raw(i.src[k]), i.sdwa_src_sel[k], i.sdwa_sext[k]);
                sdwa = &i;
            }
            alu(plain);
            ov[0] = ov[1] = ov[2] = 0;
            dpp_write = 0;
            sdwa = nullptr;
            return;
        }
        if (n == "v_nop") return;
        if (n == "v_mov_b32") return vres_u(i, S0());
        if (n == "v_add_f32") return fop(spv::OpFAdd);
        if (n == "v_sub_f32") return fop(spv::OpFSub);
        if (n == "v_subrev_f32") return vres_f(i, b.op(spv::OpFSub, t_f32, {F1(), F0()}));
        if (n == "v_mul_f32") return fop(spv::OpFMul);
        if (n == "v_mul_legacy_f32" || n == "v_mul_dx9_zero_f32") {
            Id a = F0(), z = F1();
            Id zero = b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpFOrdEqual, t_bool, {a, f(0)}), b.op(spv::OpFOrdEqual, t_bool, {z, f(0)})});
            return vres_f(i, sel(t_f32, zero, f(0), b.op(spv::OpFMul, t_f32, {a, z})));
        }
        if (n == "v_mac_f32" || n == "v_fmac_f32" || n == "v_fmac_legacy_f32" || n == "v_mac_legacy_f32") {
            Id acc = tof(get_v(i.dst.vgpr_index()));
            return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), F1(), acc}));
        }
        if (n == "v_mad_f32" || n == "v_fma_f32" || n == "v_mad_legacy_f32" || n == "v_fma_legacy_f32")
            return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), F1(), F2()}));
        if (n == "v_fmamk_f32" || n == "v_madmk_f32") return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), tof(u(i.src[2].literal)), F1()}));
        if (n == "v_fmaak_f32" || n == "v_madak_f32") return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), F1(), tof(u(i.src[2].literal))}));
        if (n == "v_min_f32" || n == "v_min_legacy_f32") return vres_f(i, ext(GLSLstd450NMin, t_f32, {F0(), F1()}));
        if (n == "v_max_f32" || n == "v_max_legacy_f32") return vres_f(i, ext(GLSLstd450NMax, t_f32, {F0(), F1()}));
        if (n == "v_min3_f32") return vres_f(i, ext(GLSLstd450NMin, t_f32, {ext(GLSLstd450NMin, t_f32, {F0(), F1()}), F2()}));
        if (n == "v_max3_f32") return vres_f(i, ext(GLSLstd450NMax, t_f32, {ext(GLSLstd450NMax, t_f32, {F0(), F1()}), F2()}));
        if (n == "v_med3_f32") {
            Id a = F0(), z = F1(), c = F2();
            Id mx = ext(GLSLstd450NMax, t_f32, {ext(GLSLstd450NMin, t_f32, {a, z}), ext(GLSLstd450NMin, t_f32, {ext(GLSLstd450NMax, t_f32, {a, z}), c})});
            return vres_f(i, mx);
        }
        if (n == "v_rcp_f32" || n == "v_rcp_iflag_f32" || n == "v_rcp_legacy_f32") return vres_f(i, b.op(spv::OpFDiv, t_f32, {f(1), F0()}));
        if (n == "v_rsq_f32" || n == "v_rsq_legacy_f32") return vres_f(i, ext(GLSLstd450InverseSqrt, t_f32, {F0()}));
        if (n == "v_sqrt_f32") return vres_f(i, ext(GLSLstd450Sqrt, t_f32, {F0()}));
        if (n == "v_exp_f32" || n == "v_exp_legacy_f32") return vres_f(i, ext(GLSLstd450Exp2, t_f32, {F0()}));
        if (n == "v_log_f32" || n == "v_log_legacy_f32") return vres_f(i, ext(GLSLstd450Log2, t_f32, {F0()}));
        if (n == "v_sin_f32") return vres_f(i, ext(GLSLstd450Sin, t_f32, {b.op(spv::OpFMul, t_f32, {F0(), f(6.28318530718f)})}));
        if (n == "v_cos_f32") return vres_f(i, ext(GLSLstd450Cos, t_f32, {b.op(spv::OpFMul, t_f32, {F0(), f(6.28318530718f)})}));
        if (n == "v_fract_f32") return vres_f(i, ext(GLSLstd450Fract, t_f32, {F0()}));
        if (n == "v_floor_f32") return vres_f(i, ext(GLSLstd450Floor, t_f32, {F0()}));
        if (n == "v_ceil_f32") return vres_f(i, ext(GLSLstd450Ceil, t_f32, {F0()}));
        if (n == "v_trunc_f32") return vres_f(i, ext(GLSLstd450Trunc, t_f32, {F0()}));
        if (n == "v_rndne_f32") return vres_f(i, ext(GLSLstd450RoundEven, t_f32, {F0()}));
        if (n == "v_ldexp_f32") return vres_f(i, ext(GLSLstd450Ldexp, t_f32, {F0(), bc(t_i32, S1())}));
        if (n == "v_cvt_f32_u32") return vres_f(i, b.op(spv::OpConvertUToF, t_f32, {S0()}));
        if (n == "v_cvt_f32_i32") return vres_f(i, b.op(spv::OpConvertSToF, t_f32, {bc(t_i32, S0())}));
        if (n == "v_cvt_u32_f32") return vres_u(i, b.op(spv::OpConvertFToU, t_u32, {ext(GLSLstd450FClamp, t_f32, {F0(), f(0), f(4294967040.0f)})}));
        if (n == "v_cvt_i32_f32") return vres_u(i, tou(b.op(spv::OpConvertFToS, t_i32, {ext(GLSLstd450FClamp, t_f32, {F0(), f(-2147483648.0f), f(2147483520.0f)})})));
        if (n == "v_cvt_flr_i32_f32") return vres_u(i, tou(b.op(spv::OpConvertFToS, t_i32, {ext(GLSLstd450Floor, t_f32, {F0()})})));
        if (n == "v_cvt_rpi_i32_f32") return vres_u(i, tou(b.op(spv::OpConvertFToS, t_i32, {ext(GLSLstd450Floor, t_f32, {b.op(spv::OpFAdd, t_f32, {F0(), f(0.5f)})})})));
        if (n.rfind("v_cvt_f32_ubyte", 0) == 0) {
            uint32_t k = uint32_t(n.back() - '0');
            return vres_f(i, b.op(spv::OpConvertUToF, t_f32, {band(shr(S0(), u(8 * k)), u(0xff))}));
        }
        if (n == "v_cvt_f32_f16") {
            Id h = band(S0(), u(0xffff));
            if (i.opsel & 1) h = shr(S0(), u(16));
            Id v2 = ext(GLSLstd450UnpackHalf2x16, t_v2f, {h});
            return vres_f(i, b.op(spv::OpCompositeExtract, t_f32, {v2, 0}));
        }
        if (n == "v_cvt_f16_f32") {
            Id pk = ext(GLSLstd450PackHalf2x16, t_u32, {b.op(spv::OpCompositeConstruct, t_v2f, {F0(), f(0)})});
            Id old = get_v(i.dst.vgpr_index());
            Id r = (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(pk, u(16))) : bor(band(old, u(0xffff0000)), pk);
            return vres_u(i, r);
        }
        if (n == "v_cvt_pkrtz_f16_f32" || n == "v_cvt_pknorm_u16_f32" || n == "v_cvt_pknorm_i16_f32") {
            Id v2 = b.op(spv::OpCompositeConstruct, t_v2f, {F0(), F1()});
            uint32_t op = n == "v_cvt_pkrtz_f16_f32" ? GLSLstd450PackHalf2x16 : n == "v_cvt_pknorm_u16_f32" ? GLSLstd450PackUnorm2x16 : GLSLstd450PackSnorm2x16;
            return vres_u(i, ext(op, t_u32, {v2}));
        }
        if (n == "v_pack_b32_f16") return vres_u(i, bor(band(S0(), u(0xffff)), shl(S1(), u(16))));
        if (n == "v_cvt_pk_u16_u32") return vres_u(i, bor(band(S0(), u(0xffff)), shl(S1(), u(16))));
        if (n == "v_cndmask_b32") {
            Id mask = i.enc == Enc::VOP3 ? mask_src(i.src[2]) : mask_reg(VCC_LO);
            Id a = S0(), z = S1();
            if (i.enc == Enc::VOP3) {  // float modifiers apply to the selected values
                if (i.src[0].abs || i.src[0].neg) a = tou(F0());
                if (i.src[1].abs || i.src[1].neg) z = tou(F1());
            }
            return vres_u(i, sel(t_u32, mask_bit(mask), z, a));
        }
        if (n == "v_and_b32") return iop(spv::OpBitwiseAnd);
        if (n == "v_or_b32") return iop(spv::OpBitwiseOr);
        if (n == "v_xor_b32") return iop(spv::OpBitwiseXor);
        if (n == "v_not_b32") return vres_u(i, bnot(S0()));
        if (n == "v_xnor_b32") return vres_u(i, bnot(bxor(S0(), S1())));
        if (n == "v_lshlrev_b32") return vres_u(i, shl(S1(), S0()));
        if (n == "v_lshrrev_b32") return vres_u(i, shr(S1(), S0()));
        if (n == "v_ashrrev_i32") return vres_u(i, sar(S1(), S0()));
        if (n == "v_add_nc_u32" || n == "v_add_u32") return iop(spv::OpIAdd);
        if (n == "v_sub_nc_u32" || n == "v_sub_u32") return iop(spv::OpISub);
        if (n == "v_subrev_nc_u32" || n == "v_subrev_u32") return vres_u(i, isub(S1(), S0()));
        if (n == "v_mul_lo_u32" || n == "v_mul_lo_i32") return iop(spv::OpIMul);
        if (n == "v_mul_hi_u32" || n == "v_mul_hi_i32") {
            bool sg = n == "v_mul_hi_i32";
            Id t64 = sg ? b.t_int(64, true) : t_u64;
            Id a = sg ? b.op(spv::OpSConvert, t64, {bc(t_i32, S0())}) : b.op(spv::OpUConvert, t64, {S0()});
            Id z = sg ? b.op(spv::OpSConvert, t64, {bc(t_i32, S1())}) : b.op(spv::OpUConvert, t64, {S1()});
            Id m = b.op(spv::OpIMul, t64, {a, z});
            Id hi = b.op(sg ? spv::OpShiftRightArithmetic : spv::OpShiftRightLogical, t64, {m, b.c_u32(32)});
            return vres_u(i, sg ? tou(b.op(spv::OpSConvert, t_i32, {hi})) : b.op(spv::OpUConvert, t_u32, {hi}));
        }
        if (n == "v_mul_u32_u24") return vres_u(i, b.op(spv::OpIMul, t_u32, {band(S0(), u(0xffffff)), band(S1(), u(0xffffff))}));
        if (n == "v_mul_i32_i24") {
            Id a = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(24)});
            Id z = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S1()), u(0), u(24)});
            return vres_u(i, tou(b.op(spv::OpIMul, t_i32, {a, z})));
        }
        if (n == "v_mad_u32_u24") return vres_u(i, iadd(b.op(spv::OpIMul, t_u32, {band(S0(), u(0xffffff)), band(S1(), u(0xffffff))}), S2()));
        if (n == "v_mad_u32_u16") return vres_u(i, iadd(b.op(spv::OpIMul, t_u32, {band(S0(), u(0xffff)), band(S1(), u(0xffff))}), S2()));
        if (n == "v_bfe_u32") return vres_u(i, b.op(spv::OpBitFieldUExtract, t_u32, {S0(), band(S1(), u(31)), band(S2(), u(31))}));
        if (n == "v_bfe_i32") return vres_u(i, tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), band(S1(), u(31)), band(S2(), u(31))})));
        if (n == "v_bfi_b32") { Id m = S0(); return vres_u(i, bor(band(m, S1()), band(bnot(m), S2()))); }
        if (n == "v_bfm_b32") return vres_u(i, shl(isub(shl(u(1), S0()), u(1)), S1()));
        if (n == "v_lshl_add_u32") return vres_u(i, iadd(shl(S0(), S1()), S2()));
        if (n == "v_add_lshl_u32") return vres_u(i, shl(iadd(S0(), S1()), S2()));
        if (n == "v_lshl_or_b32") return vres_u(i, bor(shl(S0(), S1()), S2()));
        if (n == "v_and_or_b32") return vres_u(i, bor(band(S0(), S1()), S2()));
        if (n == "v_or3_b32") return vres_u(i, bor(bor(S0(), S1()), S2()));
        if (n == "v_xor3_b32") return vres_u(i, bxor(bxor(S0(), S1()), S2()));
        if (n == "v_add3_u32") return vres_u(i, iadd(iadd(S0(), S1()), S2()));
        if (n == "v_xad_u32") return vres_u(i, iadd(bxor(S0(), S1()), S2()));
        if (n == "v_alignbit_b32") {
            Id a = b.op(spv::OpUConvert, t_u64, {S0()}), z = b.op(spv::OpUConvert, t_u64, {S1()});
            Id cat = b.op(spv::OpBitwiseOr, t_u64, {b.op(spv::OpShiftLeftLogical, t_u64, {a, b.c_u32(32)}), z});
            return vres_u(i, b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {cat, band(S2(), u(31))})}));
        }
        if (n == "v_bcnt_u32_b32") return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {S0()}), S1()));
        if (n == "v_ffbl_b32") return vres_u(i, tou(ext(GLSLstd450FindILsb, t_i32, {S0()})));
        if (n == "v_ffbh_u32") {
            Id m = ext(GLSLstd450FindUMsb, t_i32, {S0()});
            return vres_u(i, sel(t_u32, b.op(spv::OpSLessThan, t_bool, {m, b.c_i32(0)}), u(~0u), isub(u(31), tou(m))));
        }
        if (n == "v_bfrev_b32") return vres_u(i, b.op(spv::OpBitReverse, t_u32, {S0()}));
        if (n == "v_min_u32") return vres_u(i, ext(GLSLstd450UMin, t_u32, {S0(), S1()}));
        if (n == "v_max_u32") return vres_u(i, ext(GLSLstd450UMax, t_u32, {S0(), S1()}));
        if (n == "v_min_i32") return vres_u(i, tou(ext(GLSLstd450SMin, t_i32, {bc(t_i32, S0()), bc(t_i32, S1())})));
        if (n == "v_max_i32") return vres_u(i, tou(ext(GLSLstd450SMax, t_i32, {bc(t_i32, S0()), bc(t_i32, S1())})));
        if (n == "v_med3_u32") {
            Id a = S0(), z = S1(), c = S2();
            return vres_u(i, ext(GLSLstd450UMax, t_u32, {ext(GLSLstd450UMin, t_u32, {a, z}), ext(GLSLstd450UMin, t_u32, {ext(GLSLstd450UMax, t_u32, {a, z}), c})}));
        }
        if (n == "v_med3_i32") {
            Id a = bc(t_i32, S0()), z = bc(t_i32, S1()), c = bc(t_i32, S2());
            Id t_i = t_i32;
            return vres_u(i, tou(ext(GLSLstd450SMax, t_i, {ext(GLSLstd450SMin, t_i, {a, z}), ext(GLSLstd450SMin, t_i, {ext(GLSLstd450SMax, t_i, {a, z}), c})})));
        }
        if (n == "v_add_co_u32" || n == "v_sub_co_u32" || n == "v_subrev_co_u32") {
            Id a = S0(), z = S1();
            if (n == "v_subrev_co_u32") std::swap(a, z);
            bool add = n == "v_add_co_u32";
            Id r = add ? iadd(a, z) : isub(a, z);
            Id carry = add ? b.op(spv::OpULessThan, t_bool, {r, a}) : b.op(spv::OpULessThan, t_bool, {a, z});
            vres_u(i, r);
            set_mask(i.enc == Enc::VOP3 ? i.sdst.code : VCC_LO, carry);
            return;
        }
        if (n == "v_add_co_ci_u32" || n == "v_sub_co_ci_u32" || n == "v_subrev_co_ci_u32") {
            Id a = S0(), z = S1();
            if (n == "v_subrev_co_ci_u32") std::swap(a, z);
            Id cin_mask = i.enc == Enc::VOP3 ? mask_src(i.src[2]) : mask_reg(VCC_LO);
            Id cin = b2u(mask_bit(cin_mask));
            bool add = n == "v_add_co_ci_u32";
            Id r = add ? iadd(iadd(a, z), cin) : isub(isub(a, z), cin);
            Id carry = add ? b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpULessThan, t_bool, {r, a}),
                                                            b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {r, a}), ne0(cin)})})
                           : b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpULessThan, t_bool, {a, z}),
                                                            b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {a, z}), ne0(cin)})});
            vres_u(i, r);
            set_mask(i.enc == Enc::VOP3 ? i.sdst.code : VCC_LO, carry);
            return;
        }
        // (Wide mode emits these three once: they read or write both halves themselves.)
        auto shuffle = [&](Id x, Id from) { return b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), x, from}); };
        if (n == "v_readfirstlane_b32") {
            // The first lane in EXEC (every invocation runs every block, so the subgroup's
            // first invocation is not the first active lane).
            auto first_of = [&](Id e) { return sel(t_u32, ne0(e), tou(ext(GLSLstd450FindILsb, t_i32, {e})), u(0)); };
            Id e = get_s(EXEC_LO);
            Id x = shuffle(as_half(0, S0), first_of(e));
            if (wide) x = sel(t_u32, ne0(e), x, shuffle(as_half(1, S0), first_of(get_s(EXEC_HI))));
            return set_s(i.dst.vgpr() ? i.dst.vgpr_index() : i.dst.code, x);  // dst is an SGPR (encoded in vdst)
        }
        if (n == "v_readlane_b32") {
            Id at = S1();
            Id x = shuffle(as_half(0, S0), band(at, u(31)));
            if (wide) x = sel(t_u32, ne0(band(at, u(32))), shuffle(as_half(1, S0), band(at, u(31))), x);
            uint16_t d = i.enc == Enc::VOP3 ? uint16_t(i.dst.code - 256) : i.dst.code;
            return set_s(d, x);
        }
        if (n == "v_writelane_b32") {
            Id at = S1(), val = S0();
            for (int hh = 0; hh < (wide ? 2 : 1); ++hh) {
                Id is = b.op(spv::OpIEqual, t_bool, {lane, band(at, u(31))});
                if (wide) is = b.op(spv::OpLogicalAnd, t_bool, {is, b.op(spv::OpIEqual, t_bool, {band(shr(at, u(5)), u(1)), u(uint32_t(hh))})});
                Id r = vreg_of(hh, i.dst.vgpr_index());
                store(r, sel(t_u32, is, val, load(r, t_u32)));
            }
            return;
        }
        if (n == "v_mbcnt_lo_u32_b32") {
            Id below = isub(shl(u(1), lane), u(1));
            if (wide && h == 1) return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {S0()}), S1()));  // every low lane is below
            return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {band(S0(), below)}), S1()));
        }
        if (n == "v_mbcnt_hi_u32_b32") {
            if (wide) {
                if (h == 0) return vres_u(i, S1());
                Id below = isub(shl(u(1), lane), u(1));
                return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {band(S0(), below)}), S1()));
            }
            return vres_u(i, ms_half ? iadd(shl(ms_half, u(5)), S1()) : S1());
        }
        if (n == "v_permlanex16_b32" || n == "v_permlane16_b32") {
            // Lane select from two 32-bit selectors (4 bits per lane of the 16-lane row).
            Id sel_lo = S1(), sel_hi = S2();
            Id idx = band(lane, u(15));
            Id nib = sel(t_u32, b.op(spv::OpULessThan, t_bool, {idx, u(8)}), shr(sel_lo, shl(idx, u(2))), shr(sel_hi, shl(isub(idx, u(8)), u(2))));
            Id src_lane = band(nib, u(15));
            Id row = band(lane, u(16));
            if (n == "v_permlanex16_b32") row = bxor(row, u(16));
            Id from = bor(row, src_lane);
            return vres_u(i, b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), S0(), from}));
        }
        if (n == "v_interp_p1_f32" || n == "v_interp_p2_f32" || n == "v_interp_mov_f32") {
            auto it = ps_inputs.find(i.attr);
            if (it == ps_inputs.end()) return unsupported(i);
            Id val = load(it->second, t_v4f);
            return vres_u(i, tou(b.op(spv::OpCompositeExtract, t_f32, {val, i.attr_chan})));
        }
        if (n == "v_alignbyte_b32") {
            Id a = b.op(spv::OpUConvert, t_u64, {S0()}), z = b.op(spv::OpUConvert, t_u64, {S1()});
            Id cat = b.op(spv::OpBitwiseOr, t_u64, {b.op(spv::OpShiftLeftLogical, t_u64, {a, b.c_u32(32)}), z});
            return vres_u(i, b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {cat, shl(band(S2(), u(3)), u(3))})}));
        }
        if (n == "v_lshlrev_b64" || n == "v_lshrrev_b64" || n == "v_ashrrev_i64") {
            Id x = vraw64(i.src[1]), sh = band(S0(), u(63));
            Id r = n == "v_lshlrev_b64" ? b.op(spv::OpShiftLeftLogical, t_u64, {x, sh})
                 : n == "v_lshrrev_b64" ? b.op(spv::OpShiftRightLogical, t_u64, {x, sh})
                                        : b.op(spv::OpShiftRightArithmetic, t_u64, {x, sh});
            return set_v64(i.dst.vgpr_index(), r);
        }
        if (n == "v_mad_u64_u32") {
            Id m = b.op(spv::OpIMul, t_u64, {b.op(spv::OpUConvert, t_u64, {S0()}), b.op(spv::OpUConvert, t_u64, {S1()})});
            Id c = vraw64(i.src[2]);
            Id r = b.op(spv::OpIAdd, t_u64, {m, c});
            set_v64(i.dst.vgpr_index(), r);
            set_mask(i.sdst.valid ? i.sdst.code : SNULL, b.op(spv::OpULessThan, t_bool, {r, c}));
            return;
        }
        if (n == "v_mad_i32_i24") {
            Id a = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(24)});
            Id z = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S1()), u(0), u(24)});
            return vres_u(i, iadd(tou(b.op(spv::OpIMul, t_i32, {a, z})), S2()));
        }
        if (n == "v_min3_u32") return vres_u(i, ext(GLSLstd450UMin, t_u32, {ext(GLSLstd450UMin, t_u32, {S0(), S1()}), S2()}));
        if (n == "v_max3_u32") return vres_u(i, ext(GLSLstd450UMax, t_u32, {ext(GLSLstd450UMax, t_u32, {S0(), S1()}), S2()}));
        if (n == "v_perm_b32") {
            // Each result byte k picks byte sel_k of {S0, S1} (8 bytes), or a constant (sel >= 12).
            Id a = b.op(spv::OpUConvert, t_u64, {S0()}), z = b.op(spv::OpUConvert, t_u64, {S1()});
            Id cat = b.op(spv::OpBitwiseOr, t_u64, {b.op(spv::OpShiftLeftLogical, t_u64, {a, b.c_u32(32)}), z});
            Id selw = S2(), r = u(0);
            for (uint32_t k = 0; k < 4; ++k) {
                Id sk = band(shr(selw, u(8 * k)), u(0xff));
                Id byte = band(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {cat, shl(band(sk, u(7)), u(3))})}), u(0xff));
                byte = sel(t_u32, b.op(spv::OpUGreaterThanEqual, t_bool, {sk, u(12)}), sel(t_u32, b.op(spv::OpIEqual, t_bool, {sk, u(12)}), u(0), u(0xff)), byte);
                r = bor(r, shl(byte, u(8 * k)));
            }
            return vres_u(i, r);
        }
        if (n == "v_min_f16" || n == "v_max_f16" || n == "v_min3_f16" || n == "v_max3_f16") {
            bool mn = n[2] == 'm' && n[3] == 'i';
            uint32_t op = mn ? GLSLstd450NMin : GLSLstd450NMax;
            Id a = half(S0(), i.opsel & 1), z = half(S1(), i.opsel & 2);
            Id r = ext(op, t_f32, {a, z});
            if (n.find("3_") != std::string_view::npos) r = ext(op, t_f32, {r, half(S2(), i.opsel & 4)});
            Id old = get_v(i.dst.vgpr_index());
            Id h = band(pack_half(r, f(0)), u(0xffff));
            return vres_u(i, (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(h, u(16))) : bor(band(old, u(0xffff0000)), h));
        }
        if (n == "v_pk_min_f16" || n == "v_pk_max_f16" || n == "v_pk_add_f16" || n == "v_pk_mul_f16" || n == "v_pk_fma_f16") {
            Id a = S0(), z = S1();
            auto lanes = [&](int k) {
                bool hi0 = k ? (i.opsel_hi & 1) : (i.opsel & 1), hi1 = k ? (i.opsel_hi & 2) : (i.opsel & 2);
                Id x = half(a, hi0), y = half(z, hi1);
                if (n == "v_pk_min_f16") return ext(GLSLstd450NMin, t_f32, {x, y});
                if (n == "v_pk_max_f16") return ext(GLSLstd450NMax, t_f32, {x, y});
                if (n == "v_pk_add_f16") return b.op(spv::OpFAdd, t_f32, {x, y});
                if (n == "v_pk_mul_f16") return b.op(spv::OpFMul, t_f32, {x, y});
                bool hi2 = k ? (i.opsel_hi & 4) : (i.opsel & 4);
                return ext(GLSLstd450Fma, t_f32, {x, y, half(S2(), hi2)});
            };
            return vres_u(i, pack_half(lanes(0), lanes(1)));
        }
        // ---- mixed precision: each source f32, or the low or high f16 of a dword (op_sel_hi says
        // f16, op_sel which half); neg_lo negates, neg_hi takes the absolute value ----
        if (n == "v_fma_mix_f32" || n == "v_fma_mixlo_f16" || n == "v_fma_mixhi_f16" || n == "v_mad_mix_f32" || n == "v_mad_mixlo_f16" || n == "v_mad_mixhi_f16") {
            static const float fl[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.15915494f};
            auto mix = [&](int k) {
                const Operand& o = i.src[k];
                Id x;
                if ((i.opsel_hi >> k) & 1) {
                    if (o.code >= 240 && o.code <= 248) x = f(fl[o.code - 240]);  // inline constants are the value itself
                    else x = half(ov[k] ? ov[k] : raw(o), (i.opsel >> k) & 1);
                } else {
                    x = tof(ov[k] ? ov[k] : raw(o));
                }
                if ((i.neg_hi >> k) & 1) x = ext(GLSLstd450FAbs, t_f32, {x});
                if (o.neg) x = b.op(spv::OpFNegate, t_f32, {x});
                return x;
            };
            Id r = ext(GLSLstd450Fma, t_f32, {mix(0), mix(1), mix(2)});
            if (i.clamp) r = ext(GLSLstd450FClamp, t_f32, {r, f(0), f(1)});
            if (n.find("mix_f32") != std::string_view::npos) return vres_u(i, tou(r));
            Id old = get_v(i.dst.vgpr_index());
            Id hv = band(pack_half(r, f(0)), u(0xffff));
            bool hi = n.find("mixhi") != std::string_view::npos;
            return vres_u(i, hi ? bor(band(old, u(0xffff)), shl(hv, u(16))) : bor(band(old, u(0xffff0000)), hv));
        }
        // ---- 16-bit float arithmetic: low (or op_sel) halves in, the low (or op_sel[3]) half out,
        // the other half kept ----
        if (n == "v_add_f16" || n == "v_sub_f16" || n == "v_subrev_f16" || n == "v_mul_f16" || n == "v_fma_f16" || n == "v_fmac_f16" ||
            n == "v_mad_f16" || n == "v_mac_f16" || n == "v_rcp_f16" || n == "v_sqrt_f16" || n == "v_rsq_f16" || n == "v_ldexp_f16" ||
            n == "v_floor_f16" || n == "v_ceil_f16" || n == "v_trunc_f16" || n == "v_fract_f16" || n == "v_rndne_f16" || n == "v_exp_f16" ||
            n == "v_log_f16" || n == "v_sin_f16" || n == "v_cos_f16") {
            auto hs = [&](int k) {
                Id x = half(ov[k] ? ov[k] : raw(i.src[k]), (i.opsel >> k) & 1);
                if (i.src[k].abs) x = ext(GLSLstd450FAbs, t_f32, {x});
                if (i.src[k].neg) x = b.op(spv::OpFNegate, t_f32, {x});
                return x;
            };
            Id old = get_v(i.dst.vgpr_index());
            bool dst_hi = (i.opsel & 8) != 0;
            Id r;
            if (n == "v_add_f16") r = b.op(spv::OpFAdd, t_f32, {hs(0), hs(1)});
            else if (n == "v_sub_f16") r = b.op(spv::OpFSub, t_f32, {hs(0), hs(1)});
            else if (n == "v_subrev_f16") r = b.op(spv::OpFSub, t_f32, {hs(1), hs(0)});
            else if (n == "v_mul_f16") r = b.op(spv::OpFMul, t_f32, {hs(0), hs(1)});
            else if (n == "v_fma_f16" || n == "v_mad_f16") r = ext(GLSLstd450Fma, t_f32, {hs(0), hs(1), hs(2)});
            else if (n == "v_fmac_f16" || n == "v_mac_f16") r = ext(GLSLstd450Fma, t_f32, {hs(0), hs(1), half(old, dst_hi)});
            else if (n == "v_rcp_f16") r = b.op(spv::OpFDiv, t_f32, {f(1), hs(0)});
            else if (n == "v_sqrt_f16") r = ext(GLSLstd450Sqrt, t_f32, {hs(0)});
            else if (n == "v_rsq_f16") r = ext(GLSLstd450InverseSqrt, t_f32, {hs(0)});
            else if (n == "v_ldexp_f16") r = ext(GLSLstd450Ldexp, t_f32, {hs(0), bc(t_i32, b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S1()), u(0), u(16)}))});
            else if (n == "v_floor_f16") r = ext(GLSLstd450Floor, t_f32, {hs(0)});
            else if (n == "v_ceil_f16") r = ext(GLSLstd450Ceil, t_f32, {hs(0)});
            else if (n == "v_trunc_f16") r = ext(GLSLstd450Trunc, t_f32, {hs(0)});
            else if (n == "v_fract_f16") r = ext(GLSLstd450Fract, t_f32, {hs(0)});
            else if (n == "v_rndne_f16") r = ext(GLSLstd450RoundEven, t_f32, {hs(0)});
            else if (n == "v_exp_f16") r = ext(GLSLstd450Exp2, t_f32, {hs(0)});
            else if (n == "v_log_f16") r = ext(GLSLstd450Log2, t_f32, {hs(0)});
            else if (n == "v_sin_f16") r = ext(GLSLstd450Sin, t_f32, {b.op(spv::OpFMul, t_f32, {hs(0), f(6.2831853f)})});
            else r = ext(GLSLstd450Cos, t_f32, {b.op(spv::OpFMul, t_f32, {hs(0), f(6.2831853f)})});
            if (i.clamp) r = ext(GLSLstd450FClamp, t_f32, {r, f(0), f(1)});
            Id hv = band(pack_half(r, f(0)), u(0xffff));
            return vres_u(i, dst_hi ? bor(band(old, u(0xffff)), shl(hv, u(16))) : bor(band(old, u(0xffff0000)), hv));
        }
        if (n == "v_lshrrev_b16" || n == "v_lshlrev_b16" || n == "v_ashrrev_i16" || n == "v_add_u16" || n == "v_sub_u16" || n == "v_mul_lo_u16") {
            Id a = band(S0(), u(0xffff)), z = band(S1(), u(0xffff)), r;
            if (n == "v_lshrrev_b16") r = shr(z, band(a, u(15)));
            else if (n == "v_lshlrev_b16") r = shl(z, band(a, u(15)));
            else if (n == "v_ashrrev_i16") r = tou(b.op(spv::OpShiftRightArithmetic, t_i32, {b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, z), u(0), u(16)}), band(a, u(15))}));
            else if (n == "v_add_u16") r = iadd(a, z);
            else if (n == "v_sub_u16") r = isub(a, z);
            else r = b.op(spv::OpIMul, t_u32, {a, z});
            Id old = get_v(i.dst.vgpr_index());
            return vres_u(i, (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(r, u(16))) : bor(band(old, u(0xffff0000)), band(r, u(0xffff))));
        }
        // ---- cube map coordinates: face id, s and t on the face, twice the major axis ----
        if (n == "v_cubeid_f32" || n == "v_cubesc_f32" || n == "v_cubetc_f32" || n == "v_cubema_f32") {
            Id x = F0(), y = F1(), z = F2();
            auto fabs_ = [&](Id a) { return ext(GLSLstd450FAbs, t_f32, {a}); };
            auto neg = [&](Id a) { return b.op(spv::OpFNegate, t_f32, {a}); };
            auto lt0 = [&](Id a) { return b.op(spv::OpFOrdLessThan, t_bool, {a, f(0)}); };
            auto ge = [&](Id a, Id c) { return b.op(spv::OpFOrdGreaterThanEqual, t_bool, {a, c}); };
            Id ax = fabs_(x), ay = fabs_(y), az = fabs_(z);
            Id zmaj = b.op(spv::OpLogicalAnd, t_bool, {ge(az, ax), ge(az, ay)});
            Id ymaj = b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpLogicalNot, t_bool, {zmaj}), ge(ay, ax)});
            auto pick = [&](Id zv, Id yv, Id xv) { return sel(t_f32, zmaj, zv, sel(t_f32, ymaj, yv, xv)); };
            Id r;
            if (n == "v_cubeid_f32")
                r = pick(sel(t_f32, lt0(z), f(5), f(4)), sel(t_f32, lt0(y), f(3), f(2)), sel(t_f32, lt0(x), f(1), f(0)));
            else if (n == "v_cubesc_f32")
                r = pick(sel(t_f32, lt0(z), neg(x), x), x, sel(t_f32, lt0(x), z, neg(z)));
            else if (n == "v_cubetc_f32")
                r = pick(neg(y), sel(t_f32, lt0(y), neg(z), z), neg(y));
            else
                r = b.op(spv::OpFMul, t_f32, {f(2), pick(z, y, x)});
            return vres_f(i, r);
        }
        // ---- VGPR indexed by M0 ----
        if (n == "v_movrels_b32" || n == "v_movreld_b32" || n == "v_movrelsd_b32") {
            Id m0 = get_s(M0);
            uint32_t base_s = i.src[0].vgpr() ? i.src[0].vgpr_index() : 0, base_d = i.dst.vgpr_index();
            uint32_t limit = std::max<uint32_t>(p.vgprs, 1);
            if (n == "v_movreld_b32") {  // VGPR[dst + M0] = src0
                Id val = S0(), act = active();
                for (uint32_t k = 0; base_d + k < limit && base_d + k < VGPRS; ++k) {
                    Id r = vreg(uint16_t(base_d + k));
                    Id hit = b.op(spv::OpLogicalAnd, t_bool, {act, b.op(spv::OpIEqual, t_bool, {m0, u(k)})});
                    store(r, sel(t_u32, hit, val, load(r, t_u32)));
                }
                return;
            }
            Id x = u(0);  // VGPR[src0 + M0]
            for (uint32_t k = 0; base_s + k < limit && base_s + k < VGPRS; ++k)
                x = sel(t_u32, b.op(spv::OpIEqual, t_bool, {m0, u(k)}), get_v(uint16_t(base_s + k)), x);
            if (n == "v_movrels_b32") return vres_u(i, x);
            Id act = active();  // movrelsd: VGPR[dst + M0] = VGPR[src0 + M0]
            for (uint32_t k = 0; base_d + k < limit && base_d + k < VGPRS; ++k) {
                Id r = vreg(uint16_t(base_d + k));
                Id hit = b.op(spv::OpLogicalAnd, t_bool, {act, b.op(spv::OpIEqual, t_bool, {m0, u(k)})});
                store(r, sel(t_u32, hit, x, load(r, t_u32)));
            }
            return;
        }
        if (n == "v_swap_b32") {
            uint16_t a = i.dst.vgpr_index(), z = i.src[0].vgpr_index();
            Id va = get_v(a), vz = get_v(z), act = active();
            store(vreg(a), sel(t_u32, act, vz, va));
            store(vreg(z), sel(t_u32, act, va, vz));
            return;
        }
        if (n == "v_cvt_pk_u8_f32") {
            Id byte = b.op(spv::OpConvertFToU, t_u32, {ext(GLSLstd450FClamp, t_f32, {ext(GLSLstd450RoundEven, t_f32, {F0()}), f(0), f(255)})});
            Id sh = shl(band(S1(), u(3)), u(3));
            return vres_u(i, bor(band(S2(), bnot(shl(u(0xff), sh))), shl(byte, sh)));
        }
        // ---- 16-bit interpolation (attribute in src0: bits 0-5, channel in bits 6-7). The host
        // interpolates the attribute row as floats; p1ll is the first half of the pair. ----
        if (n == "v_interp_p1ll_f16" || n == "v_interp_p1lv_f16") return vres_u(i, u(0));
        if (n == "v_interp_p2_f16") {
            uint32_t attr = i.src[0].code & 63, chan = (i.src[0].code >> 6) & 3;
            auto it = ps_inputs.find(uint8_t(attr));
            if (it == ps_inputs.end()) return unsupported(i);
            // A compressed export (f16 pairs) arrives here unpacked into four floats: channel c's
            // low and high halves are the row's components 2c and 2c + 1.
            uint32_t comp = std::min<uint32_t>(2 * chan + (i.opsel & 1), 3);
            Id val = b.op(spv::OpCompositeExtract, t_f32, {load(it->second, t_v4f), comp});
            Id old = get_v(i.dst.vgpr_index());
            Id hv = band(pack_half(val, f(0)), u(0xffff));
            return vres_u(i, (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(hv, u(16))) : bor(band(old, u(0xffff0000)), hv));
        }
        if (n == "v_cmp_class_f32") return unsupported(i);
        return unsupported(i);
    }

    // Vector memory.
