// keylime: types, interface variables and the entry prologue (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    // ---------------------------------------------------------- setup ----
    void types() {
        glsl = b.import_glsl();
        t_void = b.t_void(), t_bool = b.t_bool(), t_u32 = b.t_u32(), t_i32 = b.t_i32(), t_f32 = b.t_f32(), t_u64 = b.t_u64();
        t_v4f = b.t_vec(t_f32, 4), t_v4u = b.t_vec(t_u32, 4), t_v2f = b.t_vec(t_f32, 2), t_v4b = b.t_vec(t_bool, 4);
        t_fn = b.t_fn(t_void, {});
        p_fn_u32 = b.t_ptr(spv::StorageClassFunction, t_u32);
        p_fn_bool = b.t_ptr(spv::StorageClassFunction, t_bool);
        p_psb_u32 = b.t_ptr(spv::StorageClassPhysicalStorageBuffer, t_u32);
        p_psb_u64 = b.t_ptr(spv::StorageClassPhysicalStorageBuffer, t_u64);
        p_pc_u32 = b.t_ptr(spv::StorageClassPushConstant, t_u32);
        p_pc_u64 = b.t_ptr(spv::StorageClassPushConstant, t_u64);
    }

    Id input(Id type, spv::BuiltIn bi) {
        Id var = b.variable(b.t_ptr(spv::StorageClassInput, type), spv::StorageClassInput);
        b.decorate(var, spv::DecorationBuiltIn, {uint32_t(bi)});
        interface.push_back(var);
        return var;
    }

    void globals() {
        b.capability(spv::CapabilityShader);
        b.capability(spv::CapabilityPhysicalStorageBufferAddresses);
        b.capability(spv::CapabilityGroupNonUniform);
        b.capability(spv::CapabilityGroupNonUniformBallot);
        b.capability(spv::CapabilityGroupNonUniformShuffle);
        b.capability(spv::CapabilityGroupNonUniformVote);
        b.capability(spv::CapabilityGroupNonUniformArithmetic);
        b.capability(spv::CapabilityGroupNonUniformQuad);
        b.capability(spv::CapabilityRuntimeDescriptorArray);
        b.capability(spv::CapabilitySampledImageArrayNonUniformIndexing);
        b.capability(spv::CapabilityStorageImageReadWithoutFormat);
        b.capability(spv::CapabilityStorageImageWriteWithoutFormat);
        b.capability(spv::CapabilityImageQuery);
        types();
        // Push constants: {u64 draw data, u64 page table, u64 texture table, u32 misc[2]}.
        Id a2 = b.t_array_fresh(t_u32, 2);
        b.decorate(a2, spv::DecorationArrayStride, {4});
        Id st = b.t_struct({t_u64, t_u64, t_u64, a2});
        b.decorate(st, spv::DecorationBlock);
        b.member_decorate(st, 0, spv::DecorationOffset, {0});
        b.member_decorate(st, 1, spv::DecorationOffset, {8});
        b.member_decorate(st, 2, spv::DecorationOffset, {16});
        b.member_decorate(st, 3, spv::DecorationOffset, {24});
        push = b.variable(b.t_ptr(spv::StorageClassPushConstant, st), spv::StorageClassPushConstant);
        interface.push_back(push);
        // Bindless resources.
        t_image2d = b.t_image(t_f32, spv::Dim2D, false, false, false, 1, spv::ImageFormatUnknown);
        t_storage2d = b.t_image(t_f32, spv::Dim2D, false, false, false, 2, spv::ImageFormatUnknown);
        Id t_samp = b.t_sampler();
        t_simage2d = b.t_sampled_image(t_image2d);
        Id ra_img = b.t_runtime_array(t_image2d), ra_samp = b.t_runtime_array(t_samp), ra_st = b.t_runtime_array(t_storage2d);
        p_uc_image = b.t_ptr(spv::StorageClassUniformConstant, t_image2d);
        p_uc_sampler = b.t_ptr(spv::StorageClassUniformConstant, t_samp);
        sampled_images = b.variable(b.t_ptr(spv::StorageClassUniformConstant, ra_img), spv::StorageClassUniformConstant);
        samplers = b.variable(b.t_ptr(spv::StorageClassUniformConstant, ra_samp), spv::StorageClassUniformConstant);
        storage_images = b.variable(b.t_ptr(spv::StorageClassUniformConstant, ra_st), spv::StorageClassUniformConstant);
        Id vars[3] = {sampled_images, samplers, storage_images};
        for (uint32_t i = 0; i < 3; ++i) {
            b.decorate(vars[i], spv::DecorationDescriptorSet, {0});
            b.decorate(vars[i], spv::DecorationBinding, {i});
            interface.push_back(vars[i]);
        }
        lane_var = input(t_u32, spv::BuiltInSubgroupLocalInvocationId);
        if (p.stage == HwStage::Ps) b.decorate(lane_var, spv::DecorationFlat);
        if (p.stage == HwStage::Vs) {
            vertex_index = input(t_i32, spv::BuiltInVertexIndex);
            instance_index = input(t_i32, spv::BuiltInInstanceIndex);
            position_out = b.variable(b.t_ptr(spv::StorageClassOutput, t_v4f), spv::StorageClassOutput);
            b.decorate(position_out, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInPosition)});
            interface.push_back(position_out);
            for (auto& in : p.vs_inputs) {
                if (vs_inputs.count(in.row)) continue;
                Id var = b.variable(b.t_ptr(spv::StorageClassInput, t_v4f), spv::StorageClassInput);
                b.decorate(var, spv::DecorationLocation, {in.row});
                vs_inputs[in.row] = var;
                interface.push_back(var);
                out.vs_input_rows.push_back(in.row);
            }
            for (auto& o : p.vs_outputs) {
                if (vs_outputs.count(o.param)) continue;
                Id var = b.variable(b.t_ptr(spv::StorageClassOutput, t_v4f), spv::StorageClassOutput);
                b.decorate(var, spv::DecorationLocation, {o.row});
                vs_outputs[o.param] = var;
                interface.push_back(var);
                out.vs_output_rows.push_back(o.row);
            }
        } else if (p.stage == HwStage::Ps) {
            frag_coord = input(t_v4f, spv::BuiltInFragCoord);
            front_face = input(t_bool, spv::BuiltInFrontFacing);
            for (auto& in : p.ps_inputs) {
                if (ps_inputs.count(in.attr)) continue;
                Id var = b.variable(b.t_ptr(spv::StorageClassInput, t_v4f), spv::StorageClassInput);
                b.decorate(var, spv::DecorationLocation, {ps_location(in.row)});
                if (in.flat) b.decorate(var, spv::DecorationFlat);
                ps_inputs[in.attr] = var;
                interface.push_back(var);
                out.ps_input_rows.push_back(in.row);
            }
        } else if (p.stage == HwStage::Ms) {
            mesh_globals();
        } else if (p.stage == HwStage::Cs) {
            b.capability(spv::CapabilityGroupNonUniform);
            local_id = input(b.t_vec(t_u32, 3), spv::BuiltInLocalInvocationId);
            group_id = input(b.t_vec(t_u32, 3), spv::BuiltInWorkgroupId);
            if (wide) sg_id = input(t_u32, spv::BuiltInSubgroupId);
            if (p.lds_bytes) {
                Id arr = b.t_array(t_u32, (p.lds_bytes + 3) / 4);
                lds = b.variable(b.t_ptr(spv::StorageClassWorkgroup, arr), spv::StorageClassWorkgroup);
                lds_ptr = b.t_ptr(spv::StorageClassWorkgroup, t_u32);
                interface.push_back(lds);
            }
        }
    }

    uint32_t ms_vertices() const { return std::max<uint32_t>(p.ms_max_vertices, 1); }
    uint32_t ms_primitives() const { return std::max<uint32_t>(p.ms_max_primitives, 1); }
    bool ms_lines() const { return p.ms_topology == 1; }

    Id output_array(Id type, uint32_t n) {
        Id var = b.variable(b.t_ptr(spv::StorageClassOutput, b.t_array(type, n)), spv::StorageClassOutput);
        interface.push_back(var);
        return var;
    }

    void mesh_globals() {
        b.extension("SPV_EXT_mesh_shader");
        b.capability(spv::CapabilityMeshShadingEXT);
        local_index = input(t_u32, spv::BuiltInLocalInvocationIndex);
        b.capability(spv::CapabilityGroupNonUniform);
        if (wide) sg_id = input(t_u32, spv::BuiltInSubgroupId);
        group_id = input(b.t_vec(t_u32, 3), spv::BuiltInWorkgroupId);
        num_groups = input(b.t_vec(t_u32, 3), spv::BuiltInNumWorkgroups);
        if (p.lds_bytes) {
            Id arr = b.t_array(t_u32, (p.lds_bytes + 3) / 4);
            lds = b.variable(b.t_ptr(spv::StorageClassWorkgroup, arr), spv::StorageClassWorkgroup);
            lds_ptr = b.t_ptr(spv::StorageClassWorkgroup, t_u32);
            interface.push_back(lds);
        }
        ms_count = b.variable(b.t_ptr(spv::StorageClassWorkgroup, t_u32), spv::StorageClassWorkgroup);
        interface.push_back(ms_count);
        ms_pos_out = output_array(t_v4f, ms_vertices());
        b.decorate(ms_pos_out, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInPosition)});
        ms_prim_out = output_array(b.t_vec(t_u32, ms_lines() ? 2 : 3), ms_primitives());
        b.decorate(ms_prim_out, spv::DecorationBuiltIn,
                   {uint32_t(ms_lines() ? spv::BuiltInPrimitiveLineIndicesEXT : spv::BuiltInPrimitiveTriangleIndicesEXT)});
        Id per_prim = b.t_struct({t_bool});
        b.decorate(per_prim, spv::DecorationBlock);
        b.member_decorate(per_prim, 0, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInCullPrimitiveEXT)});
        b.member_decorate(per_prim, 0, spv::DecorationPerPrimitiveEXT);
        ms_cull_out = output_array(per_prim, ms_primitives());
        for (auto& o : p.vs_outputs) {
            if ((o.param & 0x80) || ms_outs.count(o.param)) continue;
            Id var = output_array(t_v4f, ms_vertices());
            b.decorate(var, spv::DecorationLocation, {o.row});
            ms_outs[o.param] = var;
            out.vs_output_rows.push_back(o.row);
        }
    }

    // A console thread's index in its group: wide mode runs two per invocation.
    Id ms_thread(int half) {
        if (!wide) return load(local_index, t_u32);
        return iadd(iadd(b.op(spv::OpIMul, t_u32, {load(sg_id, t_u32), u(64)}), u(uint32_t(32 * half))), lane);
    }
    void mesh_prologue() {
        p_fn_v4f = b.t_ptr(spv::StorageClassFunction, t_v4f);
        Id zero4 = b.op(spv::OpCompositeConstruct, t_v4f, {f(0), f(0), f(0), f(1)});
        for (int hh = 0; hh < (wide ? 2 : 1); ++hh) {
            ms_pos_[hh] = b.local(p_fn_v4f);
            store(ms_pos_[hh], zero4);
            ms_prim_[hh] = b.local(p_fn_u32, u(0));
            ms_flags_[hh] = b.local(p_fn_u32, u(0));
            for (auto& [param, var] : ms_outs) {
                ms_vals_[hh][param] = b.local(p_fn_v4f);
                store(ms_vals_[hh][param], zero4);
            }
        }
        Id idx = load(local_index, t_u32);
        if (wide) {
            set_s(3, shl(load(sg_id, t_u32), u(24)));  // the console wave's number
            ms_half = 0;
            for (int hh = 0; hh < 2; ++hh)
                set_s(uint16_t(EXEC_LO + hh), ballot(b.op(spv::OpULessThan, t_bool, {ms_thread(hh), u(p.ms_threads)})));
        } else {
            Id sg = shr(idx, u(5));
            set_s(3, shl(p.wave64 ? shr(sg, u(1)) : sg, u(24)));
            ms_half = p.wave64 ? band(sg, u(1)) : u(0);
        }
        Id gid = load(group_id, b.t_vec(t_u32, 3)), ng = load(num_groups, b.t_vec(t_u32, 3));
        auto c = [&](Id v, uint32_t k) { return b.op(spv::OpCompositeExtract, t_u32, {v, k}); };
        Id nx = c(ng, 0), nxy = b.op(spv::OpIMul, t_u32, {nx, c(ng, 1)});
        Id linear = iadd(iadd(c(gid, 0), b.op(spv::OpIMul, t_u32, {c(gid, 1), nx})), b.op(spv::OpIMul, t_u32, {c(gid, 2), nxy}));
        Id base5 = b.op(spv::OpIMul, t_u32, {linear, u(p.ms_threads)});
        for (int hh = 0; hh < (wide ? 2 : 1); ++hh) store(vreg_of(hh, 5), base5);
        if_then(b.op(spv::OpIEqual, t_bool, {idx, u(0)}), [&] { store(ms_count, u(0)); });
        barrier();
    }

    void barrier() {
        b.op_void(spv::OpControlBarrier, {u(spv::ScopeWorkgroup), u(spv::ScopeWorkgroup),
                                          u(spv::MemorySemanticsWorkgroupMemoryMask | spv::MemorySemanticsAcquireReleaseMask)});
    }

    void mesh_export(uint8_t t, Id vec, Id raw) {
        Id act = active();
        Id ms_pos = ms_pos_[h], ms_prim = ms_prim_[h], ms_flags = ms_flags_[h];
        auto& ms_vals = ms_vals_[h];
        if (t == 12) {
            store(ms_pos, sel(t_v4f, act, vec, load(ms_pos, t_v4f)));
            store(ms_flags, bor(load(ms_flags, t_u32), sel(t_u32, act, u(1), u(0))));
        } else if (t == 20) {
            store(ms_prim, sel(t_u32, act, raw, load(ms_prim, t_u32)));
            store(ms_flags, bor(load(ms_flags, t_u32), sel(t_u32, act, u(2), u(0))));
        } else if (t >= 32 && t < 64) {
            auto it = ms_vals.find(uint8_t(t - 32));
            if (it != ms_vals.end()) store(it->second, sel(t_v4f, act, vec, load(it->second, t_v4f)));
        }
    }

    void mesh_epilogue() {
        barrier();
        Id c = load(ms_count, t_u32);
        Id nv = ext(GLSLstd450UMin, t_u32, {band(c, u(0xfff)), u(ms_vertices())});
        Id np = ext(GLSLstd450UMin, t_u32, {band(shr(c, u(12)), u(0xfff)), u(ms_primitives())});
        b.op_void(spv::OpSetMeshOutputsEXT, {nv, np});
        Id p_out_v4f = b.t_ptr(spv::StorageClassOutput, t_v4f);
        for (int hh = 0; hh < (wide ? 2 : 1); ++hh) {
            Id idx = ms_thread(hh), flags = load(ms_flags_[hh], t_u32);
            Id has_v = b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpULessThan, t_bool, {idx, nv}), ne0(band(flags, u(1)))});
            if_then(has_v, [&] {
                store(b.op(spv::OpAccessChain, p_out_v4f, {ms_pos_out, idx}), load(ms_pos_[hh], t_v4f));
                for (auto& [param, var] : ms_outs) store(b.op(spv::OpAccessChain, p_out_v4f, {var, idx}), load(ms_vals_[hh][param], t_v4f));
            });
            if_then(b.op(spv::OpULessThan, t_bool, {idx, np}), [&] {
                Id prim = load(ms_prim_[hh], t_u32);
                Id cull = b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpIEqual, t_bool, {band(flags, u(2)), u(0)}), ne0(shr(prim, u(31)))});
                store(b.op(spv::OpAccessChain, b.t_ptr(spv::StorageClassOutput, t_bool), {ms_cull_out, idx, u(0)}), cull);
                Id last = u(ms_vertices() - 1);
                auto vtx = [&](uint32_t k) { return ext(GLSLstd450UMin, t_u32, {band(shr(prim, u(10 * k)), u(0x3ff)), last}); };
                Id t3 = ms_lines() ? b.t_vec(t_u32, 2) : b.t_vec(t_u32, 3);
                Id v = ms_lines() ? b.op(spv::OpCompositeConstruct, t3, {vtx(0), vtx(1)}) : b.op(spv::OpCompositeConstruct, t3, {vtx(0), vtx(1), vtx(2)});
                store(b.op(spv::OpAccessChain, b.t_ptr(spv::StorageClassOutput, t3), {ms_prim_out, idx}), v);
            });
        }
    }

    Id ps_output(uint8_t mrt) {
        auto it = ps_outputs.find(mrt);
        if (it != ps_outputs.end()) return it->second;
        Id var = b.variable(b.t_ptr(spv::StorageClassOutput, t_v4f), spv::StorageClassOutput);
        b.decorate(var, spv::DecorationLocation, {mrt});
        interface.push_back(var);
        out.mrt_mask |= uint8_t(1u << mrt);
        return ps_outputs[mrt] = var;
    }

    // Initial register state: user SGPRs from push constants, system values.
    void prologue() {
        memory_bases();
        lane = load(lane_var, t_u32);
        lane_bit = shl(u(1), lane);
        // User SGPRs from this draw's record (push constant 0 points at it): only those root
        // parameters arrive in, the host leaves the rest zero (as the registers start).
        {
            uint32_t want = 0;
            for (auto& rp : p.layout)
                for (uint32_t k = 0; k < std::max<uint32_t>(rp.sgprs, 1) && rp.user_data + k < 32; ++k) want |= 1u << (rp.user_data + k);
            Id rec = load(b.op(spv::OpAccessChain, p_pc_u64, {push, u(0)}), t_u64);
            uint32_t base = opt ? opt->sgpr_offset : 0, first = p.stage == HwStage::Ms ? 8 : 0;
            for (uint32_t i = 0; i < 32; ++i) {
                if (!(want & (1u << i))) continue;
                Id a = b.op(spv::OpIAdd, t_u64, {rec, b.c_u64(4ull * (base + i))});
                set_s(uint16_t(first + i), b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {a}), uint32_t(spv::MemoryAccessAlignedMask), 4}));
            }
        }
        // EXEC = every lane that runs (helpers included).
        set_s(EXEC_LO, mine_if(ballot(b.c_bool(true)), !wide && !no_lane_facts));
        set_s(EXEC_HI, u(0));
        if (p.stage == HwStage::Ms) mesh_prologue();
        if (p.stage == HwStage::Vs) {
            set_v(0, bc(t_u32, load(vertex_index, t_i32)));
            set_v(3, bc(t_u32, load(instance_index, t_i32)));
            for (auto& in : p.vs_inputs) {
                Id val = load(vs_inputs[in.row], t_v4f);
                for (uint32_t k = 0; k < 4; ++k)
                    if (in.mask & (1u << k)) set_v(uint16_t(in.vgpr + k), tou(b.op(spv::OpCompositeExtract, t_f32, {val, k})));
            }
        } else if (p.stage == HwStage::Ps) {
            // System VGPRs in SPI_PS_INPUT_ADDR order (barycentrics, then position...).
            static const uint8_t counts[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            uint16_t at = 0;
            Id fc = load(frag_coord, t_v4f);
            for (int bit = 0; bit < 16; ++bit) {
                if (!(p.ps_input_addr & (1u << bit))) continue;
                for (uint8_t k = 0; k < counts[bit]; ++k, ++at) {
                    Id val = u(0);
                    if (bit >= 8 && bit <= 11) val = tou(b.op(spv::OpCompositeExtract, t_f32, {fc, uint32_t(bit - 8)}));
                    else if (bit == 12) val = b2u(load(front_face, t_bool));
                    store(vreg(at), val);
                }
            }
        } else if (p.stage == HwStage::Cs) {
            uint32_t tid_comps = ((p.cs_rsrc2 >> 11) & 3) + 1;  // TIDIG_COMP_CNT
            if (wide) {
                // Console wave w = host subgroup w; its lanes `lane` and `lane + 32` here. The
                // thread's index in the console group gives its x/y/z; lanes past the group's
                // size are not in EXEC.
                uint32_t X = std::max(p.cs_threads[0], 1u), Y = std::max(p.cs_threads[1], 1u), Z = std::max(p.cs_threads[2], 1u);
                Id wave = load(sg_id, t_u32);
                for (int hh = 0; hh < 2; ++hh) {
                    Id li = iadd(iadd(b.op(spv::OpIMul, t_u32, {wave, u(64)}), u(uint32_t(32 * hh))), lane);
                    Id xyz[3] = {b.op(spv::OpUMod, t_u32, {li, u(X)}), b.op(spv::OpUMod, t_u32, {b.op(spv::OpUDiv, t_u32, {li, u(X)}), u(Y)}),
                                 b.op(spv::OpUDiv, t_u32, {li, u(X * Y)})};
                    for (uint32_t k = 0; k < tid_comps && k < 3; ++k) store(vreg_of(hh, uint16_t(k)), xyz[k]);
                    set_s(uint16_t(EXEC_LO + hh), ballot(b.op(spv::OpULessThan, t_bool, {li, u(X * Y * Z)})));
                }
            } else {
                Id lid = load(local_id, b.t_vec(t_u32, 3));
                for (uint32_t k = 0; k < tid_comps && k < 3; ++k) store(vreg(uint16_t(k)), b.op(spv::OpCompositeExtract, t_u32, {lid, k}));
            }
            uint32_t user = (p.cs_rsrc2 >> 1) & 0x1f;
            uint16_t at = uint16_t(user);
            Id gid = load(group_id, b.t_vec(t_u32, 3));
            for (uint32_t k = 0; k < 3; ++k)
                if (p.cs_rsrc2 & (1u << (7 + k))) set_s(at++, b.op(spv::OpCompositeExtract, t_u32, {gid, k}));
            if (p.cs_rsrc2 & (1u << 10)) set_s(at++, u(0));  // TG_SIZE info: wave 0 of 1
        }
    }

