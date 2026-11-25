// keylime: exports (vertex and pixel outputs) (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    void export_(const Inst& i) {
        uint8_t t = i.exp_target;
        auto value = [&](int k) -> Id {
            if (!(i.exp_en & (1u << k))) return f(k == 3 ? 1.0f : 0.0f);
            return tof(get_v(i.exp_src[k]));
        };
        Id comps[4];
        if (i.exp_compr) {  // two f16 per VGPR: src0 = xy, src1 = zw
            for (int k = 0; k < 2; ++k) {
                Id v2 = ext(GLSLstd450UnpackHalf2x16, t_v2f, {get_v(i.exp_src[k])});
                comps[2 * k] = b.op(spv::OpCompositeExtract, t_f32, {v2, 0});
                comps[2 * k + 1] = b.op(spv::OpCompositeExtract, t_f32, {v2, 1});
            }
        } else {
            for (int k = 0; k < 4; ++k) comps[k] = value(k);
        }
        Id vec = b.op(spv::OpCompositeConstruct, t_v4f, {comps[0], comps[1], comps[2], comps[3]});
        if (p.stage == HwStage::Ms) return mesh_export(t, vec, get_v(i.exp_src[0]));
        if (p.stage == HwStage::Vs) {
            if (t == 12) store(position_out, vec);
            else if (t >= 32 && t < 64) {
                auto it = vs_outputs.find(uint8_t(t - 32));
                if (it != vs_outputs.end()) store(it->second, vec);
            }
        } else if (p.stage == HwStage::Ps) {
            if (t <= 7) store(ps_output(t), vec);
            else if (t == 8) out.writes_depth = true;  // mrtz: depth export (later)
            if (i.exp_done) {
                // Lanes not in EXEC at the final export are discarded.
                Id kill = b.id(), keep = b.id();
                Id act = active();
                b.op_void(spv::OpSelectionMerge, {keep, 0});
                b.op_void(spv::OpBranchConditional, {act, keep, kill});
                b.label(kill);
                b.op_void(spv::OpKill, {});
                b.label(keep);
                out.discards = true;
            }
        }
    }

