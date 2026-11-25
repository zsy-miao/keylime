// keylime: image sample, load and store instructions (part of the RDNA2 to SPIR-V translator, inside its class; included by translate.cpp).
    void image(const Inst& i) {
        std::string_view n = i.name;
        Id tkey_lo = get_s(i.srsrc), tkey_hi = get_s(uint16_t(i.srsrc + 1));
        Id tslot = slot_of(tkey_lo, band(tkey_hi, u(0xff)));
        bool sample = n.rfind("image_sample", 0) == 0 || n.rfind("image_gather4", 0) == 0;
        auto vaddr = [&](int k) -> Id {
            if (i.nsa && k > 0) return get_v(i.nsa_addr[k - 1]);
            return get_v(uint16_t(i.vaddr + k));
        };
        if (sample) {
            // S# key (all four words; must match images.cpp sampler_key): {w0 ^ w2 * K, (w1 ^ w3) | 1 << 31}.
            Id skey_lo = bxor(get_s(i.ssamp), b.op(spv::OpIMul, t_u32, {get_s(uint16_t(i.ssamp + 2)), u(0x85ebca6bu)}));
            Id skey_hi = bor(bxor(get_s(uint16_t(i.ssamp + 1)), get_s(uint16_t(i.ssamp + 3))), u(0x80000000u));
            Id sslot = slot_of(skey_lo, skey_hi);
            Id img = load(b.op(spv::OpAccessChain, p_uc_image, {sampled_images, tslot}), t_image2d);
            Id smp = load(b.op(spv::OpAccessChain, p_uc_sampler, {samplers, sslot}), b.t_sampler());
            Id si = b.op(spv::OpSampledImage, t_simage2d, {img, smp});
            // Address order: [offset] [bias] [z-compare] [derivatives] coords [lod].
            int k = 0;
            bool has_off = n.find("_o") != std::string_view::npos && n.back() == 'o';
            if (has_off) ++k;
            bool bias = n.find("_b") != std::string_view::npos;
            Id bias_v = 0;
            if (bias) bias_v = tof(vaddr(k++));
            bool cmp = n.find("_c") != std::string_view::npos;
            if (cmp) ++k;  // depth compare: sampled as plain for now
            bool grad = n.find("_d") != std::string_view::npos && n.find("_d16") == std::string_view::npos;
            Id dx = 0, dy = 0;
            if (grad) {
                dx = b.op(spv::OpCompositeConstruct, t_v2f, {tof(vaddr(k)), tof(vaddr(k + 1))});
                dy = b.op(spv::OpCompositeConstruct, t_v2f, {tof(vaddr(k + 2)), tof(vaddr(k + 3))});
                k += 4;
            }
            Id coord = b.op(spv::OpCompositeConstruct, t_v2f, {tof(vaddr(k)), tof(vaddr(k + 1))});
            k += (i.dim == 1 || i.dim == 5) ? 3 : 2;  // 3D / 2D array carry a third coordinate
            bool lz = n.find("_lz") != std::string_view::npos;
            bool lod = !lz && n.find("_l") != std::string_view::npos && n.find("_l_") == std::string_view::npos ? true : (n.find("_l") != std::string_view::npos && !lz);
            Id r;
            if (lz) r = b.op(spv::OpImageSampleExplicitLod, t_v4f, {si, coord, uint32_t(spv::ImageOperandsLodMask), f(0)});
            else if (lod) r = b.op(spv::OpImageSampleExplicitLod, t_v4f, {si, coord, uint32_t(spv::ImageOperandsLodMask), tof(vaddr(k))});
            else if (grad) r = b.op(spv::OpImageSampleExplicitLod, t_v4f, {si, coord, uint32_t(spv::ImageOperandsGradMask), dx, dy});
            else if (p.stage == HwStage::Ps) {
                r = bias ? b.op(spv::OpImageSampleImplicitLod, t_v4f, {si, coord, uint32_t(spv::ImageOperandsBiasMask), bias_v})
                         : b.op(spv::OpImageSampleImplicitLod, t_v4f, {si, coord});
            } else {
                r = b.op(spv::OpImageSampleExplicitLod, t_v4f, {si, coord, uint32_t(spv::ImageOperandsLodMask), f(0)});
            }
            uint16_t d = i.vdata;
            for (uint32_t c = 0; c < 4; ++c)
                if (i.dmask & (1u << c)) set_v(d++, tou(b.op(spv::OpCompositeExtract, t_f32, {r, c})));
            return;
        }
        if (n == "image_load" || n == "image_load_mip") {
            Id img = load(b.op(spv::OpAccessChain, p_uc_image, {sampled_images, tslot}), t_image2d);
            Id t_v2i = b.t_vec(t_i32, 2);
            Id coord = b.op(spv::OpCompositeConstruct, t_v2i, {bc(t_i32, vaddr(0)), bc(t_i32, vaddr(1))});
            Id mip = n == "image_load_mip" ? bc(t_i32, vaddr(2)) : b.c_i32(0);
            Id r = b.op(spv::OpImageFetch, t_v4f, {img, coord, uint32_t(spv::ImageOperandsLodMask), mip});
            uint16_t d = i.vdata;
            for (uint32_t c = 0; c < 4; ++c)
                if (i.dmask & (1u << c)) set_v(d++, tou(b.op(spv::OpCompositeExtract, t_f32, {r, c})));
            return;
        }
        if (n == "image_store" || n == "image_store_mip") {
            Id img = load(b.op(spv::OpAccessChain, b.t_ptr(spv::StorageClassUniformConstant, t_storage2d), {storage_images, tslot}), t_storage2d);
            Id t_v2i = b.t_vec(t_i32, 2);
            Id coord = b.op(spv::OpCompositeConstruct, t_v2i, {bc(t_i32, vaddr(0)), bc(t_i32, vaddr(1))});
            Id comps[4] = {f(0), f(0), f(0), f(1)};
            uint16_t d = i.vdata;
            for (uint32_t c = 0; c < 4; ++c)
                if (i.dmask & (1u << c)) comps[c] = tof(get_v(d++));
            Id texel = b.op(spv::OpCompositeConstruct, t_v4f, {comps[0], comps[1], comps[2], comps[3]});
            // Only active lanes store: branch around.
            Id yes = b.id(), after = b.id();
            Id act = active();
            b.op_void(spv::OpSelectionMerge, {after, 0});
            b.op_void(spv::OpBranchConditional, {act, yes, after});
            b.label(yes);
            b.op_void(spv::OpImageWrite, {img, coord, texel});
            b.op_void(spv::OpBranch, {after});
            b.label(after);
            return;
        }
        if (n == "image_get_resinfo") {
            Id img = load(b.op(spv::OpAccessChain, p_uc_image, {sampled_images, tslot}), t_image2d);
            Id t_v2i = b.t_vec(t_i32, 2);
            Id sz = b.op(spv::OpImageQuerySizeLod, t_v2i, {img, bc(t_i32, vaddr(0))});
            Id levels = b.op(spv::OpImageQueryLevels, t_i32, {img});
            Id vals[4] = {b.op(spv::OpCompositeExtract, t_i32, {sz, 0}), b.op(spv::OpCompositeExtract, t_i32, {sz, 1}), b.c_i32(1), levels};
            uint16_t d = i.vdata;
            for (uint32_t c = 0; c < 4; ++c)
                if (i.dmask & (1u << c)) set_v(d++, tou(vals[c]));
            return;
        }
        unsupported(i);
    }

