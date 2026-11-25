// keylime: recording one title command list into Vulkan commands (part of the command executor; included by exec.cpp).
// --------------------------------------------------------------- builder ----
struct Arg {
    uint64_t value = 0;
    uint32_t c[64] = {};
};
struct Bind {
    void* root = nullptr;
    pipeline::State* pso = nullptr;
    Arg args[64];
};

template <class... A>
void once(const char* what, A... a) {
    static std::mutex m;
    static std::unordered_map<std::string, int> seen;
    std::lock_guard g(m);
    if (seen[what]++ == 0) KL_WARN("gpu", what, a...);
}

struct Builder {
    HostQueue& hq;
    Cb* c;
    VkCommandBuffer cb;
    Bind bind[2];  // 0 compute, 1 graphics
    uint64_t view_heap = 0, sampler_heap = 0;
    cmd::IndexBuffer ib{};
    cmd::VertexView vb[32]{};
    uint32_t topology = 4;
    cmd::Viewports vps{};
    cmd::Scissors scs{};
    float blend[4] = {1, 1, 1, 1};
    uint32_t stencil_ref = 0;
    cmd::DepthBounds depth_bounds{0.0f, 1.0f};
    cmd::Targets targets{};
    bool rendering = false;
    Attachments att;
    VkExtent2D extent{0, 0};
    VkPipeline bound = VK_NULL_HANDLE;
    bool set_bound[2] = {false, false};
    std::vector<Cb*>* out;  // earlier work of this submission (a memory wait sends it first)

    Builder(HostQueue& h, Cb* cc, std::vector<Cb*>* o = nullptr) : hq(h), c(cc), cb(cc->cb), out(o) {}

    // Work after a memory wait goes in a submission of its own, held at the queue's gate.
    void wait_value(const cmd::WaitValue& w) {
        if (!out || !hq.gate) {
            once("GPU waits on memory values are ignored here");
            return;
        }
        end_rendering();
        full_barrier(cb);
        Cb* next = get_cb(hq);
        if (!next) {
            once("a GPU wait on memory was ignored (no command buffer free)");
            return;
        }
        out->push_back(c);
        prepare_and_submit(hq, int(&hq - g_q), std::move(*out));
        out->clear();
        Gate gt;
        gt.w = w;
        push_gate(hq, gt);
        c = next, cb = next->cb;
        bound = VK_NULL_HANDLE;
        set_bound[0] = set_bound[1] = false;
    }

    void end_rendering() {
        if (rendering) vk::vkCmdEndRendering(cb), rendering = false;
    }

    bool begin_rendering() {
        VkRenderingAttachmentInfo color[8]{}, depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        att = {};
        uint32_t w = ~0u, h = ~0u;
        for (uint32_t i = 0; i < targets.count; ++i) {
            descriptor::Target t;
            color[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.color[i] = VK_FORMAT_UNDEFINED;
            if (!descriptor::target_info(targets.rtv[i], t)) continue;
            images::Image* im = images::of_resource(t.resource);
            if (!im) continue;
            auto f = images::facts(im);
            VkFormat vf;
            color[i].imageView = images::target_view(im, t.format, t.mip, t.layer, &vf);
            color[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            color[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.color[i] = vf;
            w = std::min(w, std::max(1u, f.width >> t.mip));
            h = std::min(h, std::max(1u, f.height >> t.mip));
            images::gpu_wrote(im);
        }
        att.count = targets.count;
        // Trailing unbound slots need not be declared.
        while (att.count && att.color[att.count - 1] == VK_FORMAT_UNDEFINED) --att.count;
        descriptor::Target dt;
        bool has_depth = false;
        if (targets.dsv && descriptor::target_info(targets.dsv, dt)) {
            if (images::Image* im = images::of_resource(dt.resource)) {
                auto f = images::facts(im);
                if (f.depth) {
                    depth.imageView = images::target_view(im, 0, dt.mip, dt.layer);
                    depth.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    att.depth = f.format;
                    att.stencil = f.stencil;
                    has_depth = true;
                    w = std::min(w, std::max(1u, f.width >> dt.mip));
                    h = std::min(h, std::max(1u, f.height >> dt.mip));
                }
            }
        }
        if (w == ~0u) {
            // Nothing to render into: the viewport's size (rasterisation still runs for UAV writes).
            w = vps.count ? uint32_t(std::max(1.0f, vps.v[0][2])) : 1;
            h = vps.count ? uint32_t(std::max(1.0f, vps.v[0][3])) : 1;
        }
        extent = {w, h};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = att.count;
        ri.pColorAttachments = color;
        ri.pDepthAttachment = has_depth ? &depth : nullptr;
        ri.pStencilAttachment = has_depth && att.stencil ? &depth : nullptr;
        vk::vkCmdBeginRendering(cb, &ri);
        rendering = true;
        return true;
    }

    // The user SGPRs a program expects, from the bound root arguments.
    void sgprs(const shader::Program& p, const Bind& b, uint32_t* out) {
        for (auto& rp : p.layout) {
            if (rp.user_data >= 32) continue;
            const Arg& a = b.args[rp.slot & 63];
            uint32_t* o = out + rp.user_data;
            uint32_t n = std::min<uint32_t>(rp.sgprs ? rp.sgprs : 1, 32 - rp.user_data);
            switch (rp.root) {
                case shader::RootType::Constants:
                    for (uint32_t k = 0; k < n && k < 64; ++k) o[k] = a.c[k];
                    break;
                case shader::RootType::Table:
                    o[0] = uint32_t(a.value);
                    if (n > 1) o[1] = uint32_t(a.value >> 32);
                    break;
                case shader::RootType::Descriptor:
                    o[0] = uint32_t(a.value);
                    if (n > 1) o[1] = uint32_t(a.value >> 32);
                    if (a.value) memory::ensure(a.value);
                    break;
                case shader::RootType::ResourceHeap:
                    o[0] = uint32_t(view_heap);
                    if (n > 1) o[1] = uint32_t(view_heap >> 32);
                    break;
                case shader::RootType::SamplerHeap:
                    o[0] = uint32_t(sampler_heap);
                    if (n > 1) o[1] = uint32_t(sampler_heap >> 32);
                    break;
                case shader::RootType::ImmediateSampler: {
                    uint32_t w[4] = {};
                    if (root::static_sampler(b.root, rp.base_register, 0, w)) images::sampler(w);
                    for (uint32_t k = 0; k < n && k < 4; ++k) o[k] = w[k];
                    break;
                }
                default: break;
            }
        }
    }

    // The textures a program samples through tables: tell the image registry they are in use.
    void touch_images(const shader::Program& p, const Bind& b) {
        for (auto& u : p.used) {
            if (u.kind != shader::ResourceType::Srv || u.root != shader::RootType::Table) continue;
            uint64_t table = b.args[u.slot & 63].value;
            if (!table || (table >> 32)) continue;
            const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
            uint32_t type = w[3] >> 28;
            if (type < 8) continue;  // a buffer, not an image
            images::touch((uint64_t(w[0]) | uint64_t(w[1] & 0xff) << 32) << 8);
        }
    }

    void push(uint64_t rec) {
        uint64_t pc[4] = {rec, memory::page_table(), images::table(), 0};
        vk::vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_ALL, 0, 32, pc);
    }

    // KEYLIME_TRACE_DRAWS=n: describes the first n draws (what they run, read and render into).
    static int trace_budget() {
        static int n = [] {
            char v[16]{};
            return GetEnvironmentVariableA("KEYLIME_TRACE_DRAWS", v, 16) ? atoi(v) : 0;
        }();
        return n;
    }
    void trace_draw(const char* why, pipeline::State* st, Compiled* k, const uint32_t* rec) {
        static std::atomic<int> done{0}, seen{0};
        static int skip = [] {
            char v[16]{};
            return GetEnvironmentVariableA("KEYLIME_TRACE_SKIP", v, 16) ? atoi(v) : 0;
        }();
        if (seen++ < skip) return;
        if (done.load() >= trace_budget()) return;
        ++done;
        uint64_t vh = 0, ph = 0;
        for (auto& s : st->shaders) {
            if (s.stage == 0) vh = s.hash;
            if (s.stage == 1) ph = s.hash;
            if (s.stage > 1) continue;
            char path[128];
            CreateDirectoryA("D:/keylime/logs/traced", nullptr);
            snprintf(path, sizeof path, "D:/keylime/logs/traced/%s-%016llx.bin", s.stage ? "ps" : "vs", (unsigned long long)s.hash);
            if (FILE* f = fopen(path, "wb")) fwrite(s.code.data(), 1, s.code.size(), f), fclose(f);
        }
        char sg[512] = {}, *o = sg;
        if (rec)
            for (int i = 0; i < 16; ++i) o += snprintf(o, 16, "%x ", rec[i]);
        char pg[512] = {}, *q = pg;
        if (rec)
            for (int i = 32; i < 48; ++i) q += snprintf(q, 16, "%x ", rec[i]);
        KL_INFO("draw", "%s: vs %016llx ps %016llx topo %u | %u targets (fmt %d, %ux%u) depth %d | vb0 %llx/%u ib %llx | vs sgprs %s| ps sgprs %s",
                why, (unsigned long long)vh, (unsigned long long)ph, topology, att.count, int(att.color[0]), extent.width, extent.height,
                int(att.depth), (unsigned long long)vb[0].address, vb[0].stride, (unsigned long long)ib.address, sg, pg);
        {
            const auto& bl = st->blend[0];
            KL_INFO("draw", "  blend0 enable %u src %u dst %u op %u | alpha src %u dst %u op %u | mask %x | rts %u fmt0 %u dsv %u | cull %u depth %u/%u/%u",
                    bl.enable, bl.src, bl.dst, bl.op, bl.src_alpha, bl.dst_alpha, bl.op_alpha, bl.write_mask, st->rt_count,
                    st->rt_formats[0], st->dsv_format, st->raster.cull, st->depth.depth_enable, st->depth.depth_write, st->depth.depth_func);
        }
        if (k && k->ps)
            for (auto& u : k->ps->used)
                if (u.root == shader::RootType::Table && u.kind == shader::ResourceType::Cbv) {
                    uint64_t table = bind[1].args[u.slot & 63].value;
                    if (!table) continue;
                    const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
                    uint64_t base = uint64_t(w[0]) | uint64_t(w[1] & 0xffff) << 32;
                    const float* c = reinterpret_cast<const float*>(base);
                    KL_INFO("draw", "  ps cb at %llx: %g %g %g %g | %g %g %g %g | %g %g %g %g", (unsigned long long)base, c[0], c[1], c[2], c[3],
                            c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]);
                }
        for (auto& e : st->input)
            KL_INFO("draw", "  input %s%u format %u slot %u offset %u%s", e.semantic.c_str(), e.index, e.format, e.slot, e.offset,
                    e.per_instance ? " per instance" : "");
        if (k)
            for (auto& a : k->attrs) KL_INFO("draw", "  attribute location %u binding %u vk format %d offset %u", a.location, a.binding, int(a.format), a.offset);
        if (k && k->vs)
            for (auto& u : k->vs->used) {
                if (u.root != shader::RootType::Table) continue;
                uint64_t table = bind[1].args[u.slot & 63].value;
                if (!table) continue;
                const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
                uint64_t base = uint64_t(w[0]) | uint64_t(w[1] & 0xffff) << 32;
                KL_INFO("draw", "  vs reads kind %d slot %u +%u dwords %u: %08x %08x %08x %08x (base %llx stride %u records %u)", int(u.kind),
                        u.slot, u.table_offset_dwords, u.dwords, w[0], w[1], w[2], w[3], (unsigned long long)base, (w[1] >> 16) & 0x3fff, w[2]);
                if (base) {
                    const float* f = reinterpret_cast<const float*>(base);
                    KL_INFO("draw", "    data: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g", f[0], f[1], f[2], f[3], f[4], f[5],
                            f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
                }
            }
        if (k && k->vs)
            for (auto& in : k->vs->vs_inputs) KL_INFO("draw", "  vs input row %u mask %x -> v%u", in.row, in.mask, in.vgpr);
        if (vb[0].address) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(vb[0].address);
            char t[400], *o = t;
            for (int i = 0; i < 18; ++i) o += snprintf(o, 24, "%08x ", w[i]);
            KL_INFO("draw", "  vb0: %s", t);
        }
        if (k && k->vs)
            for (auto& rp : k->vs->layout)
                KL_INFO("draw", "  vs root: type %d slot %u sgpr %u n %u off %u", int(rp.root), rp.slot, rp.user_data, rp.sgprs, rp.table_offset_dwords);
        if (k && k->ps) {
            for (auto& rp : k->ps->layout)
                KL_INFO("draw", "  ps root: type %d slot %u sgpr %u n %u off %u", int(rp.root), rp.slot, rp.user_data, rp.sgprs, rp.table_offset_dwords);
            // The descriptors the PS reads from tables: what they name.
            for (auto& u : k->ps->used) {
                if (u.root != shader::RootType::Table) continue;
                uint64_t table = bind[1].args[u.slot & 63].value;
                if (!table) continue;
                const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
                uint64_t addr = (uint64_t(w[0]) | uint64_t(w[1] & 0xff) << 32) << 8;
                KL_INFO("draw", "  ps reads kind %d slot %u +%u: words %08x %08x %08x %08x -> %s", int(u.kind), u.slot, u.table_offset_dwords,
                        w[0], w[1], w[2], w[3], u.kind == shader::ResourceType::Srv ? images::describe(addr) : "");
            }
        }
    }

    bool draw_setup(bool indexed, bool mesh = false) {
        pipeline::State* st = bind[1].pso;
        if (!st || st->compute) return false;
        Compiled* k = obtain(st, true);
        if (!wait_budget([k] { return k->translated.load(std::memory_order_acquire); }, 0)) {
            dropped("draw");
            return false;
        }
        if (!k->ok) {
            trace_draw("skipped (pipeline not translated)", st, nullptr, nullptr);
            return false;
        }
        if (k->mesh != mesh) {
            once(mesh ? "a mesh dispatch with a vertex shader pipeline was skipped" : "a draw with a mesh shader pipeline was skipped");
            return false;
        }
        if (!mesh && (topology_class(topology) == ~0u || topology_class(topology) == 3)) {
            once("draws with topology %u are skipped (not supported yet)", topology);
            return false;
        }
        if (!rendering && !begin_rendering()) return false;
        VkPipeline p = variant(k, st, att, mesh ? 4 : topology);
        if (!p) return false;
        if (p != bound) vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p), bound = p;
        if (!set_bound[1]) {
            VkDescriptorSet s = images::set();
            vk::vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_layout, 0, 1, &s, 0, nullptr);
            set_bound[1] = true;
        }
        uint64_t rec;
        uint32_t* r = record(c, RECORD_DWORDS, rec);
        if (!r) return false;
        memset(r, 0, RECORD_DWORDS * 4);
        sgprs(*k->vs, bind[1], r);
        if (k->ps) sgprs(*k->ps, bind[1], r + 32);
        if (k->ps) touch_images(*k->ps, bind[1]);
        touch_images(*k->vs, bind[1]);
        push(rec);
        trace_draw("draw", st, k, r);
        // Viewport (flipped: D3D's y points down in NDC terms), scissor.
        VkViewport v{0, float(extent.height), float(extent.width), -float(extent.height), 0, 1};
        if (vps.count) {
            const float* x = vps.v[0];
            v = {x[0], x[1] + x[3], x[2], -x[3], x[4], x[5]};
        }
        vk::vkCmdSetViewport(cb, 0, 1, &v);
        VkRect2D sc{{0, 0}, extent};
        if (scs.count) {
            const int32_t* s = scs.r[0];
            int32_t l = std::max(0, s[0]), t = std::max(0, s[1]);
            sc = {{l, t}, {uint32_t(std::max(0, s[2] - l)), uint32_t(std::max(0, s[3] - t))}};
        }
        vk::vkCmdSetScissor(cb, 0, 1, &sc);
        vk::vkCmdSetBlendConstants(cb, blend);
        vk::vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref);
        if (st->depth.bounds && att.depth != VK_FORMAT_UNDEFINED && host::get().caps.depth_bounds)
            vk::vkCmdSetDepthBounds(cb, std::clamp(depth_bounds.min, 0.0f, 1.0f), std::clamp(depth_bounds.max, 0.0f, 1.0f));
        if (mesh) return true;
        vk::vkCmdSetPrimitiveTopology(cb, vk_topology(topology));
        bool strip = topology == 3 || topology == 5 || topology == 11 || topology == 13;
        vk::vkCmdSetPrimitiveRestartEnable(cb, indexed && strip && st->strip_cut ? VK_TRUE : VK_FALSE);
        // Vertex streams.
        for (auto& bd : k->binds) {
            const auto& view = vb[bd.binding];
            VkBuffer buf = g_zero.buffer;
            VkDeviceSize off = 0, stride = view.stride;
            uint64_t left;
            if (view.address && memory::ensure(view.address, 1)) memory::locate(view.address, buf, off, left);
            else stride = 0;
            vk::vkCmdBindVertexBuffers2(cb, bd.binding, 1, &buf, &off, nullptr, &stride);
        }
        if (indexed) {
            VkBuffer buf;
            VkDeviceSize off;
            uint64_t left;
            if (!ib.address || !memory::ensure(ib.address, 1) || !memory::locate(ib.address, buf, off, left)) {
                once("indexed draw without a usable index buffer (%llx) skipped", (unsigned long long)ib.address);
                return false;
            }
            vk::vkCmdBindIndexBuffer(cb, buf, off, ib.format == 42 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        }
        return true;
    }

    Compiled* dispatch_setup() {
        pipeline::State* st = bind[0].pso;
        if (!st || !st->compute) return nullptr;
        bool pending;
        VkPipeline p = runnable(st, pending);
        if (!p) {
            if (pending) dropped("dispatch");
            else once("a dispatch whose pipeline could not be built was dropped");
            return nullptr;
        }
        Compiled* k = static_cast<Compiled*>(st->host);
        end_rendering();
        if (p != bound) vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p), bound = p;
        if (!set_bound[0]) {
            VkDescriptorSet s = images::set();
            vk::vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_layout, 0, 1, &s, 0, nullptr);
            set_bound[0] = true;
        }
        uint64_t rec;
        uint32_t* r = record(c, RECORD_DWORDS, rec);
        if (!r) return nullptr;
        memset(r, 0, RECORD_DWORDS * 4);
        sgprs(*k->cs, bind[0], r);
        touch_images(*k->cs, bind[0]);
        push(rec);
        return k;
    }

    void mark(uint32_t kind, uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) {
        const pipeline::State* st = bind[kind == 1 || kind == 7 ? 0 : 1].pso;
        checkpoint(cb, kind, st ? pipeline_key(st) : 0, x, y, z);
        if (kind >= 1 && kind <= 3) stamp(kind, st ? pipeline_key(st) : 0);
        else if (kind == 7) stamp(0, 0);  // the dispatch's end
    }
    // A profiling timestamp: kind 1-3 starts a dispatch or draw, 0 ends it (read_profile charges
    // the time between the two).
    void stamp(uint32_t kind, uint64_t key) {
        if (!hq.prof || hq.prof_open.size() >= PROF - 2) return;
        uint32_t slot = hq.prof_next;
        hq.prof_next = (hq.prof_next + 1) % PROF;
        vk::vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, hq.prof, slot);
        hq.prof_open.push_back({slot, kind, key, hq.prof_list});
    }
    void dispatch(const cmd::Dispatch& d) {
        if (!dispatch_setup()) return;
        mark(1, d.x, d.y, d.z);
        struct After {  // a checkpoint behind the dispatch too: proves whether it finished
            Builder* b;
            ~After() { b->mark(7); }
        } after{this};
        if (uint64_t(d.x) * d.y * d.z > (1ull << 22)) {
            static std::atomic<int> n{0};
            if (n++ < 8) KL_WARN("gpu", "a dispatch of %u x %u x %u groups (pipeline %016llx)", d.x, d.y, d.z,
                                 (unsigned long long)pipeline_key(bind[0].pso));
        }
        vk::vkCmdDispatch(cb, d.x, d.y, d.z);
        stats::add(stats::DispatchesRun);
    }

    static uint64_t device_of(VkBuffer b, VkDeviceSize off) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b;
        return vk::vkGetBufferDeviceAddress(dev(), &ai) + off;
    }

    void dispatch_mesh(const cmd::Dispatch& d) {
        uint32_t x = std::min(d.x, 65535u), y = std::min(d.y, 65535u), z = std::min(d.z, 65535u);
        if (!x || !y || !z) return;
        if (uint64_t(x) * y * z > (1u << 22)) {
            once("a mesh dispatch of %ux%ux%u groups was skipped (over the GPU's limit)", d.x, d.y, d.z);
            stats::add(stats::DrawsSkipped);
            return;
        }
        if (!draw_setup(false, true)) {
            stats::add(stats::DrawsSkipped);
            return;
        }
        mark(3);
        vk::vkCmdDrawMeshTasksEXT(cb, x, y, z);
        stats::add(stats::DrawsRun);
        stamp(0, 0);
    }

    // ExecuteIndirect with a single Draw, DrawIndexed, Dispatch or DispatchMesh argument per command.
    void indirect(const cmd::Indirect& x) {
        static const uint32_t bytes[] = {16, 20, 12, 12};
        uint32_t stride = 0;
        std::vector<std::array<uint32_t, 4>> args;
        if (!misc::signature(x.signature, stride, args) || args.size() != 1) {
            once("ExecuteIndirect with %zu arguments per command is skipped (single-argument signatures run)", args.size());
            return;
        }
        uint32_t type = args[0][0], max = uint32_t(std::min<uint64_t>(x.max, 1u << 16));
        if (type == 105) type = 2;
        if (type == 10 && host::get().caps.mesh_shader) type = 3;
        if (type > 3) {
            once("ExecuteIndirect with argument type %u is skipped (not supported yet)", type);
            return;
        }
        if (!max) return;
        if (max > 1 && ((stride & 3) || stride < bytes[type])) {
            once("ExecuteIndirect with stride %u is skipped", stride);
            return;
        }
        VkBuffer ab, cbuf = VK_NULL_HANDLE;
        VkDeviceSize aoff, coff = 0;
        if ((x.args & 3) || !where(x.args, uint64_t(stride) * (max - 1) + bytes[type], ab, aoff)) {
            once("ExecuteIndirect with arguments the GPU cannot see (%llx) is skipped", (unsigned long long)x.args);
            return;
        }
        bool counted = x.count != 0;
        if (counted && ((x.count & 3) || !where(x.count, 4, cbuf, coff))) {
            once("ExecuteIndirect with a count the GPU cannot see (%llx) is skipped", (unsigned long long)x.count);
            return;
        }
        // Arguments go through a GPU pass first: past the count, or sizes no frame would use
        // (memory a dropped dispatch never wrote), a command becomes empty instead of a GPU hang.
        uint32_t words = bytes[type] / 4;
        uint64_t tight;
        uint32_t* r = g_clamp[type] ? record(c, max * words + 2, tight) : nullptr;
        if (!r) return;
        r[max * words] = max;
        r[max * words + 1] = 0;
        auto& blk = c->records[c->block];
        VkBuffer db = blk.buffer;
        VkDeviceSize doff = tight - blk.device;
        end_rendering();
        full_barrier(cb);
        vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_clamp[type]);
        bound = g_clamp[type];
        struct {
            uint64_t src, dst, count;
            uint32_t stride, max;
        } pc{device_of(ab, aoff), tight, counted ? device_of(cbuf, coff) : tight + uint64_t(max) * words * 4, stride, max};
        vk::vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_ALL, 0, 32, &pc);
        checkpoint(cb, 4, 0, (max + 63) / 64, 1, 1);  // keylime's indirect argument check
        vk::vkCmdDispatch(cb, (max + 63) / 64, 1, 1);
        full_barrier(cb);
        if (type == 2) {
            if (!dispatch_setup()) return;
            mark(1);
            for (uint32_t i = 0; i < max; ++i) vk::vkCmdDispatchIndirect(cb, db, doff + VkDeviceSize(i) * 12);
            stats::add(stats::DispatchesRun, max);
            stamp(0, 0);
            return;
        }
        if (!draw_setup(type == 1, type == 3)) {
            stats::add(stats::DrawsSkipped);
            return;
        }
        mark(type == 3 ? 3 : 2);
        if (type == 3) vk::vkCmdDrawMeshTasksIndirectEXT(cb, db, doff, max, 12);
        else if (type == 0) vk::vkCmdDrawIndirect(cb, db, doff, max, 16);
        else vk::vkCmdDrawIndexedIndirect(cb, db, doff, max, 20);
        stats::add(stats::DrawsRun);
        stamp(0, 0);
    }

    // A title buffer range as (buffer, offset); false if the GPU cannot see it.
    bool where(uint64_t a, uint64_t len, VkBuffer& b, VkDeviceSize& off) {
        uint64_t left;
        return a && memory::ensure(a, len) && memory::locate(a, b, off, left) && left >= len;
    }

    void write_value(const cmd::Write& w, bool is64) {
        if (!w.address) return;
        uint64_t v = w.value;
        if (is64 && (w.flags == 1 || w.flags == 2)) {
            LARGE_INTEGER q;
            QueryPerformanceCounter(&q);
            v = uint64_t(q.QuadPart);
        }
        VkBuffer b;
        VkDeviceSize off;
        if ((w.address & 3) == 0 && where(w.address, w.bytes, b, off)) {
            end_rendering();
            full_barrier(cb);
            vk::vkCmdUpdateBuffer(cb, b, off, w.bytes, &v);
            full_barrier(cb);
        } else if (w.bytes == 4) {
            *reinterpret_cast<volatile uint32_t*>(w.address) = uint32_t(v);
        } else {
            *reinterpret_cast<volatile uint64_t*>(w.address) = v;
        }
    }

    void copy_memory(uint64_t dst, uint64_t src, uint64_t len) {
        if (!dst || !src || !len) return;
        checkpoint(cb, 5, dst, uint32_t(len), 0, 0);
        VkBuffer db, sb;
        VkDeviceSize doff, soff;
        if (where(dst, len, db, doff) && where(src, len, sb, soff)) {
            end_rendering();
            full_barrier(cb);
            VkBufferCopy r{soff, doff, len};
            vk::vkCmdCopyBuffer(cb, sb, db, 1, &r);
            full_barrier(cb);
            images::written(dst, len);
        } else if (!safe_move(dst, src, len)) {
            once("a copy between unmapped title memory (%llx <- %llx) was dropped", (unsigned long long)dst, (unsigned long long)src);
        }
    }

    static void subresource(images::Image* im, uint32_t index, uint32_t& mip, uint32_t& layer) {
        auto f = images::facts(im);
        mip = index % f.mips, layer = index / f.mips;
    }

    void copy_texture(const cmd::CopyTexture& t) {
        end_rendering();
        checkpoint(cb, 6, 0);
        images::Image* di = t.dst.type == 0 ? images::of_resource(t.dst.resource) : nullptr;
        images::Image* si = t.src.type == 0 ? images::of_resource(t.src.resource) : nullptr;
        full_barrier(cb);
        if (di && si) {
            uint32_t dm, dl, sm, sl;
            subresource(di, uint32_t(t.dst.offset), dm, dl);
            subresource(si, uint32_t(t.src.offset), sm, sl);
            auto sf = images::facts(si), df = images::facts(di);
            VkImageCopy r{};
            r.srcSubresource = {VkImageAspectFlags(sf.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), sm, sl, 1};
            r.dstSubresource = {VkImageAspectFlags(df.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), dm, dl, 1};
            uint32_t sw = std::max(1u, sf.width >> sm), sh = std::max(1u, sf.height >> sm);
            if (t.has_box) r.srcOffset = {int32_t(t.box[0]), int32_t(t.box[1]), 0}, sw = t.box[3] - t.box[0], sh = t.box[4] - t.box[1];
            r.dstOffset = {int32_t(t.x), int32_t(t.y), 0};
            uint32_t dw = std::max(1u, df.width >> dm), dh = std::max(1u, df.height >> dm);
            r.extent = {std::min(sw, dw - std::min(dw, t.x)), std::min(sh, dh - std::min(dh, t.y)), 1};
            if (r.extent.width && r.extent.height)
                vk::vkCmdCopyImage(cb, sf.image, VK_IMAGE_LAYOUT_GENERAL, df.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            images::gpu_wrote(di);
        } else if (di && t.src.type == 1) {
            // Buffer footprint -> texture.
            resource::Info ri;
            if (!resource::info(t.src.resource, ri)) return;
            uint32_t bytes = 0, block = 1;
            texture::element(t.src.format, bytes, block);
            if (!bytes) return;
            uint32_t dm, dl;
            subresource(di, uint32_t(t.dst.offset), dm, dl);
            auto df = images::facts(di);
            uint64_t rows = (t.src.height + block - 1) / block;
            uint64_t len = uint64_t(t.src.row_pitch) * rows * std::max(1u, t.src.depth);
            VkBuffer b;
            VkDeviceSize off;
            if (!where(ri.address + t.src.offset, len, b, off)) return;
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.bufferRowLength = t.src.row_pitch / bytes * block;
            r.bufferImageHeight = uint32_t(rows * block);
            r.imageSubresource = {VkImageAspectFlags(df.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), dm, dl, 1};
            r.imageOffset = {int32_t(t.x), int32_t(t.y), 0};
            uint32_t w = t.src.width, h = t.src.height;
            if (t.has_box) {
                r.bufferOffset += uint64_t(t.box[1] / block) * t.src.row_pitch + uint64_t(t.box[0] / block) * bytes;
                w = t.box[3] - t.box[0], h = t.box[4] - t.box[1];
            }
            uint32_t dw = std::max(1u, df.width >> dm), dh = std::max(1u, df.height >> dm);
            r.imageExtent = {std::min(w, dw - std::min(dw, t.x)), std::min(h, dh - std::min(dh, t.y)), 1};
            if (r.imageExtent.width && r.imageExtent.height)
                vk::vkCmdCopyBufferToImage(cb, b, df.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            images::gpu_wrote(di);
        } else if (si && t.dst.type == 1) {
            // Texture -> buffer footprint (readback).
            resource::Info ri;
            if (!resource::info(t.dst.resource, ri)) return;
            uint32_t bytes = 0, block = 1;
            texture::element(t.dst.format, bytes, block);
            if (!bytes) return;
            uint32_t sm, sl;
            subresource(si, uint32_t(t.src.offset), sm, sl);
            auto sf = images::facts(si);
            uint64_t rows = (t.dst.height + block - 1) / block;
            uint64_t len = uint64_t(t.dst.row_pitch) * rows;
            VkBuffer b;
            VkDeviceSize off;
            if (!where(ri.address + t.dst.offset, len, b, off)) return;
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.bufferRowLength = t.dst.row_pitch / bytes * block;
            r.bufferImageHeight = uint32_t(rows * block);
            r.imageSubresource = {VkImageAspectFlags(sf.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), sm, sl, 1};
            uint32_t w = std::min(t.dst.width, std::max(1u, sf.width >> sm)), h = std::min(t.dst.height, std::max(1u, sf.height >> sm));
            r.imageExtent = {w, h, 1};
            vk::vkCmdCopyImageToBuffer(cb, sf.image, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r);
        }
        full_barrier(cb);
    }

    void copy_resource(const cmd::CopyResource& r) {
        resource::Info d, s;
        if (!resource::info(r.dst, d) || !resource::info(r.src, s)) return;
        if (d.desc.dimension == resource::BUFFER && s.desc.dimension == resource::BUFFER) {
            copy_memory(d.address, s.address, std::min(d.desc.width, s.desc.width));
            return;
        }
        images::Image* di = images::of_resource(r.dst);
        images::Image* si = images::of_resource(r.src);
        if (!di || !si) return;
        end_rendering();
        full_barrier(cb);
        auto df = images::facts(di), sf = images::facts(si);
        std::vector<VkImageCopy> rs;
        for (uint32_t m = 0; m < std::min(df.mips, sf.mips); ++m) {
            VkImageCopy c{};
            VkImageAspectFlags a = df.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            c.srcSubresource = {a, m, 0, std::min(df.depth_or_layers, sf.depth_or_layers)};
            c.dstSubresource = c.srcSubresource;
            c.extent = {std::max(1u, std::min(df.width, sf.width) >> m), std::max(1u, std::min(df.height, sf.height) >> m), 1};
            rs.push_back(c);
        }
        vk::vkCmdCopyImage(cb, sf.image, VK_IMAGE_LAYOUT_GENERAL, df.image, VK_IMAGE_LAYOUT_GENERAL, uint32_t(rs.size()), rs.data());
        images::gpu_wrote(di);
        full_barrier(cb);
    }

    void clear_rtv(const cmd::ClearRtv& c) {
        descriptor::Target t;
        if (!descriptor::target_info(c.handle, t)) return;
        images::Image* im = images::of_resource(t.resource);
        if (!im) return;
        end_rendering();
        auto f = images::facts(im);
        VkClearColorValue v;
        memcpy(v.float32, c.color, 16);
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, std::min(t.mip, f.mips - 1), 1, std::min(t.layer, f.depth_or_layers - 1), 1};
        full_barrier(cb);
        vk::vkCmdClearColorImage(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        full_barrier(cb);
        images::gpu_wrote(im);
    }

    void clear_dsv(const cmd::ClearDsv& c) {
        descriptor::Target t;
        if (!descriptor::target_info(c.handle, t)) return;
        images::Image* im = images::of_resource(t.resource);
        if (!im) return;
        auto f = images::facts(im);
        if (!f.depth) return;
        end_rendering();
        VkImageAspectFlags a = 0;
        if (c.flags & 1) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if ((c.flags & 2) && f.stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
        if (!a) return;
        VkClearDepthStencilValue v{c.depth, c.stencil};
        VkImageSubresourceRange r{a, std::min(t.mip, f.mips - 1), 1, std::min(t.layer, f.depth_or_layers - 1), 1};
        full_barrier(cb);
        vk::vkCmdClearDepthStencilImage(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        full_barrier(cb);
    }

    void clear_uav(const cmd::ClearUav& c, bool is_float) {
        void* res = reinterpret_cast<void*>(c.resource);
        resource::Info ri;
        if (!resource::info(res, ri)) return;
        end_rendering();
        if (ri.desc.dimension == resource::BUFFER) {
            VkBuffer b;
            VkDeviceSize off;
            if (where(ri.address, ri.desc.width, b, off)) {
                full_barrier(cb);
                vk::vkCmdFillBuffer(cb, b, off, ri.desc.width & ~3ull, c.values[0]);
                full_barrier(cb);
            }
            return;
        }
        images::Image* im = images::of_resource(res);
        if (!im) return;
        auto f = images::facts(im);
        if (f.depth) return;
        VkClearColorValue v;
        memcpy(v.uint32, c.values, 16);
        (void)is_float;
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        full_barrier(cb);
        vk::vkCmdClearColorImage(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        full_barrier(cb);
        images::gpu_wrote(im);
    }

    void run(const Stream& s) {
        uint32_t index = 0;
        s.each([&](Op op, const uint8_t* p, uint32_t size) {
            auto get = [&](auto& v) { memcpy(&v, p, std::min<size_t>(sizeof v, size)); };
            static const bool every = [] { char v[8]{}; return GetEnvironmentVariableA("KEYLIME_CHECKPOINTS", v, 8) && !strcmp(v, "all"); }();
            if (every) checkpoint(cb, 8, uint64_t(op), index);  // KEYLIME_CHECKPOINTS=all: every command
            ++index;
            switch (op) {
                case Op::SetPipeline: {
                    uint64_t v;
                    get(v);
                    if (auto* st = pipeline::state(reinterpret_cast<void*>(v))) bind[st->compute ? 0 : 1].pso = st;
                    break;
                }
                case Op::RootSignature: {
                    cmd::Root r;
                    get(r);
                    bind[r.gfx & 1].root = reinterpret_cast<void*>(r.value);
                    break;
                }
                case Op::RootTable:
                case Op::RootView: {
                    cmd::Root r;
                    get(r);
                    bind[r.gfx & 1].args[r.slot & 63].value = r.value;
                    break;
                }
                case Op::RootConstants: {
                    cmd::RootConstants r{};
                    get(r);
                    Arg& a = bind[r.gfx & 1].args[r.slot & 63];
                    for (uint32_t k = 0; k < r.count && r.first + k < 64; ++k) a.c[r.first + k] = r.values[k];
                    break;
                }
                case Op::Heaps: {
                    cmd::Heaps h{};
                    get(h);
                    for (uint32_t i = 0; i < h.count; ++i) {
                        uint32_t kind;
                        uint64_t start;
                        if (!descriptor::heap_info(h.heaps[i], kind, start)) continue;
                        if (kind == 0) view_heap = start;
                        if (kind == 1) sampler_heap = start;
                    }
                    break;
                }
                case Op::IndexBuffer: get(ib); break;
                case Op::VertexBuffers: {
                    cmd::VertexBuffers v{};
                    get(v);
                    for (uint32_t i = 0; i < v.count && v.start + i < 32; ++i) vb[v.start + i] = v.views[i];
                    break;
                }
                case Op::Topology: {
                    uint64_t t;
                    get(t);
                    topology = uint32_t(t);
                    break;
                }
                case Op::Viewports: get(vps); break;
                case Op::Scissors: get(scs); break;
                case Op::BlendFactor: get(blend); break;
                case Op::DepthBounds: get(depth_bounds); break;
                case Op::StencilRef: {
                    uint64_t v;
                    get(v);
                    stencil_ref = uint32_t(v);
                    break;
                }
                case Op::Targets:
                    end_rendering();
                    targets = {};
                    get(targets);
                    break;
                case Op::ClearRtv: {
                    cmd::ClearRtv c;
                    get(c);
                    clear_rtv(c);
                    break;
                }
                case Op::ClearDsv: {
                    cmd::ClearDsv c;
                    get(c);
                    clear_dsv(c);
                    break;
                }
                case Op::ClearUavUint:
                case Op::ClearUavFloat: {
                    cmd::ClearUav c;
                    get(c);
                    clear_uav(c, op == Op::ClearUavFloat);
                    break;
                }
                case Op::Draw: {
                    cmd::Draw d;
                    get(d);
                    if (draw_setup(false)) {
                        mark(2);
                        vk::vkCmdDraw(cb, d.count, d.instances, d.first, d.first_instance);
                        stats::add(stats::DrawsRun);
        stamp(0, 0);
                    } else {
                        stats::add(stats::DrawsSkipped);
                    }
                    break;
                }
                case Op::DrawIndexed: {
                    cmd::DrawIndexed d;
                    get(d);
                    if (draw_setup(true)) {
                        mark(2);
                        vk::vkCmdDrawIndexed(cb, d.count, d.instances, d.first, d.base, d.first_instance);
                        stats::add(stats::DrawsRun);
        stamp(0, 0);
                    } else {
                        stats::add(stats::DrawsSkipped);
                    }
                    break;
                }
                case Op::Dispatch: {
                    cmd::Dispatch d;
                    get(d);
                    dispatch(d);
                    break;
                }
                case Op::Barrier:
                    end_rendering();
                    full_barrier(cb);
                    break;
                case Op::CopyBuffer:
                case Op::CopyMem: {
                    cmd::CopyBuffer c;
                    get(c);
                    copy_memory(c.dst, c.src, c.len);
                    break;
                }
                case Op::CopyTexture: {
                    cmd::CopyTexture t;
                    get(t);
                    copy_texture(t);
                    break;
                }
                case Op::CopyResource: {
                    cmd::CopyResource r;
                    get(r);
                    copy_resource(r);
                    break;
                }
                case Op::Write32:
                case Op::Write64: {
                    cmd::Write w;
                    get(w);
                    write_value(w, op == Op::Write64);
                    break;
                }
                case Op::ResolveQueries: {
                    cmd::ResolveQueries r;
                    get(r);
                    VkBuffer b;
                    VkDeviceSize off;
                    if (r.dst && (r.dst & 3) == 0 && where(r.dst, uint64_t(r.count) * 8, b, off)) {
                        end_rendering();
                        full_barrier(cb);
                        vk::vkCmdFillBuffer(cb, b, off, uint64_t(r.count) * 8, 0);
                        full_barrier(cb);
                    } else if (r.dst) {
                        memset(reinterpret_cast<void*>(r.dst), 0, size_t(r.count) * 8);
                    }
                    break;
                }
                case Op::ExecuteIndirect: {
                    cmd::Indirect x;
                    get(x);
                    indirect(x);
                    break;
                }
                case Op::DispatchMesh: {
                    cmd::Dispatch d;
                    get(d);
                    dispatch_mesh(d);
                    break;
                }
                case Op::WaitValue: {
                    cmd::WaitValue w;
                    get(w);
                    wait_value(w);
                    break;
                }
                default: break;
            }
        });
        end_rendering();
        full_barrier(cb);
    }
};
