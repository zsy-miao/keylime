// keylime GPU: title command lists -> Vulkan command buffers (see exec.h).
#include "gpu/exec.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gpu/descriptor.h"
#include "gpu/host.h"
#include "gpu/images.h"
#include "gpu/list.h"
#include "gpu/memory.h"
#include "gpu/pipeline.h"
#include "gpu/resource.h"
#include "gpu/retire.h"
#include "gpu/root.h"
#include "gpu/stream.h"
#include "gpu/texture.h"
#include "log.h"
#include "shader/container.h"
#include "shader/rectlist.h"
#include "shader/translate.h"
#include "stats.h"

namespace kl::gpu {
namespace exec {
namespace {

constexpr uint64_t RECORD_BLOCK = 4ull << 20;
constexpr uint32_t RECORD_DWORDS = 64;  // VS (or CS) SGPRs at 0, PS SGPRs at 32

struct Cb {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    uint64_t value = 0;
    std::vector<memory::Block> records;
    size_t block = 0;
    uint64_t used = 0;
};

struct HostQueue {
    VkQueue q = VK_NULL_HANDLE;
    uint32_t family = 0;
    std::mutex lock;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<uint64_t> submitted{0};
    // Commands are being recorded (they may name objects retired meanwhile: those wait for the
    // submission that carries them).
    std::atomic<int> recording{0};
    std::deque<Cb*> busy;
    std::vector<Cb*> idle;
};
HostQueue g_q[2];
VkPipelineLayout g_layout = VK_NULL_HANDLE;
memory::Block g_zero;  // binds where a vertex stream is missing
bool g_started = false;

VkDevice dev() { return host::get().device; }

uint64_t completed(HostQueue& h) {
    uint64_t v = 0;
    if (h.timeline) vk::vkGetSemaphoreCounterValue(dev(), h.timeline, &v);
    return v;
}

// A CPU copy that survives memory the title has not mapped (yet).
bool safe_move(uint64_t dst, uint64_t src, uint64_t len) {
    __try {
        memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), size_t(len));
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

// ---------------------------------------------------------- retiring ----
struct Retiree {
    uint64_t need[2];
    std::function<void()> f;
};
std::mutex g_retire_lock;
std::deque<Retiree> g_retirees;

void collect() {
    std::vector<std::function<void()>> due;
    {
        std::lock_guard g(g_retire_lock);
        if (g_retirees.empty()) return;
        uint64_t done[2] = {completed(g_q[0]), completed(g_q[1])};
        uint64_t sub[2] = {g_q[0].submitted.load(), g_q[1].submitted.load()};
        bool rec[2] = {g_q[0].recording.load() != 0, g_q[1].recording.load() != 0};
        while (!g_retirees.empty()) {
            auto& r = g_retirees.front();
            bool ok = true;
            // Done once the submission after the retiring is done; a queue that records nothing
            // now and finished all it was given will not use the object either.
            for (int q = 0; q < 2; ++q) ok &= done[q] >= r.need[q] || (!rec[q] && sub[q] < r.need[q] && done[q] >= sub[q]);
            if (!ok) break;
            due.push_back(std::move(r.f));
            g_retirees.pop_front();
        }
    }
    for (auto& f : due) f();
}

// ----------------------------------------------------- command buffers ----
Cb* get_cb(HostQueue& h) {
    uint64_t done = completed(h);
    while (!h.busy.empty() && h.busy.front()->value <= done) {
        Cb* c = h.busy.front();
        h.busy.pop_front();
        vk::vkResetCommandBuffer(c->cb, 0);
        h.idle.push_back(c);
    }
    Cb* c;
    if (!h.idle.empty()) {
        c = h.idle.back();
        h.idle.pop_back();
    } else {
        c = new Cb();
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = h.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (!KL_VK_CHECK(vk::vkAllocateCommandBuffers(dev(), &ai, &c->cb))) {
            delete c;
            return nullptr;
        }
    }
    c->block = 0, c->used = 0;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk::vkBeginCommandBuffer(c->cb, &bi);
    return c;
}

// A draw record of `dwords`: host pointer and device address.
uint32_t* record(Cb* c, uint32_t dwords, uint64_t& device) {
    uint64_t size = (uint64_t(dwords) * 4 + 15) & ~15ull;
    while (c->block < c->records.size() && c->used + size > c->records[c->block].size) ++c->block, c->used = 0;
    if (c->block == c->records.size()) {
        memory::Block b;
        if (!memory::make_block(RECORD_BLOCK, true, b)) return nullptr;
        c->records.push_back(b);
        c->used = 0;
    }
    auto& b = c->records[c->block];
    device = b.device + c->used;
    auto* p = reinterpret_cast<uint32_t*>(b.map + c->used);
    c->used += size;
    return p;
}

struct Semaphore {
    VkSemaphore s;
    uint64_t value;
};

void submit(HostQueue& h, const std::vector<Cb*>& cbs, const std::vector<Semaphore>& waits, std::vector<Semaphore> signals) {
    std::vector<VkCommandBufferSubmitInfo> ci;
    for (Cb* c : cbs) {
        vk::vkEndCommandBuffer(c->cb);
        VkCommandBufferSubmitInfo x{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        x.commandBuffer = c->cb;
        ci.push_back(x);
    }
    uint64_t value = h.submitted.load() + 1;
    signals.push_back({h.timeline, value});
    std::vector<VkSemaphoreSubmitInfo> w, s;
    for (auto& x : waits) {
        VkSemaphoreSubmitInfo i{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        i.semaphore = x.s, i.value = x.value, i.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        w.push_back(i);
    }
    for (auto& x : signals) {
        VkSemaphoreSubmitInfo i{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        i.semaphore = x.s, i.value = x.value, i.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        s.push_back(i);
    }
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = uint32_t(ci.size());
    si.pCommandBufferInfos = ci.data();
    si.waitSemaphoreInfoCount = uint32_t(w.size());
    si.pWaitSemaphoreInfos = w.data();
    si.signalSemaphoreInfoCount = uint32_t(s.size());
    si.pSignalSemaphoreInfos = s.data();
    VkResult r = vk::vkQueueSubmit2(h.q, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 8) KL_ERROR("vk", "queue submission failed: %s", vk::result_text(r));
        // Keep the counter honest: signal from the host so waits on it do not hang.
        VkSemaphoreSignalInfo sg{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        sg.semaphore = h.timeline, sg.value = value;
        vk::vkSignalSemaphore(dev(), &sg);
    }
    h.submitted.store(value);
    for (Cb* c : cbs) c->value = value, h.busy.push_back(c);
}

void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vk::vkCmdPipelineBarrier2(cb, &di);
}

// ------------------------------------------------------------ pipelines ----
struct Compiled {
    bool ok = false;
    shader::Container vsc, psc, csc;
    const shader::Program *vs = nullptr, *ps = nullptr, *cs = nullptr;
    VkShaderModule vsm = VK_NULL_HANDLE, psm = VK_NULL_HANDLE, csm = VK_NULL_HANDLE;
    std::vector<VkVertexInputAttributeDescription> attrs;
    std::vector<VkVertexInputBindingDescription> binds;
    VkPipeline compute = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, VkPipeline> variants;
    bool writes_color = false;
    std::vector<uint8_t> vs_rows;                          // VS output locations
    VkShaderModule gs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // rectangle / quad expansion
};

constexpr uint32_t RECTLIST = 17, QUADLIST = 19;

const shader::Program* find(const shader::Container& c, shader::HwStage s) {
    for (auto& p : c.programs)
        if (p.stage == s) return &p;
    return nullptr;
}

bool icase_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (tolower(uint8_t(a[i])) != tolower(uint8_t(b[i]))) return false;
    return true;
}

VkShaderModule module(const shader::Program& p, const shader::Options& o, uint64_t hash, shader::Translation& t) {
    std::string err;
    if (!shader::translate(p, t, err, &o)) {
        static int n = 0;
        if (n++ < 32) KL_WARN("gpu", "shader %016llx (%s) not translated: %s", (unsigned long long)hash, shader::stage_name(p.stage), err.c_str());
        return VK_NULL_HANDLE;
    }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = t.spirv.size() * 4;
    mi.pCode = t.spirv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    KL_VK_CHECK(vk::vkCreateShaderModule(dev(), &mi, nullptr, &m));
    return m;
}

Compiled* compile(pipeline::State* st) {
    std::lock_guard g(st->lock);
    if (st->host) return static_cast<Compiled*>(st->host);
    auto* k = new Compiled();
    st->host = k;
    const pipeline::Shader *vs = nullptr, *ps = nullptr, *cs = nullptr;
    for (auto& s : st->shaders) {
        if (s.stage == 0) vs = &s;
        if (s.stage == 1) ps = &s;
        if (s.stage == 5) cs = &s;
    }
    std::string err;
    if (st->compute) {
        if (!cs || !shader::parse(cs->code.data(), cs->code.size(), k->csc, err) || !(k->cs = find(k->csc, shader::HwStage::Cs))) {
            KL_WARN("gpu", "compute pipeline without a usable CS: %s", err.c_str());
            return k;
        }
        shader::Translation t;
        k->csm = module(*k->cs, {}, cs->hash, t);
        if (!k->csm) return k;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, k->csm, "main", nullptr};
        ci.layout = g_layout;
        if (vk::vkCreateComputePipelines(dev(), VK_NULL_HANDLE, 1, &ci, nullptr, &k->compute) != VK_SUCCESS) {
            KL_WARN("gpu", "compute pipeline %016llx not built by the driver", (unsigned long long)cs->hash);
            return k;
        }
        k->ok = true;
        return k;
    }
    if (!vs || !shader::parse(vs->code.data(), vs->code.size(), k->vsc, err) || !(k->vs = find(k->vsc, shader::HwStage::Vs))) {
        static int n = 0;
        if (n++ < 16) KL_WARN("gpu", "graphics pipeline without a usable VS (%s)", err.empty() ? "no VS program" : err.c_str());
        return k;
    }
    shader::Translation vt;
    shader::Options vo;
    vo.sgpr_offset = 0;
    k->vsm = module(*k->vs, vo, vs->hash, vt);
    if (!k->vsm) return k;
    k->vs_rows = vt.vs_output_rows;
    if (ps && shader::parse(ps->code.data(), ps->code.size(), k->psc, err) && (k->ps = find(k->psc, shader::HwStage::Ps))) {
        // Link PS inputs to VS outputs by semantic (rows differ when the signatures do).
        shader::Options po;
        po.sgpr_offset = 32;
        for (auto& in : k->psc.inputs) {
            if (in.system_value) continue;
            for (auto& out : k->vsc.outputs)
                if (!out.system_value && out.index == in.index && icase_equal(out.name, in.name)) {
                    po.ps_location[uint8_t(in.reg)] = uint8_t(out.reg);
                    break;
                }
        }
        shader::Translation pt;
        k->psm = module(*k->ps, po, ps->hash, pt);
        k->writes_color = pt.mrt_mask != 0;
        if (!k->psm) return k;
    }
    // Vertex input: each VS input row from the input layout element with its semantic.
    uint32_t slots_used = 0;
    for (auto& sig : k->vsc.inputs) {
        if (sig.system_value) continue;
        for (auto& e : st->input) {
            if (e.index != sig.index || !icase_equal(e.semantic, sig.name)) continue;
            VkFormat f = images::vk_format(e.format);
            if (f == VK_FORMAT_UNDEFINED) break;
            k->attrs.push_back({sig.reg, e.slot & 31, f, e.offset});
            if (!(slots_used & (1u << (e.slot & 31)))) {
                slots_used |= 1u << (e.slot & 31);
                k->binds.push_back({e.slot & 31, 0, e.per_instance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
            }
            break;
        }
    }
    k->ok = true;
    return k;
}

VkBlendFactor blend_factor(uint32_t b) {
    static const VkBlendFactor m[] = {VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR,
                                      VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                      VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_DST_COLOR,
                                      VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_ZERO,
                                      VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                                      VK_BLEND_FACTOR_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR, VK_BLEND_FACTOR_SRC1_ALPHA,
                                      VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, VK_BLEND_FACTOR_CONSTANT_ALPHA,
                                      VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
    return b < sizeof m / sizeof m[0] ? m[b] : VK_BLEND_FACTOR_ONE;
}
VkBlendOp blend_op(uint32_t o) {
    switch (o) {
        case 2: return VK_BLEND_OP_SUBTRACT;
        case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4: return VK_BLEND_OP_MIN;
        case 5: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}
VkCompareOp compare(uint32_t f) { return f >= 1 && f <= 8 ? VkCompareOp(f - 1) : VK_COMPARE_OP_ALWAYS; }
VkStencilOp stencil_op(uint32_t o) { return o >= 1 && o <= 8 ? VkStencilOp(o - 1) : VK_STENCIL_OP_KEEP; }
VkStencilOpState stencil(const pipeline::StencilFace& f, const pipeline::DepthStencil& d) {
    return {stencil_op(f.fail), stencil_op(f.pass), stencil_op(f.depth_fail), compare(f.func), d.read_mask, d.write_mask, 0};
}

// What a draw renders into (the variant key's parts).
struct Attachments {
    uint32_t count = 0;
    VkFormat color[8]{};
    VkFormat depth = VK_FORMAT_UNDEFINED;
    bool stencil = false;
};

uint32_t topology_class(uint32_t t) {  // 0 point 1 line 2 triangle 3 patch, ~0 unknown
    switch (t) {
        case 1: return 0;
        case 2: case 3: case 10: case 11: return 1;
        case 4: case 5: case 12: case 13: case RECTLIST: case QUADLIST: return 2;
        default: return t >= 33 && t <= 64 ? 3 : ~0u;
    }
}
VkPrimitiveTopology vk_topology(uint32_t t) {
    switch (t) {
        case 1: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case 2: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case 10: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
        case 11: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
        case 12: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
        case 13: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
        case QUADLIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;  // 4 vertices a primitive, into the GS
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

VkPipeline variant(Compiled* k, pipeline::State* st, const Attachments& a, uint32_t topo) {
    uint64_t key = 0xcbf29ce484222325ull;
    auto mix = [&](uint64_t v) { key = (key ^ v) * 0x100000001b3ull; };
    int expand = topo == RECTLIST ? 0 : topo == QUADLIST ? 1 : -1;
    mix(a.count), mix(a.depth), mix(a.stencil), mix(topology_class(topo)), mix(uint64_t(expand + 7));
    for (uint32_t i = 0; i < a.count; ++i) mix(a.color[i]);
    std::lock_guard g(st->lock);
    auto it = k->variants.find(key);
    if (it != k->variants.end()) return it->second;
    VkPipelineShaderStageCreateInfo stages[3];
    uint32_t nstages = 0;
    stages[nstages++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, k->vsm, "main", nullptr};
    if (expand >= 0) {
        if (!k->gs[expand]) {
            auto code = shader::rect_geometry(k->vs_rows, expand == 1);
            VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            mi.codeSize = code.size() * 4;
            mi.pCode = code.data();
            KL_VK_CHECK(vk::vkCreateShaderModule(dev(), &mi, nullptr, &k->gs[expand]));
        }
        if (!k->gs[expand]) return VK_NULL_HANDLE;
        stages[nstages++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_GEOMETRY_BIT, k->gs[expand], "main", nullptr};
    }
    if (k->psm)
        stages[nstages++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, k->psm, "main", nullptr};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = uint32_t(k->binds.size());
    vi.pVertexBindingDescriptions = k->binds.data();
    vi.vertexAttributeDescriptionCount = uint32_t(k->attrs.size());
    vi.pVertexAttributeDescriptions = k->attrs.data();
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = vk_topology(topo);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1, vp.scissorCount = 1;
    const auto& r = st->raster;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = r.depth_clip ? VK_FALSE : VK_TRUE;
    rs.polygonMode = r.fill == 2 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = r.cull == 2 ? VK_CULL_MODE_FRONT_BIT : r.cull == 3 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rs.frontFace = r.front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    rs.depthBiasEnable = (r.depth_bias || r.slope_bias != 0.0f) ? VK_TRUE : VK_FALSE;
    rs.depthBiasConstantFactor = float(r.depth_bias);
    rs.depthBiasClamp = r.bias_clamp;
    rs.depthBiasSlopeFactor = r.slope_bias;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ms.alphaToCoverageEnable = st->alpha_to_coverage ? VK_TRUE : VK_FALSE;
    const auto& d = st->depth;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (a.depth != VK_FORMAT_UNDEFINED) {
        ds.depthTestEnable = d.depth_enable ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = d.depth_enable && d.depth_write ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = compare(d.depth_func);
        ds.stencilTestEnable = d.stencil_enable && a.stencil ? VK_TRUE : VK_FALSE;
        ds.front = stencil(d.front, d);
        ds.back = stencil(d.back, d);
    }
    VkPipelineColorBlendAttachmentState cb[8]{};
    for (uint32_t i = 0; i < a.count; ++i) {
        const auto& b = st->blend[i];
        cb[i].blendEnable = b.enable ? VK_TRUE : VK_FALSE;
        cb[i].srcColorBlendFactor = blend_factor(b.src);
        cb[i].dstColorBlendFactor = blend_factor(b.dst);
        cb[i].colorBlendOp = blend_op(b.op);
        cb[i].srcAlphaBlendFactor = blend_factor(b.src_alpha);
        cb[i].dstAlphaBlendFactor = blend_factor(b.dst_alpha);
        cb[i].alphaBlendOp = blend_op(b.op_alpha);
        cb[i].colorWriteMask = b.write_mask & 0xf;
    }
    VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    bs.attachmentCount = a.count;
    bs.pAttachments = cb;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                            VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
                            VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = sizeof dyn / sizeof dyn[0];
    dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = a.count;
    ri.pColorAttachmentFormats = a.color;
    ri.depthAttachmentFormat = a.depth;
    ri.stencilAttachmentFormat = a.stencil ? a.depth : VK_FORMAT_UNDEFINED;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &ri;
    ci.stageCount = nstages;
    ci.pStages = stages;
    ci.pVertexInputState = &vi;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &bs;
    ci.pDynamicState = &dy;
    ci.layout = g_layout;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult res = vk::vkCreateGraphicsPipelines(dev(), VK_NULL_HANDLE, 1, &ci, nullptr, &p);
    if (res != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 16) KL_WARN("gpu", "graphics pipeline not built by the driver: %s", vk::result_text(res));
        p = VK_NULL_HANDLE;
    }
    k->variants[key] = p;
    stats::add(stats::Pipelines, 0);
    return p;
}

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
    cmd::Targets targets{};
    bool rendering = false;
    Attachments att;
    VkExtent2D extent{0, 0};
    VkPipeline bound = VK_NULL_HANDLE;
    bool set_bound[2] = {false, false};

    Builder(HostQueue& h, Cb* cc) : hq(h), c(cc), cb(cc->cb) {}

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

    bool draw_setup(bool indexed) {
        pipeline::State* st = bind[1].pso;
        if (!st || st->compute) return false;
        Compiled* k = compile(st);
        if (!k->ok) {
            trace_draw("skipped (pipeline not translated)", st, nullptr, nullptr);
            return false;
        }
        if (topology_class(topology) == ~0u || topology_class(topology) == 3) {
            once("draws with topology %u are skipped (not supported yet)", topology);
            return false;
        }
        if (!rendering && !begin_rendering()) return false;
        VkPipeline p = variant(k, st, att, topology);
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
        vk::vkCmdSetPrimitiveTopology(cb, vk_topology(topology));
        bool strip = topology == 3 || topology == 5 || topology == 11 || topology == 13;
        vk::vkCmdSetPrimitiveRestartEnable(cb, indexed && strip && st->strip_cut ? VK_TRUE : VK_FALSE);
        vk::vkCmdSetBlendConstants(cb, blend);
        vk::vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref);
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

    void dispatch(const cmd::Dispatch& d) {
        pipeline::State* st = bind[0].pso;
        if (!st || !st->compute) return;
        Compiled* k = compile(st);
        if (!k->ok) return;
        end_rendering();
        if (k->compute != bound) vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->compute), bound = k->compute;
        if (!set_bound[0]) {
            VkDescriptorSet s = images::set();
            vk::vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_layout, 0, 1, &s, 0, nullptr);
            set_bound[0] = true;
        }
        uint64_t rec;
        uint32_t* r = record(c, RECORD_DWORDS, rec);
        if (!r) return;
        memset(r, 0, RECORD_DWORDS * 4);
        sgprs(*k->cs, bind[0], r);
        touch_images(*k->cs, bind[0]);
        push(rec);
        vk::vkCmdDispatch(cb, d.x, d.y, d.z);
        stats::add(stats::DispatchesRun);
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
        s.each([&](Op op, const uint8_t* p, uint32_t size) {
            auto get = [&](auto& v) { memcpy(&v, p, std::min<size_t>(sizeof v, size)); };
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
                        vk::vkCmdDraw(cb, d.count, d.instances, d.first, d.first_instance);
                        stats::add(stats::DrawsRun);
                    } else {
                        stats::add(stats::DrawsSkipped);
                    }
                    break;
                }
                case Op::DrawIndexed: {
                    cmd::DrawIndexed d;
                    get(d);
                    if (draw_setup(true)) {
                        vk::vkCmdDrawIndexed(cb, d.count, d.instances, d.first, d.base, d.first_instance);
                        stats::add(stats::DrawsRun);
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
                case Op::ExecuteIndirect: once("ExecuteIndirect is skipped (not supported yet)"); break;
                case Op::DispatchMesh: once("mesh shader dispatches are skipped (not supported yet)"); break;
                case Op::WaitValue: once("GPU waits on memory values are ignored"); break;
                default: break;
            }
        });
        end_rendering();
        full_barrier(cb);
    }
};

}  // namespace

bool start() {
    if (g_started) return true;
    auto& h = host::get();
    uint32_t fam[2] = {h.queues.graphics_family, h.queues.compute_family};
    VkQueue qs[2] = {h.queues.graphics, h.queues.compute};
    for (int i = 0; i < 2; ++i) {
        g_q[i].q = qs[i];
        g_q[i].family = fam[i];
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = fam[i];
        if (!KL_VK_CHECK(vk::vkCreateCommandPool(h.device, &pi, nullptr, &g_q[i].pool))) return false;
        VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        si.pNext = &ti;
        if (!KL_VK_CHECK(vk::vkCreateSemaphore(h.device, &si, nullptr, &g_q[i].timeline))) return false;
    }
    VkDescriptorSetLayout sl = images::set_layout();
    VkPushConstantRange pc{VK_SHADER_STAGE_ALL, 0, 32};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &sl;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pc;
    if (!KL_VK_CHECK(vk::vkCreatePipelineLayout(h.device, &li, nullptr, &g_layout))) return false;
    if (!memory::make_block(1 << 16, false, g_zero)) return false;
    memset(g_zero.map, 0, size_t(g_zero.size));
    g_started = true;
    KL_INFO("gpu", "executor ready: title command lists run on %s", h.name.c_str());
    return true;
}

int queue_for(uint32_t kind) { return kind == 2 ? COMPUTE : GRAPHICS; }
VkQueue vk_queue(int q) { return g_q[q & 1].q; }

// Texture uploads and first-use layouts for what was just recorded go in a submission of their
// own ahead of it; work waits for the other queue's latest uploads too (images are shared).
void prepare_and_submit(HostQueue& h, int qi, std::vector<Cb*> cbs) {
    Cb* pre = get_cb(h);
    if (pre && images::prepare(pre->cb, qi, h.submitted.load() + 1)) submit(h, {pre}, {}, {});
    else if (pre) cbs.insert(cbs.begin(), pre);  // empty: rides along
    std::vector<Semaphore> waits;
    int other = (qi & 1) ^ 1;
    uint64_t up = images::upload_value(other);
    if (up && g_q[other].timeline && up > completed(g_q[other])) waits.push_back({g_q[other].timeline, up});
    submit(h, cbs, waits, {});
}

void execute(int qi, uint32_t count, void* const* lists) {
    HostQueue& h = g_q[qi & 1];
    {
        std::lock_guard g(h.lock);
        h.recording++;
        std::vector<Cb*> cbs;
        for (uint32_t i = 0; i < count; ++i) {
            Stream* s = list::stream_of(lists[i]);
            if (!s) continue;
            Cb* c = get_cb(h);
            if (!c) continue;
            Builder b(h, c);
            b.run(*s);
            cbs.push_back(c);
        }
        if (!cbs.empty()) prepare_and_submit(h, qi, std::move(cbs));
        h.recording--;
    }
    collect();
}

void signal(int qi, VkSemaphore s, uint64_t v) {
    HostQueue& h = g_q[qi & 1];
    {
        std::lock_guard g(h.lock);
        submit(h, {}, {}, {{s, v}});
    }
    collect();
}

void wait(int qi, VkSemaphore s, uint64_t v) {
    HostQueue& h = g_q[qi & 1];
    std::lock_guard g(h.lock);
    submit(h, {}, {{s, v}}, {});
}

bool run(int qi, const std::function<void(VkCommandBuffer)>& f, VkSemaphore wait_bin, VkSemaphore signal_bin) {
    HostQueue& h = g_q[qi & 1];
    std::lock_guard g(h.lock);
    struct Recording {
        HostQueue& h;
        Recording(HostQueue& q) : h(q) { h.recording++; }
        ~Recording() { h.recording--; }
    } rec(h);
    std::vector<Cb*> cbs;
    Cb* pre = get_cb(h);
    if (pre && images::prepare(pre->cb, qi, h.submitted.load() + 1)) submit(h, {pre}, {}, {});
    else if (pre) cbs.push_back(pre);  // empty: rides along
    Cb* c = get_cb(h);
    if (!c) return false;
    cbs.push_back(c);
    f(c->cb);
    std::vector<Semaphore> w, s;
    if (wait_bin) w.push_back({wait_bin, 0});
    if (signal_bin) s.push_back({signal_bin, 0});
    int other = (qi & 1) ^ 1;
    uint64_t up = images::upload_value(other);
    if (up && g_q[other].timeline && up > completed(g_q[other])) w.push_back({g_q[other].timeline, up});
    submit(h, cbs, w, s);
    return true;
}

void with_queue(int qi, const std::function<void(VkQueue)>& f) {
    HostQueue& h = g_q[qi & 1];
    std::lock_guard g(h.lock);
    f(h.q);
}

}  // namespace exec

void retire(std::function<void()> f) {
    std::lock_guard g(exec::g_retire_lock);
    // +2: what is being recorded now may go as two submissions (its uploads, then its work).
    exec::g_retirees.push_back({{exec::g_q[0].submitted.load() + 2, exec::g_q[1].submitted.load() + 2}, std::move(f)});
}

}  // namespace kl::gpu
