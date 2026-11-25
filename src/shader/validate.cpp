#include <chrono>
#include <cstdlib>
// keylime shader <dir> --spirv: translates every program to SPIR-V and builds a real
// Vulkan pipeline from it, so the driver's compiler validates the translator.
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "gpu/host.h"
#include "log.h"
#include "shader/container.h"
#include "shader/spirv.h"
#include "shader/translate.h"

namespace fs = std::filesystem;

namespace kl::shader {
namespace {

using namespace gpu;
using sp::Id;

struct Layout {
    VkDescriptorSetLayout set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

Layout make_layout(VkDevice d) {
    Layout l;
    VkDescriptorSetLayoutBinding binds[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 65536, VK_SHADER_STAGE_ALL, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, 4096, VK_SHADER_STAGE_ALL, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16384, VK_SHADER_STAGE_ALL, nullptr},
    };
    VkDescriptorBindingFlags flags[3] = {};
    for (auto& f : flags) f = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bf.bindingCount = 3;
    bf.pBindingFlags = flags;
    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.pNext = &bf;
    ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    ci.bindingCount = 3;
    ci.pBindings = binds;
    KL_VK_CHECK(vk::vkCreateDescriptorSetLayout(d, &ci, nullptr, &l.set));
    VkPushConstantRange pc{VK_SHADER_STAGE_ALL, 0, 32};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1;
    pl.pSetLayouts = &l.set;
    pl.pushConstantRangeCount = 1;
    pl.pPushConstantRanges = &pc;
    KL_VK_CHECK(vk::vkCreatePipelineLayout(d, &pl, nullptr, &l.layout));
    return l;
}

VkShaderModule module(VkDevice d, const std::vector<uint32_t>& code, VkResult& r) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size() * 4;
    ci.pCode = code.data();
    VkShaderModule m = VK_NULL_HANDLE;
    r = vk::vkCreateShaderModule(d, &ci, nullptr, &m);
    return m;
}

// A vertex shader writing position and a vec4 at each of `rows` (for testing pixel shaders).
std::vector<uint32_t> stub_vs(const std::vector<uint8_t>& rows) {
    sp::Builder b;
    b.capability(spv::CapabilityShader);
    Id t_void = b.t_void(), t_f32 = b.t_f32(), t_v4 = b.t_vec(t_f32, 4);
    Id fn_t = b.t_fn(t_void, {});
    Id p_out = b.t_ptr(spv::StorageClassOutput, t_v4);
    std::vector<Id> iface;
    Id pos = b.variable(p_out, spv::StorageClassOutput);
    b.decorate(pos, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInPosition)});
    iface.push_back(pos);
    std::vector<Id> outs;
    for (uint8_t r : rows) {
        Id v = b.variable(p_out, spv::StorageClassOutput);
        b.decorate(v, spv::DecorationLocation, {r});
        outs.push_back(v);
        iface.push_back(v);
    }
    Id zero = b.c_composite(t_v4, {b.c_f32(0), b.c_f32(0), b.c_f32(0), b.c_f32(1)});
    Id fn = b.id();
    b.begin_function(fn, t_void, fn_t);
    b.label();
    b.op_void(spv::OpStore, {pos, zero});
    for (Id o : outs) b.op_void(spv::OpStore, {o, zero});
    b.op_void(spv::OpReturn, {});
    b.end_function();
    b.entry(spv::ExecutionModelVertex, fn, "main", iface);
    return b.finish();
}

VkResult graphics(VkDevice d, VkPipelineLayout layout, VkShaderModule vs, VkShaderModule ps, const Translation& vt,
                  uint8_t mrt_mask, VkPipeline& out) {
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT, st[0].module = vs, st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT, st[1].module = ps, st[1].pName = "main";
    std::vector<VkVertexInputBindingDescription> vb;
    std::vector<VkVertexInputAttributeDescription> va;
    for (size_t k = 0; k < vt.vs_input_rows.size(); ++k) {
        vb.push_back({uint32_t(k), 16, VK_VERTEX_INPUT_RATE_VERTEX});
        va.push_back({vt.vs_input_rows[k], uint32_t(k), VK_FORMAT_R32G32B32A32_SFLOAT, 0});
    }
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = uint32_t(vb.size());
    vi.pVertexBindingDescriptions = vb.data();
    vi.vertexAttributeDescriptionCount = uint32_t(va.size());
    vi.pVertexAttributeDescriptions = va.data();
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    std::vector<VkFormat> formats;
    std::vector<VkPipelineColorBlendAttachmentState> blends;
    for (int k = 0; k < 8; ++k)
        if (mrt_mask & (1 << k)) {
            while (int(formats.size()) < k) formats.push_back(VK_FORMAT_UNDEFINED), blends.push_back({});
            formats.push_back(VK_FORMAT_R32G32B32A32_SFLOAT);
            VkPipelineColorBlendAttachmentState bs{};
            bs.colorWriteMask = 0xf;
            blends.push_back(bs);
        }
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = uint32_t(blends.size());
    cb.pAttachments = blends.data();
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = uint32_t(formats.size());
    ri.pColorAttachmentFormats = formats.data();
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &ri;
    gp.stageCount = ps ? 2 : 1;
    gp.pStages = st;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &ds;
    gp.layout = layout;
    return vk::vkCreateGraphicsPipelines(d, VK_NULL_HANDLE, 1, &gp, nullptr, &out);
}

}  // namespace

int validate(const fs::path& dir, bool driver) {
    VkDevice d = VK_NULL_HANDLE;
    Layout L;
    if (driver) {
        if (!host::start()) return 2;
        d = host::get().device;
        L = make_layout(d);
    }
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".bin") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    fs::create_directories(dir / "spv");
    std::map<std::string, size_t> unsupported, failures;
    size_t translated = 0, pipelines = 0, clean = 0, total = 0;
    double cs_seconds = 0;
    for (auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
        Container c;
        std::string err;
        if (!parse(bytes.data(), bytes.size(), c, err)) continue;
        for (auto& p : c.programs) {
            ++total;
            Translation t;
            if (!translate(p, t, err)) {
                ++failures["translate: " + err];
                continue;
            }
            ++translated;
            if (t.unsupported.empty()) ++clean;
            for (auto& [k, v] : t.unsupported) unsupported[k] += v;
            std::ofstream(dir / "spv" / (f.stem().string() + ".spv"), std::ios::binary)
                .write(reinterpret_cast<const char*>(t.spirv.data()), std::streamsize(t.spirv.size() * 4));
            if (!driver) continue;
            fprintf(stderr, "%ls %s\n", f.filename().c_str(), stage_name(p.stage));
            fflush(stderr);
            VkResult r;
            if (getenv("KEYLIME_NO_DRIVER_CACHE") && t.spirv.size() > 3) t.spirv[2] = uint32_t(GetTickCount64() ^ total);  // generator word: misses the driver's cache
            VkShaderModule m = module(d, t.spirv, r);
            if (r != VK_SUCCESS) {
                ++failures[std::string("module: ") + vk::result_text(r)];
                continue;
            }
            VkPipeline pipe = VK_NULL_HANDLE;
            if (p.stage == HwStage::Cs) {
                VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
                ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main"};
                ci.layout = L.layout;
                if (getenv("KEYLIME_FAST_PIPELINES")) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
                auto t0 = std::chrono::steady_clock::now();
                r = vk::vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe);
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                cs_seconds += sec;
                printf("  %.2f s (%zu KB code, %zu KB SPIR-V, %u blocks, %u by number)\n", sec, p.code.size() >> 10, (t.spirv.size() * 4) >> 10,
                       t.blocks, t.guarded_blocks);
            } else if (p.stage == HwStage::Ms) {
                vk::vkDestroyShaderModule(d, m, nullptr);
                continue;
            } else if (p.stage == HwStage::Vs) {
                r = graphics(d, L.layout, m, VK_NULL_HANDLE, t, 0, pipe);
            } else {
                VkResult r2;
                VkShaderModule vs = module(d, stub_vs(t.ps_input_rows), r2);
                Translation none;
                r = graphics(d, L.layout, vs, m, none, t.mrt_mask, pipe);
                vk::vkDestroyShaderModule(d, vs, nullptr);
            }
            if (r == VK_SUCCESS) {
                ++pipelines;
                vk::vkDestroyPipeline(d, pipe, nullptr);
            } else {
                ++failures[std::string(stage_name(p.stage)) + " pipeline: " + vk::result_text(r)];
                printf("pipeline failed: %ls (%s)\n", f.filename().c_str(), stage_name(p.stage));
            }
            vk::vkDestroyShaderModule(d, m, nullptr);
        }
    }
    printf("\n%zu programs: %zu translated (%zu with no unsupported instruction), %zu pipelines built by the driver, %.1f s in compute builds\n",
           total, translated, clean, pipelines, cs_seconds);
    for (auto& [k, v] : failures) printf("  failure x%zu: %s\n", v, k.c_str());
    std::vector<std::pair<size_t, std::string>> u;
    for (auto& [k, v] : unsupported) u.push_back({v, k});
    std::sort(u.rbegin(), u.rend());
    printf("unsupported instructions (%zu kinds):\n", u.size());
    for (auto& [v, k] : u) printf("  %6zu %s\n", v, k.c_str());
    return 0;
}

}  // namespace kl::shader
