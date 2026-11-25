// keylime GPU: what a title pipeline state object holds (its shaders and fixed-function
// state, in D3D12 terms), for the executor to build Vulkan pipelines from.
#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace kl::gpu::pipeline {

struct Shader {
    uint32_t stage;  // 0 VS 1 PS 2 DS 3 HS 4 GS 5 CS 6 AS 7 MS
    std::vector<uint8_t> code;
    uint64_t hash;
};

struct BlendTarget {
    uint32_t enable, logic_enable, src, dst, op, src_alpha, dst_alpha, op_alpha, logic_op, write_mask;
};
struct Raster {
    uint32_t fill, cull, front_ccw;
    int32_t depth_bias;
    float bias_clamp, slope_bias;
    uint32_t depth_clip, multisample, aa_lines, forced_samples, conservative;
};
struct StencilFace {
    uint32_t fail, depth_fail, pass, func;
};
struct DepthStencil {
    uint32_t depth_enable, depth_write, depth_func, stencil_enable;
    uint8_t read_mask, write_mask;
    StencilFace front, back;
    uint32_t bounds;
};
struct InputElement {
    std::string semantic;
    uint32_t index, format, slot, offset, per_instance, step;
};

struct State {
    bool compute = false;
    void* root = nullptr;
    std::vector<Shader> shaders;
    uint32_t alpha_to_coverage = 0, independent_blend = 0;
    BlendTarget blend[8]{};
    uint32_t sample_mask = ~0u;
    Raster raster{};
    DepthStencil depth{};
    std::vector<InputElement> input;
    uint32_t strip_cut = 0, topology_type = 3, rt_count = 0, rt_formats[8]{}, dsv_format = 0, samples = 1;
    // The executor's compiled forms (its own type), made on first use under `lock`.
    std::mutex lock;
    void* host = nullptr;
};

// The state of a title pipeline object; null if `obj` is not one.
State* state(void* obj);

}  // namespace kl::gpu::pipeline
