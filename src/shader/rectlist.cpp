// keylime shaders: the console's rectangle lists (topology 17, hardware
// DI_PT_RECTLIST). Each rectangle comes as three vertices v0, v1, v2 with
// v0 the corner both edges share; the hardware makes the fourth as
// v1 + v2 - v0, for the position and every attribute. On the host this is a
// geometry shader taking each triangle and emitting a 4-vertex strip.
#include "shader/rectlist.h"

#include "shader/spirv.h"

namespace kl::shader {
using sp::Id;

// quad: quad lists (topology 19) arrive as 4-vertex primitives (lines with adjacency) and leave as v0 v1 v3 v2.
std::vector<uint32_t> rect_geometry(const std::vector<uint8_t>& locations, bool quad) {
    const uint32_t N = quad ? 4 : 3;
    sp::Builder b;
    b.capability(spv::CapabilityShader);
    b.capability(spv::CapabilityGeometry);
    b.capability(spv::CapabilityPhysicalStorageBufferAddresses);
    Id t_void = b.t_void(), t_f32 = b.t_f32(), t_v4 = b.t_vec(t_f32, 4), t_u32 = b.t_u32();
    Id t_fn = b.t_fn(t_void, {});
    // gl_in[3].gl_Position and gl_Position out, as blocks.
    Id in_block = b.t_struct({t_v4});
    b.decorate(in_block, spv::DecorationBlock);
    b.member_decorate(in_block, 0, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInPosition)});
    Id in_arr = b.t_array(in_block, N);
    Id in_pos = b.variable(b.t_ptr(spv::StorageClassInput, in_arr), spv::StorageClassInput);
    Id out_block = b.t_struct({t_v4});
    b.decorate(out_block, spv::DecorationBlock);
    b.member_decorate(out_block, 0, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInPosition)});
    Id out_pos = b.variable(b.t_ptr(spv::StorageClassOutput, out_block), spv::StorageClassOutput);
    std::vector<Id> interface = {in_pos, out_pos};
    Id p_in_v4 = b.t_ptr(spv::StorageClassInput, t_v4), p_out_v4 = b.t_ptr(spv::StorageClassOutput, t_v4);
    Id arr_v4 = b.t_array(t_v4, N);
    std::vector<std::pair<Id, Id>> vars;  // (in array, out)
    for (uint8_t loc : locations) {
        Id i = b.variable(b.t_ptr(spv::StorageClassInput, arr_v4), spv::StorageClassInput);
        Id o = b.variable(p_out_v4, spv::StorageClassOutput);
        b.decorate(i, spv::DecorationLocation, {loc});
        b.decorate(o, spv::DecorationLocation, {loc});
        interface.push_back(i), interface.push_back(o);
        vars.push_back({i, o});
    }
    Id fn = b.id();
    b.begin_function(fn, t_void, t_fn);
    b.label();
    auto at = [&](Id arr, uint32_t k, bool block) {
        if (block) return b.op(spv::OpAccessChain, p_in_v4, {arr, b.c_u32(k), b.c_u32(0)});
        return b.op(spv::OpAccessChain, p_in_v4, {arr, b.c_u32(k)});
    };
    auto load = [&](Id ptr) { return b.op(spv::OpLoad, t_v4, {ptr}); };
    Id pos[4], val[16][4];
    for (uint32_t k = 0; k < N; ++k) pos[k] = load(at(in_pos, k, true));
    for (size_t v = 0; v < vars.size() && v < 16; ++v)
        for (uint32_t k = 0; k < N; ++k) val[v][k] = load(at(vars[v].first, k, false));
    Id pos_out = b.op(spv::OpAccessChain, p_out_v4, {out_pos, b.c_u32(0)});
    auto emit = [&](Id p, const std::vector<Id>& vs) {
        b.op_void(spv::OpStore, {pos_out, p});
        for (size_t v = 0; v < vs.size(); ++v) b.op_void(spv::OpStore, {vars[v].second, vs[v]});
        b.op_void(spv::OpEmitVertex, {});
    };
    auto fourth = [&](Id a0, Id a1, Id a2) { return b.op(spv::OpFSub, t_v4, {b.op(spv::OpFAdd, t_v4, {a1, a2}), a0}); };
    size_t n = vars.size() < 16 ? vars.size() : 16;
    if (quad) {
        for (uint32_t k : {0u, 1u, 3u, 2u}) {
            std::vector<Id> vs;
            for (size_t v = 0; v < n; ++v) vs.push_back(val[v][k]);
            emit(pos[k], vs);
        }
    } else {
        for (uint32_t k = 0; k < 3; ++k) {
            std::vector<Id> vs;
            for (size_t v = 0; v < n; ++v) vs.push_back(val[v][k]);
            emit(pos[k], vs);
        }
        std::vector<Id> vs3;
        for (size_t v = 0; v < n; ++v) vs3.push_back(fourth(val[v][0], val[v][1], val[v][2]));
        emit(fourth(pos[0], pos[1], pos[2]), vs3);
    }
    b.op_void(spv::OpEndPrimitive, {});
    b.op_void(spv::OpReturn, {});
    b.end_function();
    b.entry(spv::ExecutionModelGeometry, fn, "main", interface);
    b.mode(fn, quad ? spv::ExecutionModeInputLinesAdjacency : spv::ExecutionModeTriangles);
    b.mode(fn, spv::ExecutionModeOutputTriangleStrip);
    b.mode(fn, spv::ExecutionModeOutputVertices, {4});
    b.mode(fn, spv::ExecutionModeInvocations, {1});
    (void)t_u32;
    return b.finish();
}

}  // namespace kl::shader
