#include "shader/indirect.h"

#include "shader/spirv.h"

namespace kl::shader {
using sp::Id;

std::vector<uint32_t> dispatch_clamp() {
    sp::Builder b;
    b.capability(spv::CapabilityShader);
    b.capability(spv::CapabilityPhysicalStorageBufferAddresses);
    b.capability(spv::CapabilityInt64);
    Id t_void = b.t_void(), t_bool = b.t_bool(), t_u32 = b.t_u32(), t_u64 = b.t_u64(), t_v3u = b.t_vec(t_u32, 3);
    Id t_fn = b.t_fn(t_void, {});
    Id st = b.t_struct({t_u64, t_u64, t_u64, t_u32, t_u32});
    b.decorate(st, spv::DecorationBlock);
    const uint32_t offsets[] = {0, 8, 16, 24, 28};
    for (uint32_t m = 0; m < 5; ++m) b.member_decorate(st, m, spv::DecorationOffset, {offsets[m]});
    Id push = b.variable(b.t_ptr(spv::StorageClassPushConstant, st), spv::StorageClassPushConstant);
    Id gid = b.variable(b.t_ptr(spv::StorageClassInput, t_v3u), spv::StorageClassInput);
    b.decorate(gid, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInGlobalInvocationId)});
    Id p_pc_u64 = b.t_ptr(spv::StorageClassPushConstant, t_u64), p_pc_u32 = b.t_ptr(spv::StorageClassPushConstant, t_u32);
    Id p_psb = b.t_ptr(spv::StorageClassPhysicalStorageBuffer, t_u32);
    Id fn = b.id();
    b.begin_function(fn, t_void, t_fn);
    b.label();
    auto member = [&](uint32_t m) {
        Id ptr = b.op(spv::OpAccessChain, m < 3 ? p_pc_u64 : p_pc_u32, {push, b.c_u32(m)});
        return b.op(spv::OpLoad, m < 3 ? t_u64 : t_u32, {ptr});
    };
    auto at = [&](Id base, Id off) {
        return b.op(spv::OpConvertUToPtr, p_psb, {b.op(spv::OpIAdd, t_u64, {base, b.op(spv::OpUConvert, t_u64, {off})})});
    };
    Id i = b.op(spv::OpCompositeExtract, t_u32, {b.op(spv::OpLoad, t_v3u, {gid}), 0});
    Id go = b.op(spv::OpULessThan, t_bool, {i, member(4)});
    Id yes = b.id(), after = b.id();
    b.op_void(spv::OpSelectionMerge, {after, 0});
    b.op_void(spv::OpBranchConditional, {go, yes, after});
    b.label(yes);
    Id count = b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb, {member(2)}), uint32_t(spv::MemoryAccessAlignedMask), 4});
    Id live = b.op(spv::OpULessThan, t_bool, {i, count});
    Id src = b.op(spv::OpIMul, t_u32, {i, member(3)}), dst = b.op(spv::OpIMul, t_u32, {i, b.c_u32(12)});
    for (uint32_t k = 0; k < 3; ++k) {
        Id v = b.op(spv::OpLoad, t_u32, {at(member(0), b.op(spv::OpIAdd, t_u32, {src, b.c_u32(4 * k)})), uint32_t(spv::MemoryAccessAlignedMask), 4});
        v = b.op(spv::OpSelect, t_u32, {live, v, b.c_u32(0)});
        b.op_void(spv::OpStore, {at(member(1), b.op(spv::OpIAdd, t_u32, {dst, b.c_u32(4 * k)})), v, uint32_t(spv::MemoryAccessAlignedMask), 4});
    }
    b.op_void(spv::OpBranch, {after});
    b.label(after);
    b.op_void(spv::OpReturn, {});
    b.end_function();
    b.entry(spv::ExecutionModelGLCompute, fn, "main", {push, gid});
    b.mode(fn, spv::ExecutionModeLocalSize, {64, 1, 1});
    return b.finish();
}

}  // namespace kl::shader
