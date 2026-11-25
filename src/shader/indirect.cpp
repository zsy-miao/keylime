#include "shader/indirect.h"

#include "shader/spirv.h"

namespace kl::shader {
using sp::Id;

std::vector<uint32_t> indirect_clamp(uint32_t type, uint32_t max_x) {
    const uint32_t words = type == 0 ? 4 : type == 1 ? 5 : 3;
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
    Id src = b.op(spv::OpIMul, t_u32, {i, member(3)}), dst = b.op(spv::OpIMul, t_u32, {i, b.c_u32(4 * words)});
    std::vector<Id> v;
    for (uint32_t k = 0; k < words; ++k)
        v.push_back(b.op(spv::OpLoad, t_u32, {at(member(0), b.op(spv::OpIAdd, t_u32, {src, b.c_u32(4 * k)})), uint32_t(spv::MemoryAccessAlignedMask), 4}));
    auto le = [&](Id x, uint32_t lim) { return b.op(spv::OpULessThanEqual, t_bool, {x, b.c_u32(lim)}); };
    auto both = [&](Id a, Id c) { return b.op(spv::OpLogicalAnd, t_bool, {a, c}); };
    auto wide = [&](Id x) { return b.op(spv::OpUConvert, t_u64, {x}); };
    Id ok, units;
    if (type >= 2) {
        Id groups = b.op(spv::OpIMul, t_u64, {b.op(spv::OpIMul, t_u64, {wide(v[0]), wide(v[1])}), wide(v[2])});
        ok = both(both(le(v[0], type == 2 ? max_x : 65535), le(v[1], 65535)), le(v[2], 65535));
        ok = both(ok, b.op(spv::OpULessThanEqual, t_bool, {groups, b.c_u64(type == 3 ? 1ull << 22 : 1ull << 24)}));
        units = b.op(spv::OpShiftRightLogical, t_u64, {b.op(spv::OpIAdd, t_u64, {groups, b.c_u64(511)}), b.c_u64(9)});
    } else {
        Id work = b.op(spv::OpIMul, t_u64, {wide(v[0]), wide(v[1])});
        ok = both(both(le(v[0], 1u << 24), le(v[1], 1u << 20)), b.op(spv::OpULessThanEqual, t_bool, {work, b.c_u64(1ull << 27)}));
        units = b.op(spv::OpShiftRightLogical, t_u64, {b.op(spv::OpIAdd, t_u64, {work, b.c_u64(4095)}), b.c_u64(12)});
    }
    live = both(live, ok);
    Id cost = b.op(spv::OpSelect, t_u32, {live, b.op(spv::OpUConvert, t_u32, {units}), b.c_u32(0)});
    Id spent_at = b.op(spv::OpIAdd, t_u32, {b.op(spv::OpIMul, t_u32, {member(4), b.c_u32(4 * words)}), b.c_u32(4)});
    Id before = b.op(spv::OpAtomicIAdd, t_u32, {at(member(1), spent_at), b.c_u32(1), b.c_u32(0), cost});
    Id after_spend = b.op(spv::OpIAdd, t_u64, {wide(before), wide(cost)});
    live = both(live, b.op(spv::OpULessThanEqual, t_bool, {after_spend, b.c_u64(1u << 16)}));
    for (uint32_t k = 0; k < words; ++k) {
        Id x = b.op(spv::OpSelect, t_u32, {live, v[k], b.c_u32(0)});
        b.op_void(spv::OpStore, {at(member(1), b.op(spv::OpIAdd, t_u32, {dst, b.c_u32(4 * k)})), x, uint32_t(spv::MemoryAccessAlignedMask), 4});
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
