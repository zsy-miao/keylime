// keylime shaders: GFX10 -> SPIR-V. See docs/translator.md and translate.h.
//
// Model (v1): one GFX10 lane per invocation, subgroups of 32 (NVIDIA). Wave-
// wide masks (EXEC, VCC, VOPC results) are subgroup ballots of the lanes in
// the subgroup; for wave64 programs the upper 32 bits stay zero (graphics
// lanes are fixed by the rasterizer). Control flow is structured: blocks in
// program order, loops only where the code branches backwards (see body()).
// Memory goes through the title page table.
#include "shader/translate.h"
#include "shader/structure.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <unordered_set>
#include <set>
#include <string_view>
#include <unordered_map>

#include "GLSL.std.450.h"
#include "shader/gfx10.h"
#include "shader/spirv.h"

namespace kl::shader {
namespace {

using sp::Id;
using gfx10::Enc;
using gfx10::Inst;
using gfx10::Operand;

constexpr uint32_t SGPRS = 128, VGPRS = 256;
constexpr uint16_t VCC_LO = 106, VCC_HI = 107, M0 = 124, SNULL = 125, EXEC_LO = 126, EXEC_HI = 127;

struct T {
    const Program& p;
    Translation& out;
    sp::Builder b;
    Id glsl = 0;
    // types
    Id t_void, t_bool, t_u32, t_i32, t_f32, t_u64, t_v4f, t_v4u, t_v2f, t_v4b, t_fn;
    Id p_fn_u32, p_fn_bool, p_psb_u32, p_psb_u64, p_pc_u32, p_pc_u64;
    // globals
    Id push = 0, lane_var = 0, frag_coord = 0, front_face = 0, vertex_index = 0, instance_index = 0, position_out = 0;
    Id local_id = 0, group_id = 0, depth_out = 0;
    Id local_index = 0, num_groups = 0, ms_count = 0, ms_pos_out = 0, ms_prim_out = 0, ms_cull_out = 0;
    Id ms_pos_[2] = {}, ms_prim_[2] = {}, ms_flags_[2] = {}, ms_half = 0, p_fn_v4f = 0;  // per half in wide mode
    std::map<uint8_t, Id> ms_outs, ms_vals_[2];
    Id lds = 0, lds_ptr = 0;
    std::vector<Id> interface;
    std::map<uint8_t, Id> vs_inputs, vs_outputs, ps_inputs, ps_outputs;  // by row / mrt
    Id sampled_images = 0, samplers = 0, storage_images = 0, p_uc_image = 0, p_uc_sampler = 0, t_image2d = 0, t_simage2d = 0, t_storage2d = 0;
    // registers (Function vars, made on first use); v[1] holds the upper half's lanes in wide mode
    Id s[SGPRS] = {}, v[2][VGPRS] = {}, scc = 0, pc = 0, lane = 0, lane_bit = 0;
    // Wide mode (wave64 compute): one invocation runs console lanes `lane` and `lane + 32`, so a
    // whole console wave is one 32-lane host subgroup and EXEC_HI, VCC_HI, v_readlane 32-63,
    // prefix sums across the halves and ds_bpermute are exact. Vector instructions are emitted
    // once per half (`h`), each with its own VGPRs and its own half of every lane mask.
    bool wide = false;
    int h = 0;
    Id sg_id = 0;
    std::unordered_map<Id, uint32_t> consts;
    std::vector<Inst> insts;
    std::map<uint32_t, uint32_t> block_of;
    // DPP / SDWA: source overrides and destination handling for the current instruction.
    Id ov[3] = {};
    Id dpp_write = 0;           // bool: this lane writes its destination
    const Inst* sdwa = nullptr;  // destination byte/word select  // byte offset -> block number
    std::string error;

    const Options* opt = nullptr;
    T(const Program& prog, Translation& o, const Options* op) : p(prog), out(o), opt(op) { wide = prog.wave64 && (prog.stage == HwStage::Cs || prog.stage == HwStage::Ms); }
    uint32_t ps_location(uint8_t row) const {
        if (opt) {
            auto it = opt->ps_location.find(row);
            if (it != opt->ps_location.end()) return it->second;
        }
        return row;
    }

    // ---------------------------------------------------------- helpers ----
    Id u(uint32_t x) {
        Id c = b.c_u32(x);
        consts[c] = x;
        return c;
    }
    Id shift_amount(Id n) {
        auto it = consts.find(n);
        return it != consts.end() ? u(it->second & 31) : band(n, u(31));
    }
    Id f(float x) { return b.c_f32(x); }
    // Register variables keep their last value per basic block: a load after a store or load of
    // the same register in the same block reuses the value (far less SPIR-V for the driver).
    struct Known { Id value; uint32_t block; };
    std::unordered_map<Id, Known> known;
    std::unordered_set<Id> registers;
    bool cache_loads = false;  // measured: reusing loaded values made driver compiles slightly slower
    Id load(Id var, Id type) {
        if (cache_loads && registers.count(var)) {
            auto it = known.find(var);
            if (it != known.end() && it->second.block == b.labels) return it->second.value;
            Id x = b.op(spv::OpLoad, type, {var});
            known[var] = {x, b.labels};
            return x;
        }
        return b.op(spv::OpLoad, type, {var});
    }
    void store(Id var, Id val) {
        b.op_void(spv::OpStore, {var, val});
        if (registers.count(var)) known[var] = {val, b.labels};
    }
    Id bc(Id type, Id x) { return b.op(spv::OpBitcast, type, {x}); }
    Id ext(uint32_t inst, Id type, std::initializer_list<Id> args) {
        std::vector<uint32_t> w = {glsl, inst};
        w.insert(w.end(), args);
        return b.op(spv::OpExtInst, type, w);
    }
    Id sel(Id type, Id c, Id a, Id z) { return b.op(spv::OpSelect, type, {c, a, z}); }
    Id iadd(Id a, Id z) { return b.op(spv::OpIAdd, t_u32, {a, z}); }
    Id isub(Id a, Id z) { return b.op(spv::OpISub, t_u32, {a, z}); }
    Id band(Id a, Id z) { return b.op(spv::OpBitwiseAnd, t_u32, {a, z}); }
    Id bor(Id a, Id z) { return b.op(spv::OpBitwiseOr, t_u32, {a, z}); }
    Id bxor(Id a, Id z) { return b.op(spv::OpBitwiseXor, t_u32, {a, z}); }
    Id bnot(Id a) { return b.op(spv::OpNot, t_u32, {a}); }
    Id shl(Id a, Id n) { return b.op(spv::OpShiftLeftLogical, t_u32, {a, shift_amount(n)}); }
    Id shr(Id a, Id n) { return b.op(spv::OpShiftRightLogical, t_u32, {a, shift_amount(n)}); }
    Id sar(Id a, Id n) { return b.op(spv::OpShiftRightArithmetic, t_u32, {a, shift_amount(n)}); }
    Id ne0(Id a) { return b.op(spv::OpINotEqual, t_bool, {a, u(0)}); }
    Id b2u(Id c) { return sel(t_u32, c, u(1), u(0)); }
    Id tof(Id x) { return bc(t_f32, x); }
    Id tou(Id x) { return bc(t_u32, x); }

    Id sreg(uint16_t i) {
        if (i >= SGPRS) i = SNULL;
        if (!s[i]) s[i] = b.local(p_fn_u32, u(0)), registers.insert(s[i]);
        return s[i];
    }
    Id vreg(uint16_t i) { return vreg_of(h, i); }
    Id vreg_of(int half, uint16_t i) {
        Id& r = v[half & 1][i & 0xff];
        if (!r) r = b.local(p_fn_u32, u(0)), registers.insert(r);
        return r;
    }
    // A lane-mask SGPR as this half sees it: the pair's upper dword for the upper half.
    Id mask_reg(uint16_t lo) { return get_s(wide && lo != SNULL ? uint16_t(lo + h) : lo); }
    Id mask_src(const Operand& o) {
        if (wide && o.valid && !o.vgpr() && (o.code <= 105 || o.code == VCC_LO || o.code == EXEC_LO)) return get_s(uint16_t(o.code + h));
        return raw(o);
    }
    // Runs `f` as if emitting for half `half` (wide mode reads of the other half's lanes).
    template <class F>
    auto as_half(int half, F f) {
        int was = h;
        h = half;
        auto r = f();
        h = was;
        return r;
    }
    Id get_s(uint16_t i) { return i == SNULL ? u(0) : load(sreg(i), t_u32); }
    void set_s(uint16_t i, Id x) {
        if (i == EXEC_LO || i == EXEC_HI) ++exec_gen;
        if (i != SNULL && i < SGPRS) store(sreg(i), x);
    }
    // Both dwords of a lane mask ORed: zero when no lane of the 64 is set (the upper one stays 0 for wave32).
    // (wave32 code may use vcc_hi as a plain SGPR, so only wave64 programs read the upper dword.)
    Id mask64(uint16_t lo) { return p.wave64 ? bor(get_s(lo), get_s(uint16_t(lo + 1))) : get_s(lo); }
    Id get_scc() { return load(scc, t_bool); }
    void set_scc(Id c) { store(scc, c); }
    Id get_v(uint16_t i) { return load(vreg(i), t_u32); }

    // A 9-bit source as u32 (no float modifiers applied).
    Id raw(const Operand& o) {
        uint16_t c = o.code;
        if (c >= 256) return get_v(uint16_t(c - 256));
        if (c <= 105 || c == VCC_LO || c == VCC_HI || c == M0 || c == EXEC_LO || c == EXEC_HI || (c >= 108 && c <= 123))
            return get_s(c);
        if (c == SNULL) return u(0);
        if (c >= 128 && c <= 192) return u(c - 128);
        if (c >= 193 && c <= 208) return u(uint32_t(-(int(c) - 192)));
        static const float fl[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.15915494f};
        if (c >= 240 && c <= 248) {
            uint32_t w;
            memcpy(&w, &fl[c - 240], 4);
            return u(w);
        }
        if (c == 255) return u(o.literal);
        if (c == 253) return b2u(get_scc());
        if (c == 251) return b2u(b.op(spv::OpIEqual, t_bool, {mask64(VCC_LO), u(0)}));  // vccz
        if (c == 252) return b2u(b.op(spv::OpIEqual, t_bool, {mask64(EXEC_LO), u(0)}));  // execz
        return u(0);
    }
    Id fmods(Id x, const Operand& o) {
        if (o.abs) x = ext(GLSLstd450FAbs, t_f32, {x});
        if (o.neg) x = b.op(spv::OpFNegate, t_f32, {x});
        return x;
    }
    Id fsrc(const Operand& o) {
        Id x = tof(raw(o));
        if (o.abs) x = ext(GLSLstd450FAbs, t_f32, {x});
        if (o.neg) x = b.op(spv::OpFNegate, t_f32, {x});
        return x;
    }
    // 64-bit scalar pair as u64.
    Id raw64(const Operand& o) {
        if (o.code <= 105 || o.code == VCC_LO || o.code == EXEC_LO || (o.code >= 108 && o.code <= 123)) {
            Id lo = b.op(spv::OpUConvert, t_u64, {get_s(o.code)});
            Id hi = b.op(spv::OpUConvert, t_u64, {get_s(uint16_t(o.code + 1))});
            return b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
        }
        Id x = b.op(spv::OpUConvert, t_u64, {raw(o)});
        if (o.code >= 193 && o.code <= 208) x = b.op(spv::OpSConvert, t_u64, {bc(t_i32, raw(o))});
        return x;
    }
    void set_s64(uint16_t i, Id x64) {
        set_s(i, b.op(spv::OpUConvert, t_u32, {x64}));
        set_s(uint16_t(i + 1), b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {x64, b.c_u32(32)})}));
    }

    // 64-bit vector operand (VGPR pair, SGPR pair or constant) as u64.
    Id vraw64(const Operand& o) {
        if (o.vgpr()) {
            Id lo = b.op(spv::OpUConvert, t_u64, {get_v(o.vgpr_index())});
            Id hi = b.op(spv::OpUConvert, t_u64, {get_v(uint16_t(o.vgpr_index() + 1))});
            return b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
        }
        return raw64(o);
    }
    void set_v64(uint16_t i, Id x64) {
        set_v(i, b.op(spv::OpUConvert, t_u32, {x64}));
        set_v(uint16_t(i + 1), b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {x64, b.c_u32(32)})}));
    }
    Id half(Id x, bool hi) {  // f16 half of a dword as f32
        Id h = hi ? shr(x, u(16)) : band(x, u(0xffff));
        return b.op(spv::OpCompositeExtract, t_f32, {ext(GLSLstd450UnpackHalf2x16, t_v2f, {h}), 0});
    }
    Id pack_half(Id lo, Id hi) { return ext(GLSLstd450PackHalf2x16, t_u32, {b.op(spv::OpCompositeConstruct, t_v2f, {lo, hi})}); }

    // Lane state.
    Id act_value[2] = {};
    uint32_t act_block[2] = {~0u, ~0u}, act_gen[2] = {~0u, ~0u}, exec_gen = 0;
    Id active() {
        if (act_gen[h] == exec_gen && act_block[h] == b.labels) return act_value[h];
        act_gen[h] = exec_gen, act_block[h] = b.labels;
        return act_value[h] = mask_bit(mask_reg(EXEC_LO));
    }
    void set_v(uint16_t i, Id x) {
        Id old = get_v(i);
        Id act = active();
        if (dpp_write) act = b.op(spv::OpLogicalAnd, t_bool, {act, dpp_write});
        if (sdwa && sdwa->sdwa_dst_sel != 6) {
            uint32_t s_ = sdwa->sdwa_dst_sel;
            uint32_t shift = s_ <= 3 ? 8 * s_ : 16 * (s_ - 4), width = s_ <= 3 ? 8 : 16;
            Id part = b.op(spv::OpBitFieldUExtract, t_u32, {x, u(0), u(width)});
            Id placed = shl(part, u(shift));
            Id mask = u(((width == 32 ? 0xffffffffu : ((1u << width) - 1)) << shift));
            Id rest = sdwa->sdwa_dst_unused == 2 ? band(old, bnot(mask)) : u(0);  // 2 = preserve, else zero
            if (sdwa->sdwa_dst_unused == 1 && shift + width < 32)  // sign-extend upward
                rest = band(bnot(mask), sel(t_u32, ne0(band(part, u(1u << (width - 1)))), u(~0u << (shift + width)), u(0)));
            x = bor(rest, placed);
        }
        store(vreg(i), sel(t_u32, act, x, old));
    }
    Id ballot(Id cond) {
        Id m = b.op(spv::OpGroupNonUniformBallot, t_v4u, {u(spv::ScopeSubgroup), cond});
        return b.op(spv::OpCompositeExtract, t_u32, {m, 0});
    }
    // A per-lane condition into a wave mask register (VOPC semantics: inactive lanes read 0).
    void set_mask(uint16_t sdst, Id cond) {
        Id m = ballot(b.op(spv::OpLogicalAnd, t_bool, {cond, active()}));
        if (wide) return set_s(sdst == SNULL ? sdst : uint16_t(sdst + h), m);
        set_s(sdst, m);
        if (sdst != SNULL && p.wave64) set_s(uint16_t(sdst + 1), u(0));
    }
    Id mask_bit(Id mask) { return ne0(band(mask, lane_bit)); }
    Id fclamp(Id x, bool clamp) { return clamp ? ext(GLSLstd450FClamp, t_f32, {x, f(0), f(1)}) : x; }
    Id omod(Id x, uint8_t m) {
        if (m == 1) return b.op(spv::OpFMul, t_f32, {x, f(2)});
        if (m == 2) return b.op(spv::OpFMul, t_f32, {x, f(4)});
        if (m == 3) return b.op(spv::OpFMul, t_f32, {x, f(0.5f)});
        return x;
    }

    // -------------------------------------------------------- memory ----
    // title address -> device address through the 2-level page table (1 GiB, 64 KiB pages).
    // The page table's address and the sink page's (loaded once, in the entry block).
    Id pt_base = 0, sink_base = 0, zero_base = 0;
    Id page_table() { return pt_base ? pt_base : load(b.op(spv::OpAccessChain, p_pc_u64, {push, u(1)}), t_u64); }
    void memory_bases() {
        pt_base = page_table();
        Id at = b.op(spv::OpIAdd, t_u64, {pt_base, b.c_u64((1ull << 17) * 8 + 16)});
        sink_base = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {at}), uint32_t(spv::MemoryAccessAlignedMask), 8});
        Id at2 = b.op(spv::OpIAdd, t_u64, {pt_base, b.c_u64((1ull << 17) * 8 + 24)});
        zero_base = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {at2}), uint32_t(spv::MemoryAccessAlignedMask), 8});
    }
    Id translate_addr(Id addr) { return dev_of(page_entry(addr), addr); }
    Id dev_of(Id entry, Id addr) {
        Id page = b.op(spv::OpBitwiseAnd, t_u64, {entry, b.c_u64(~(3ull << 62))});  // bits 63 guard, 62 next-page-follows
        return b.op(spv::OpIAdd, t_u64, {page, b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(0xffff)})});
    }
    // Entries of imported pages that no GPU resource or heap covers carry bit 63: stores there
    // are skipped, counted and the first one's address kept for the host's log.
    Id guarded(Id entry) { return b.op(spv::OpINotEqual, t_bool, {b.op(spv::OpShiftRightLogical, t_u64, {entry, b.c_u32(63)}), b.c_u64(0)}); }
    void report_guard(Id bad, Id addr) {
        Id yes = b.id(), after = b.id(), first = b.id(), done = b.id();
        b.op_void(spv::OpSelectionMerge, {after, 0});
        b.op_void(spv::OpBranchConditional, {bad, yes, after});
        b.label(yes);
        Id rec = b.op(spv::OpIAdd, t_u64, {page_table(), b.c_u64((1ull << 17) * 8)});
        auto at = [&](uint32_t k) { return b.op(spv::OpConvertUToPtr, p_psb_u32, {b.op(spv::OpIAdd, t_u64, {rec, b.c_u64(4 * k)})}); };
        Id old = b.op(spv::OpAtomicIAdd, t_u32, {at(0), u(spv::ScopeDevice), u(0), u(1)});
        Id is_first = b.op(spv::OpIEqual, t_bool, {old, u(0)});
        b.op_void(spv::OpSelectionMerge, {done, 0});
        b.op_void(spv::OpBranchConditional, {is_first, first, done});
        b.label(first);
        auto put = [&](uint32_t k, Id v) { b.op_void(spv::OpStore, {at(k), v, uint32_t(spv::MemoryAccessAlignedMask), 4}); };
        put(1, u(code_id(p)));
        put(2, b.op(spv::OpUConvert, t_u32, {addr}));
        put(3, b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {addr, b.c_u32(32)})}));
        b.op_void(spv::OpBranch, {done});
        b.label(done);
        b.op_void(spv::OpBranch, {after});
        b.label(after);
    }
    Id page_entry(Id addr) {
        addr = b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64((1ull << 47) - 1)});
        Id pt = page_table();
        Id i1 = b.op(spv::OpShiftRightLogical, t_u64, {addr, b.c_u32(30)});
        Id a1 = b.op(spv::OpIAdd, t_u64, {pt, b.op(spv::OpShiftLeftLogical, t_u64, {i1, b.c_u32(3)})});
        Id l2 = b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {a1}), uint32_t(spv::MemoryAccessAlignedMask), 8});
        Id i2 = b.op(spv::OpBitwiseAnd, t_u64, {b.op(spv::OpShiftRightLogical, t_u64, {addr, b.c_u32(16)}), b.c_u64(0x3fff)});
        Id a2 = b.op(spv::OpIAdd, t_u64, {l2, b.op(spv::OpShiftLeftLogical, t_u64, {i2, b.c_u32(3)})});
        return b.op(spv::OpLoad, t_u64, {b.op(spv::OpConvertUToPtr, p_psb_u64, {a2}), uint32_t(spv::MemoryAccessAlignedMask), 8});
    }
    // `count` consecutive dwords from `addr64`: one instruction touches at most two 64 KiB pages,
    // so two page-table walks serve all of them (the walks are half of a shader's driver
    // compile time when done per dword).
    // `count` consecutive dwords from `addr64` with one page walk, at contiguous device addresses
    // the driver can fetch as one load (per-dword page selects cost a third of a big shader's
    // compile time). Spanning into the next page is fine where the host marked it as following
    // on the device (bit 62, true inside every import); an access spanning two separate imports
    // reads zeros.
    std::vector<Id> mem_load_n(Id addr64, int count) {
        std::vector<Id> out;
        auto word = [&](Id dev) {
            return b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {dev}), uint32_t(spv::MemoryAccessAlignedMask), 4});
        };
        Id e0 = page_entry(addr64);
        Id dev0 = dev_of(e0, addr64);
        if (count == 1) return {word(dev0)};
        Id in_page = band(b.op(spv::OpUConvert, t_u32, {addr64}), u(0xffff));
        Id fits = b.op(spv::OpULessThanEqual, t_bool, {in_page, u(uint32_t(0x10000 - 4 * count))});
        Id next = b.op(spv::OpINotEqual, t_bool, {b.op(spv::OpBitwiseAnd, t_u64, {e0, b.c_u64(1ull << 62)}), b.c_u64(0)});
        Id base = b.op(spv::OpSelect, t_u64, {b.op(spv::OpLogicalOr, t_bool, {fits, next}), dev0, zero_base});
        for (int k = 0; k < count; ++k) out.push_back(word(b.op(spv::OpIAdd, t_u64, {base, b.c_u64(uint64_t(4 * k))})));
        return out;
    }
    Id mem_load(Id addr64) {
        Id dev = translate_addr(addr64);
        return b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {dev}), uint32_t(spv::MemoryAccessAlignedMask), 4});
    }
    // Branch-free: lanes that must not store (inactive, address 0, or a page no GPU resource or
    // heap covers) store to the sink page instead (a branch per store cost the driver much of
    // its compile time).
    void mem_store_n(Id addr64, Id go, const std::vector<Id>& vals) {
        int count = int(vals.size());
        go = b.op(spv::OpLogicalAnd, t_bool, {go, b.op(spv::OpINotEqual, t_bool, {addr64, b.c_u64(0)})});
        Id e0 = page_entry(addr64);
        Id ok = b.op(spv::OpLogicalAnd, t_bool, {go, b.op(spv::OpLogicalNot, t_bool, {guarded(e0)})});
        if (count > 1) {  // one walk, contiguous on the device (see mem_load_n); else the sink
            Id in_page = band(b.op(spv::OpUConvert, t_u32, {addr64}), u(0xffff));
            Id fits = b.op(spv::OpULessThanEqual, t_bool, {in_page, u(uint32_t(0x10000 - 4 * count))});
            Id next = b.op(spv::OpINotEqual, t_bool, {b.op(spv::OpBitwiseAnd, t_u64, {e0, b.c_u64(1ull << 62)}), b.c_u64(0)});
            ok = b.op(spv::OpLogicalAnd, t_bool, {ok, b.op(spv::OpLogicalOr, t_bool, {fits, next})});
        }
        Id base = b.op(spv::OpSelect, t_u64, {ok, dev_of(e0, addr64), sink_base});
        for (int k = 0; k < count; ++k) {
            Id dev = b.op(spv::OpIAdd, t_u64, {base, b.c_u64(uint64_t(4 * k))});
            b.op_void(spv::OpStore, {b.op(spv::OpConvertUToPtr, p_psb_u32, {dev}), vals[size_t(k)], uint32_t(spv::MemoryAccessAlignedMask), 4});
        }
    }
    void mem_store(Id addr64, Id val) { mem_store_n(addr64, b.c_bool(true), {val}); }
    Id add64(Id a64, Id off32) { return b.op(spv::OpIAdd, t_u64, {a64, b.op(spv::OpUConvert, t_u64, {off32})}); }
    Id add64s(Id a64, Id off32) { return b.op(spv::OpIAdd, t_u64, {a64, b.op(spv::OpSConvert, t_u64, {bc(t_i32, off32)})}); }
    // V# base (48 bits) from four SGPRs.
    Id vsharp_base(uint16_t s0) {
        Id lo = b.op(spv::OpUConvert, t_u64, {get_s(s0)});
        Id hi = b.op(spv::OpUConvert, t_u64, {band(get_s(uint16_t(s0 + 1)), u(0xffff))});
        return b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
    }

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
        set_s(EXEC_LO, ballot(b.c_bool(true)));
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

    // ------------------------------------------------------- the code ----
    bool decode_all() {
        auto code = p.patched_code();
        uint32_t at = 0;
        std::set<uint32_t> leaders = {0};
        bool ended = false;
        while (at < code.size()) {
            Inst i;
            if (!gfx10::decode(code.data(), code.size(), at, i)) {
                if (ended) break;  // data after the program's end
                char e[64];
                snprintf(e, sizeof e, "undecodable instruction at 0x%x", at);
                error = e;
                return false;
            }
            insts.push_back(i);
            uint32_t target;
            if (gfx10::branch_target(i, target)) leaders.insert(target), leaders.insert(at + i.size);
            if (gfx10::ends_program(i)) {
                ended = true;
                leaders.insert(at + i.size);
                // Code after the end may be data or unreachable padding; keep decoding while it decodes.
            }
            at += i.size;
        }
        uint32_t n = 0;
        for (uint32_t l : leaders) block_of[l] = n++;
        return true;
    }

    void unsupported(const Inst& i) { ++out.unsupported[i.name]; }

    // Writes a VALU result (VOP1/2/3), honouring clamp / omod for floats.
    void vres_f(const Inst& i, Id x) { set_v(i.dst.vgpr_index(), tou(fclamp(omod(x, i.omod), i.clamp))); }
    void vres_u(const Inst& i, Id x) { set_v(i.dst.vgpr_index(), x); }

    // v_cmp_class: whether the float's class is one the mask names (bit 0 signaling NaN, 1 quiet
    // NaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf).
    Id fclass(Id bits, Id mask, const Operand& o) {
        if (o.abs) bits = band(bits, u(0x7fffffff));
        if (o.neg) bits = bxor(bits, u(0x80000000));
        Id e = band(shr(bits, u(23)), u(0xff)), m = band(bits, u(0x7fffff));
        Id neg = ne0(shr(bits, u(31)));
        auto eq = [&](Id a, uint32_t k) { return b.op(spv::OpIEqual, t_bool, {a, u(k)}); };
        Id e_max = eq(e, 255), e_zero = eq(e, 0), m_zero = eq(m, 0);
        auto side = [&](uint32_t nk, uint32_t pk) { return sel(t_u32, neg, u(nk), u(pk)); };
        Id idx = sel(t_u32, e_max, sel(t_u32, m_zero, side(2, 9), sel(t_u32, ne0(band(m, u(0x400000))), u(1), u(0))),
                     sel(t_u32, e_zero, sel(t_u32, m_zero, side(5, 6), side(4, 7)), side(3, 8)));
        return ne0(band(shr(mask, idx), u(1)));
    }

    bool vcmp(const Inst& i, std::string_view n) {
        // v_cmp{x}_<op>_<type>
        bool x = n.rfind("v_cmpx_", 0) == 0;
        std::string_view rest = n.substr(x ? 7 : 6);
        size_t us = rest.rfind('_');
        if (us == std::string_view::npos) return false;
        std::string_view op = rest.substr(0, us), ty = rest.substr(us + 1);
        Id c = 0;
        if (ty == "f32") {
            Id a = fsrc(i.src[0]), z = fsrc(i.src[1]);
            static const std::pair<const char*, spv::Op> ops[] = {
                {"lt", spv::OpFOrdLessThan}, {"eq", spv::OpFOrdEqual}, {"le", spv::OpFOrdLessThanEqual},
                {"gt", spv::OpFOrdGreaterThan}, {"lg", spv::OpFOrdNotEqual}, {"ge", spv::OpFOrdGreaterThanEqual},
                {"o", spv::OpOrdered}, {"u", spv::OpUnordered}, {"nge", spv::OpFUnordLessThan}, {"nlg", spv::OpFUnordEqual},
                {"ngt", spv::OpFUnordLessThanEqual}, {"nle", spv::OpFUnordGreaterThan}, {"neq", spv::OpFUnordNotEqual},
                {"nlt", spv::OpFUnordGreaterThanEqual}};
            if (op == "f") c = b.c_bool(false);
            else if (op == "tru" || op == "t") c = b.c_bool(true);
            else
                for (auto& [nm, o] : ops)
                    if (op == nm) c = b.op(o, t_bool, {a, z});
            if (op == "class") c = fclass(raw(i.src[0]), raw(i.src[1]), i.src[0]);
        } else if (ty == "u32" || ty == "i32") {
            Id a = raw(i.src[0]), z = raw(i.src[1]);
            bool s = ty == "i32";
            if (s) a = bc(t_i32, a), z = bc(t_i32, z);
            if (op == "lt") c = b.op(s ? spv::OpSLessThan : spv::OpULessThan, t_bool, {a, z});
            else if (op == "le") c = b.op(s ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, {a, z});
            else if (op == "gt") c = b.op(s ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, {a, z});
            else if (op == "ge") c = b.op(s ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, {a, z});
            else if (op == "eq") c = b.op(spv::OpIEqual, t_bool, {a, z});
            else if (op == "ne" || op == "lg") c = b.op(spv::OpINotEqual, t_bool, {a, z});
            else if (op == "f") c = b.c_bool(false);
            else if (op == "t") c = b.c_bool(true);
        } else if (ty == "u64" || ty == "i64" || ty == "u16" || ty == "i16" || ty == "f16") {
            bool s = ty[0] == 'i';
            Id a, z, T;
            if (ty == "f16") {
                Id fa = half(raw(i.src[0]), i.opsel & 1), fz = half(raw(i.src[1]), i.opsel & 2);
                if (op == "lt") c = b.op(spv::OpFOrdLessThan, t_bool, {fa, fz});
                else if (op == "gt") c = b.op(spv::OpFOrdGreaterThan, t_bool, {fa, fz});
                else if (op == "le") c = b.op(spv::OpFOrdLessThanEqual, t_bool, {fa, fz});
                else if (op == "ge") c = b.op(spv::OpFOrdGreaterThanEqual, t_bool, {fa, fz});
                else if (op == "eq") c = b.op(spv::OpFOrdEqual, t_bool, {fa, fz});
                else if (op == "neq" || op == "lg") c = b.op(spv::OpFUnordNotEqual, t_bool, {fa, fz});
            } else {
                if (ty == "u64" || ty == "i64") {
                    a = vraw64(i.src[0]), z = vraw64(i.src[1]);
                    T = s ? b.t_int(64, true) : t_u64;
                    if (s) a = bc(T, a), z = bc(T, z);
                } else {
                    a = b.op(spv::OpBitFieldUExtract, t_u32, {raw(i.src[0]), u(0), u(16)});
                    z = b.op(spv::OpBitFieldUExtract, t_u32, {raw(i.src[1]), u(0), u(16)});
                    if (s) a = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, a), u(0), u(16)}), z = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, z), u(0), u(16)});
                }
                if (op == "lt") c = b.op(s ? spv::OpSLessThan : spv::OpULessThan, t_bool, {a, z});
                else if (op == "le") c = b.op(s ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, {a, z});
                else if (op == "gt") c = b.op(s ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, {a, z});
                else if (op == "ge") c = b.op(s ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, {a, z});
                else if (op == "eq") c = b.op(spv::OpIEqual, t_bool, {a, z});
                else if (op == "ne" || op == "lg") c = b.op(spv::OpINotEqual, t_bool, {a, z});
            }
        } else if (ty == "f64") {
            return false;
        }
        if (!c) return false;
        if (x) {
            Id m = ballot(b.op(spv::OpLogicalAnd, t_bool, {c, active()}));
            set_s(uint16_t(EXEC_LO + (wide ? h : 0)), m);
            if (i.enc == Enc::VOP3 && i.sdst.valid) set_s(uint16_t(i.sdst.code + (wide ? h : 0)), m);
        } else {
            uint16_t sd = i.enc == Enc::VOP3 ? i.sdst.code : VCC_LO;
            set_mask(sd, c);
        }
        return true;
    }

    // SDWA source selection: byte 0-3, word 0-1, dword; zero- or sign-extended.
    Id sdwa_src(Id x, uint32_t s_, bool sext) {
        uint32_t shift = s_ <= 3 ? 8 * s_ : 16 * (s_ - 4), width = s_ <= 3 ? 8 : 16;
        if (sext) return tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, x), u(shift), u(width)}));
        return b.op(spv::OpBitFieldUExtract, t_u32, {x, u(shift), u(width)});
    }
    // DPP: src0 read from another lane of its row (16 lanes); sets dpp_write for lanes that write.
    Id dpp_src(const Inst& i) {
        Id src = i.src[0].vgpr() ? get_v(i.src[0].vgpr_index()) : raw(i.src[0]);
        Id from, valid = b.c_bool(true);
        Id row_lane = band(lane, u(15)), row_base = band(lane, u(~15u));
        if (i.dpp8) {
            Id sh = b.op(spv::OpIMul, t_u32, {band(lane, u(7)), u(3)});
            from = bor(band(lane, u(~7u)), band(shr(u(i.dpp_sel8), sh), u(7)));
        } else {
            uint32_t c = i.dpp_ctrl;
            if (c <= 0xff) {
                Id sh = shl(band(lane, u(3)), u(1));
                from = bor(band(lane, u(~3u)), band(shr(u(c), sh), u(3)));
            } else if (c >= 0x101 && c <= 0x10f) {  // row_shl: lane i reads i + n
                uint32_t n = c & 0xf;
                from = iadd(lane, u(n));
                valid = b.op(spv::OpULessThanEqual, t_bool, {iadd(row_lane, u(n)), u(15)});
            } else if (c >= 0x111 && c <= 0x11f) {  // row_shr: lane i reads i - n
                uint32_t n = c & 0xf;
                from = isub(lane, u(n));
                valid = b.op(spv::OpUGreaterThanEqual, t_bool, {row_lane, u(n)});
            } else if (c >= 0x121 && c <= 0x12f) {  // row_ror
                uint32_t n = c & 0xf;
                from = bor(row_base, band(isub(iadd(row_lane, u(16)), u(n)), u(15)));
            } else if (c == 0x140) {
                from = bor(row_base, isub(u(15), row_lane));
            } else if (c == 0x141) {
                from = bor(band(lane, u(~7u)), isub(u(7), band(lane, u(7))));
            } else if (c >= 0x150 && c <= 0x15f) {  // row_share
                from = bor(row_base, u(c & 0xf));
            } else if (c >= 0x160 && c <= 0x16f) {  // row_xmask
                from = bor(row_base, bxor(row_lane, u(c & 0xf)));
            } else {
                from = lane;
            }
            // Row / bank masks: lanes of masked-off rows or banks keep their destination.
            if (i.dpp_row_mask != 0xf || i.dpp_bank_mask != 0xf) {
                Id row_on = ne0(band(shr(u(i.dpp_row_mask), iadd(shr(lane, u(4)), u(uint32_t(2 * h)))), u(1)));
                Id bank_on = ne0(band(shr(u(i.dpp_bank_mask), band(shr(lane, u(2)), u(3))), u(1)));
                dpp_write = b.op(spv::OpLogicalAnd, t_bool, {row_on, bank_on});
            }
        }
        Id moved = b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), src, band(from, u(31))});
        if (!i.dpp8) {
            // An invalid source reads 0 with bound_ctrl, else the lane keeps its destination.
            if (i.dpp_bound_ctrl) moved = sel(t_u32, valid, moved, u(0));
            else dpp_write = dpp_write ? b.op(spv::OpLogicalAnd, t_bool, {dpp_write, valid}) : valid;
        }
        return moved;
    }

    // One instruction inside a block. Returns false only on a hard error.
    void alu(const Inst& i) {
        std::string_view n = i.name;
        auto S0 = [&] { return ov[0] ? ov[0] : raw(i.src[0]); };
        auto S1 = [&] { return ov[1] ? ov[1] : raw(i.src[1]); };
        auto S2 = [&] { return ov[2] ? ov[2] : raw(i.src[2]); };
        auto F0 = [&] { return ov[0] ? fmods(tof(ov[0]), i.src[0]) : fsrc(i.src[0]); };
        auto F1 = [&] { return ov[1] ? fmods(tof(ov[1]), i.src[1]) : fsrc(i.src[1]); };
        auto F2 = [&] { return ov[2] ? fmods(tof(ov[2]), i.src[2]) : fsrc(i.src[2]); };
        auto fop = [&](spv::Op o) { vres_f(i, b.op(o, t_f32, {F0(), F1()})); };
        auto iop = [&](spv::Op o) { vres_u(i, b.op(o, t_u32, {S0(), S1()})); };
        uint16_t sd = i.dst.code;
        auto sres = [&](Id x) {  // SALU 32-bit result; SCC = result != 0 for logic ops
            set_s(sd, x);
        };
        auto slogic = [&](spv::Op o, bool b64, bool inv0 = false, bool inv1 = false, bool invr = false) {
            if (!b64) {
                Id a = S0(), z = S1();
                if (inv0) a = bnot(a);
                if (inv1) z = bnot(z);
                Id r = b.op(o, t_u32, {a, z});
                if (invr) r = bnot(r);
                sres(r);
                set_scc(ne0(r));
            } else {
                Id a = raw64(i.src[0]), z = raw64(i.src[1]);
                if (inv0) a = b.op(spv::OpNot, t_u64, {a});
                if (inv1) z = b.op(spv::OpNot, t_u64, {z});
                Id r = b.op(o, t_u64, {a, z});
                if (invr) r = b.op(spv::OpNot, t_u64, {r});
                set_s64(sd, r);
                set_scc(b.op(spv::OpINotEqual, t_bool, {r, b.c_u64(0)}));
            }
        };
        // ---- scalar ALU ----
        if (i.enc == Enc::SOP1 || i.enc == Enc::SOP2 || i.enc == Enc::SOPK || i.enc == Enc::SOPC) {
            if (n.rfind("s_waitcnt", 0) == 0 || n == "s_version" || n == "s_subvector_loop_begin" || n == "s_subvector_loop_end") return;
            if (n == "s_mov_b32") return sres(S0());
            if (n == "s_mov_b64") return set_s64(sd, raw64(i.src[0]));
            if (n == "s_movk_i32") return sres(u(uint32_t(int32_t(int16_t(i.simm16)))));
            if (n == "s_cmov_b32") return sres(sel(t_u32, get_scc(), S0(), get_s(sd)));
            if (n == "s_not_b32") { Id r = bnot(S0()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_and_b32") return slogic(spv::OpBitwiseAnd, false);
            if (n == "s_and_b64") return slogic(spv::OpBitwiseAnd, true);
            if (n == "s_or_b32") return slogic(spv::OpBitwiseOr, false);
            if (n == "s_or_b64") return slogic(spv::OpBitwiseOr, true);
            if (n == "s_xor_b32") return slogic(spv::OpBitwiseXor, false);
            if (n == "s_xor_b64") return slogic(spv::OpBitwiseXor, true);
            if (n == "s_andn2_b32") return slogic(spv::OpBitwiseAnd, false, false, true);
            if (n == "s_andn2_b64") return slogic(spv::OpBitwiseAnd, true, false, true);
            if (n == "s_orn2_b32") return slogic(spv::OpBitwiseOr, false, false, true);
            if (n == "s_orn2_b64") return slogic(spv::OpBitwiseOr, true, false, true);
            if (n == "s_nand_b32") return slogic(spv::OpBitwiseAnd, false, false, false, true);
            if (n == "s_nor_b32") return slogic(spv::OpBitwiseOr, false, false, false, true);
            if (n == "s_xnor_b32") return slogic(spv::OpBitwiseXor, false, false, false, true);
            if (n == "s_add_u32" || n == "s_add_i32") {
                Id a = S0(), z = S1(), r = iadd(a, z);
                sres(r);
                return set_scc(n == "s_add_u32" ? b.op(spv::OpULessThan, t_bool, {r, a}) : b.c_bool(false));
            }
            if (n == "s_addc_u32") {
                Id a = S0(), z = S1(), c = b2u(get_scc());
                Id r = iadd(iadd(a, z), c);
                sres(r);
                return set_scc(b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpULessThan, t_bool, {r, a}),
                                                               b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {r, a}), ne0(c)})}));
            }
            if (n == "s_sub_u32" || n == "s_sub_i32") {
                Id a = S0(), z = S1();
                sres(isub(a, z));
                return set_scc(n == "s_sub_u32" ? b.op(spv::OpULessThan, t_bool, {a, z}) : b.c_bool(false));
            }
            if (n == "s_subb_u32") return sres(isub(isub(S0(), S1()), b2u(get_scc())));
            if (n == "s_mul_i32") return sres(b.op(spv::OpIMul, t_u32, {S0(), S1()}));
            if (n == "s_addk_i32") return sres(iadd(get_s(sd), u(uint32_t(int32_t(int16_t(i.simm16))))));
            if (n == "s_mulk_i32") return sres(b.op(spv::OpIMul, t_u32, {get_s(sd), u(uint32_t(int32_t(int16_t(i.simm16))))}));
            if (n == "s_lshl_b32") { Id r = shl(S0(), S1()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_lshr_b32") { Id r = shr(S0(), S1()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_ashr_i32") { Id r = sar(S0(), S1()); sres(r); return set_scc(ne0(r)); }
            if (n == "s_lshl_b64") {
                Id r = b.op(spv::OpShiftLeftLogical, t_u64, {raw64(i.src[0]), band(S1(), u(63))});
                set_s64(sd, r);
                return set_scc(b.op(spv::OpINotEqual, t_bool, {r, b.c_u64(0)}));
            }
            if (n == "s_lshr_b64") {
                Id r = b.op(spv::OpShiftRightLogical, t_u64, {raw64(i.src[0]), band(S1(), u(63))});
                set_s64(sd, r);
                return set_scc(b.op(spv::OpINotEqual, t_bool, {r, b.c_u64(0)}));
            }
            if (n.rfind("s_lshl", 0) == 0 && n.find("_add_u32") != std::string_view::npos) {  // s_lshl1..4_add_u32
                Id r = iadd(shl(S0(), u(uint32_t(n[6] - '0'))), S1());
                sres(r);
                return set_scc(b.c_bool(false));
            }
            if (n == "s_min_u32" || n == "s_max_u32" || n == "s_min_i32" || n == "s_max_i32") {
                bool mn = n[3] == 'i', sg = n.back() == '2' && n[n.size() - 3] == 'i';
                uint32_t op = sg ? (mn ? GLSLstd450SMin : GLSLstd450SMax) : (mn ? GLSLstd450UMin : GLSLstd450UMax);
                Id r = ext(op, t_u32, {S0(), S1()});
                sres(r);
                return set_scc(b.op(spv::OpIEqual, t_bool, {r, S0()}));
            }
            if (n == "s_cselect_b32") return sres(sel(t_u32, get_scc(), S0(), S1()));
            if (n == "s_cselect_b64") return set_s64(sd, sel(t_u64, get_scc(), raw64(i.src[0]), raw64(i.src[1])));
            if (n == "s_bfe_u32" || n == "s_bfe_i32") {
                Id off = band(S1(), u(31)), w = band(shr(S1(), u(16)), u(0x7f));
                Id r = n == "s_bfe_u32" ? b.op(spv::OpBitFieldUExtract, t_u32, {S0(), off, w})
                                        : tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), off, w}));
                sres(r);
                return set_scc(ne0(r));
            }
            if (n == "s_bfm_b32") return sres(shl(isub(shl(u(1), S0()), u(1)), S1()));
            if (n == "s_brev_b32") return sres(b.op(spv::OpBitReverse, t_u32, {S0()}));
            if (n == "s_bcnt1_i32_b32") { Id r = b.op(spv::OpBitCount, t_u32, {S0()}); sres(r); return set_scc(ne0(r)); }
            if (n == "s_ff1_i32_b32") return sres(tou(ext(GLSLstd450FindILsb, t_i32, {S0()})));
            if (n == "s_ff1_i32_b64") {
                Id lo = get_s(i.src[0].code), hi = get_s(uint16_t(i.src[0].code + 1));
                Id lsb_lo = tou(ext(GLSLstd450FindILsb, t_i32, {lo})), lsb_hi = tou(ext(GLSLstd450FindILsb, t_i32, {hi}));
                return sres(sel(t_u32, ne0(lo), lsb_lo, sel(t_u32, ne0(hi), iadd(lsb_hi, u(32)), u(~0u))));
            }
            if (n == "s_flbit_i32_b32") {
                Id m = ext(GLSLstd450FindUMsb, t_i32, {S0()});
                return sres(sel(t_u32, b.op(spv::OpSLessThan, t_bool, {m, b.c_i32(0)}), u(~0u), isub(u(31), tou(m))));
            }
            if (n == "s_sext_i32_i16") return sres(tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(16)})));
            if (n == "s_sext_i32_i8") return sres(tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(8)})));
            if (n == "s_pack_ll_b32_b16") return sres(bor(band(S0(), u(0xffff)), shl(S1(), u(16))));
            if (n == "s_pack_lh_b32_b16") return sres(bor(band(S0(), u(0xffff)), band(S1(), u(0xffff0000))));
            if (n == "s_pack_hh_b32_b16") return sres(bor(shr(S0(), u(16)), band(S1(), u(0xffff0000))));
            if (n.find("_saveexec_b32") != std::string_view::npos || n.find("_saveexec_b64") != std::string_view::npos ||
                n.find("_wrexec_b32") != std::string_view::npos || n.find("_wrexec_b64") != std::string_view::npos) {
                bool wrexec = n.find("wrexec") != std::string_view::npos;
                // 64-bit forms on a wave64 program work on both dwords (EXEC_HI is real there).
                bool both = n.find("_b64") != std::string_view::npos && p.wave64;
                auto combine = [&](Id s0v, Id e) -> Id {
                    if (n.rfind("s_and_", 0) == 0) return band(s0v, e);
                    if (n.rfind("s_or_", 0) == 0) return bor(s0v, e);
                    if (n.rfind("s_xor_", 0) == 0) return bxor(s0v, e);
                    if (n.rfind("s_andn2_", 0) == 0) return band(s0v, bnot(e));
                    if (n.rfind("s_orn2_", 0) == 0) return bor(s0v, bnot(e));
                    if (n.rfind("s_andn1_", 0) == 0) return band(bnot(s0v), e);
                    if (n.rfind("s_nand_", 0) == 0) return bnot(band(s0v, e));
                    if (n.rfind("s_nor_", 0) == 0) return bnot(bor(s0v, e));
                    if (n.rfind("s_xnor_", 0) == 0) return bnot(bxor(s0v, e));
                    if (n.rfind("s_orn1_", 0) == 0) return bor(bnot(s0v), e);
                    return 0;
                };
                Id e_lo = get_s(EXEC_LO), e_hi = both ? get_s(EXEC_HI) : u(0);
                Id s_lo = S0(), s_hi = both ? (i.src[0].code <= 105 || i.src[0].code == VCC_LO || i.src[0].code == EXEC_LO
                                                   ? get_s(uint16_t(i.src[0].code + 1))
                                                   : (i.src[0].code >= 193 && i.src[0].code <= 208 ? u(~0u) : u(0)))
                                            : u(0);
                Id r_lo = combine(s_lo, e_lo);
                if (!r_lo) return unsupported(i);
                Id r_hi = both ? combine(s_hi, e_hi) : u(0);
                if (!wrexec) {
                    sres(e_lo);
                    if (n.find("_b64") != std::string_view::npos) set_s(uint16_t(sd + 1), e_hi);
                } else {
                    sres(r_lo);
                    if (both) set_s(uint16_t(sd + 1), r_hi);
                }
                set_s(EXEC_LO, r_lo);
                if (both) set_s(EXEC_HI, r_hi);
                return set_scc(ne0(bor(r_lo, r_hi)));
            }
            if (n == "s_wqm_b32" || n == "s_wqm_b64") {
                // Any lane of a quad -> the whole quad.
                auto wqm = [&](Id x) {
                    Id q = bor(bor(x, shr(x, u(1))), bor(shr(x, u(2)), shr(x, u(3))));
                    return b.op(spv::OpIMul, t_u32, {band(q, u(0x11111111)), u(0xf)});
                };
                Id q = wqm(S0()), qh = u(0);
                if (n == "s_wqm_b64" && p.wave64) qh = wqm(get_s(uint16_t(i.src[0].code + 1)));
                sres(q);
                if (n == "s_wqm_b64") set_s(uint16_t(sd + 1), qh);
                return set_scc(ne0(bor(q, qh)));
            }
            if (n == "s_getpc_b64") return set_s64(sd, b.c_u64(uint64_t(i.offset + 4)));
            if (n == "s_swappc_b64" && p.stage == HwStage::Vs) {
                // The fetch shader call that opens every vertex shader: attributes are already in
                // their VGPRs (prologue), so the call is a no-op; the return address is what it saves.
                return set_s64(sd, b.c_u64(uint64_t(i.offset + 4)));
            }
            if (n == "s_bcnt1_i32_b64") {
                Id r = iadd(b.op(spv::OpBitCount, t_u32, {get_s(i.src[0].code)}), b.op(spv::OpBitCount, t_u32, {get_s(uint16_t(i.src[0].code + 1))}));
                sres(r);
                return set_scc(ne0(r));
            }
            if (n == "s_bitset1_b32") return sres(bor(get_s(sd), shl(u(1), S0())));
            if (n == "s_bitset0_b32") return sres(band(get_s(sd), bnot(shl(u(1), S0()))));
            if (n == "s_mul_hi_u32") {
                Id m = b.op(spv::OpIMul, t_u64, {b.op(spv::OpUConvert, t_u64, {S0()}), b.op(spv::OpUConvert, t_u64, {S1()})});
                return sres(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {m, b.c_u32(32)})}));
            }
            if (n == "s_getreg_b32") return sres(u(0));
            if (n == "s_setreg_b32" || n == "s_setreg_imm32_b32") return;
            if (i.enc == Enc::SOPC || n.rfind("s_cmpk_", 0) == 0) {
                Id a = S0(), z = i.enc == Enc::SOPK ? u(uint32_t(uint16_t(i.simm16))) : S1();
                if (i.enc == Enc::SOPK) a = get_s(sd);
                std::string_view op = n.substr(n.rfind("s_cmp", 0) == 0 ? (n[5] == 'k' ? 7 : 6) : 0);
                bool s = op.find("_i32") != std::string_view::npos;
                if (s && i.enc == Enc::SOPK) z = u(uint32_t(int32_t(int16_t(i.simm16))));
                Id A = s ? bc(t_i32, a) : a, Z = s ? bc(t_i32, z) : z;
                Id c = 0;
                if (op.rfind("eq", 0) == 0) c = b.op(spv::OpIEqual, t_bool, {a, z});
                else if (op.rfind("lg", 0) == 0) c = b.op(spv::OpINotEqual, t_bool, {a, z});
                else if (op.rfind("gt", 0) == 0) c = b.op(s ? spv::OpSGreaterThan : spv::OpUGreaterThan, t_bool, {A, Z});
                else if (op.rfind("ge", 0) == 0) c = b.op(s ? spv::OpSGreaterThanEqual : spv::OpUGreaterThanEqual, t_bool, {A, Z});
                else if (op.rfind("lt", 0) == 0) c = b.op(s ? spv::OpSLessThan : spv::OpULessThan, t_bool, {A, Z});
                else if (op.rfind("le", 0) == 0) c = b.op(s ? spv::OpSLessThanEqual : spv::OpULessThanEqual, t_bool, {A, Z});
                else if (n == "s_bitcmp0_b32") c = b.op(spv::OpIEqual, t_bool, {band(shr(a, z), u(1)), u(0)});
                else if (n == "s_bitcmp1_b32") c = ne0(band(shr(a, z), u(1)));
                if (c) return set_scc(c);
            }
            return unsupported(i);
        }
        if (i.enc == Enc::SOPP) {
            if (n == "s_nop" || n == "s_waitcnt" || n == "s_inst_prefetch" || n == "s_sendmsg" || n == "s_setprio" ||
                n == "s_ttracedata" || n == "s_sleep" || n == "s_icache_inv" || n == "s_clause" || n == "s_round_mode" ||
                n == "s_denorm_mode" || n == "s_waitcnt_depctr" || n == "s_waitcnt_vscnt" || n == "s_waitcnt_vmcnt" ||
                n == "s_waitcnt_lgkmcnt" || n == "s_waitcnt_expcnt" || n == "s_code_end" || n == "s_sethalt" || n == "s_trap")
                return;
            if (n == "s_sendmsg" && p.stage == HwStage::Ms && (i.simm16 & 0xf) == 9) return store(ms_count, get_s(M0));
            if (n == "s_barrier") {
                if (p.stage == HwStage::Cs || p.stage == HwStage::Ms)
                    b.op_void(spv::OpControlBarrier, {u(spv::ScopeWorkgroup), u(spv::ScopeWorkgroup),
                                                      u(spv::MemorySemanticsWorkgroupMemoryMask | spv::MemorySemanticsAcquireReleaseMask)});
                return;
            }
            return unsupported(i);
        }
        // ---- scalar memory ----
        if (i.enc == Enc::SMEM) {
            int count = 1;
            if (n.find("x2") != std::string_view::npos) count = 2;
            else if (n.find("x4") != std::string_view::npos) count = 4;
            else if (n.find("x8") != std::string_view::npos) count = 8;
            else if (n.find("x16") != std::string_view::npos) count = 16;
            bool buffer = n.rfind("s_buffer_load", 0) == 0;
            if (n.rfind("s_load_dword", 0) == 0 || buffer) {
                Operand base;
                base.code = i.sbase;
                Id addr = buffer ? vsharp_base(i.sbase) : raw64(base);
                Id off = u(uint32_t(i.smem_offset));
                if (i.soffset != SNULL) off = iadd(off, get_s(i.soffset));
                addr = add64(addr, off);
                auto words = mem_load_n(addr, count);
                for (int k = 0; k < count; ++k) set_s(uint16_t(i.sdata + k), words[k]);
                return;
            }
            if (n.rfind("s_dcache", 0) == 0 || n == "s_gl1_inv") return;
            if (n == "s_memtime" || n == "s_memrealtime") return set_s64(i.sdata, b.c_u64(0));
            return unsupported(i);
        }
        // ---- vector ALU ----
        if (i.enc == Enc::VOPC || (i.enc == Enc::VOP3 && n.rfind("v_cmp", 0) == 0)) {
            if (!vcmp(i, n)) unsupported(i);
            return;
        }
        if (i.dpp || i.sdwa) {
            Inst plain = i;
            plain.dpp = plain.sdwa = false;
            if (i.dpp) {
                ov[0] = dpp_src(i);
            } else {
                for (int k = 0; k < 2; ++k)
                    if (i.src[k].valid && i.sdwa_src_sel[k] != 6) ov[k] = sdwa_src(raw(i.src[k]), i.sdwa_src_sel[k], i.sdwa_sext[k]);
                sdwa = &i;
            }
            alu(plain);
            ov[0] = ov[1] = ov[2] = 0;
            dpp_write = 0;
            sdwa = nullptr;
            return;
        }
        if (n == "v_nop") return;
        if (n == "v_mov_b32") return vres_u(i, S0());
        if (n == "v_add_f32") return fop(spv::OpFAdd);
        if (n == "v_sub_f32") return fop(spv::OpFSub);
        if (n == "v_subrev_f32") return vres_f(i, b.op(spv::OpFSub, t_f32, {F1(), F0()}));
        if (n == "v_mul_f32") return fop(spv::OpFMul);
        if (n == "v_mul_legacy_f32" || n == "v_mul_dx9_zero_f32") {
            Id a = F0(), z = F1();
            Id zero = b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpFOrdEqual, t_bool, {a, f(0)}), b.op(spv::OpFOrdEqual, t_bool, {z, f(0)})});
            return vres_f(i, sel(t_f32, zero, f(0), b.op(spv::OpFMul, t_f32, {a, z})));
        }
        if (n == "v_mac_f32" || n == "v_fmac_f32" || n == "v_fmac_legacy_f32" || n == "v_mac_legacy_f32") {
            Id acc = tof(get_v(i.dst.vgpr_index()));
            return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), F1(), acc}));
        }
        if (n == "v_mad_f32" || n == "v_fma_f32" || n == "v_mad_legacy_f32" || n == "v_fma_legacy_f32")
            return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), F1(), F2()}));
        if (n == "v_fmamk_f32" || n == "v_madmk_f32") return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), tof(u(i.src[2].literal)), F1()}));
        if (n == "v_fmaak_f32" || n == "v_madak_f32") return vres_f(i, ext(GLSLstd450Fma, t_f32, {F0(), F1(), tof(u(i.src[2].literal))}));
        if (n == "v_min_f32" || n == "v_min_legacy_f32") return vres_f(i, ext(GLSLstd450NMin, t_f32, {F0(), F1()}));
        if (n == "v_max_f32" || n == "v_max_legacy_f32") return vres_f(i, ext(GLSLstd450NMax, t_f32, {F0(), F1()}));
        if (n == "v_min3_f32") return vres_f(i, ext(GLSLstd450NMin, t_f32, {ext(GLSLstd450NMin, t_f32, {F0(), F1()}), F2()}));
        if (n == "v_max3_f32") return vres_f(i, ext(GLSLstd450NMax, t_f32, {ext(GLSLstd450NMax, t_f32, {F0(), F1()}), F2()}));
        if (n == "v_med3_f32") {
            Id a = F0(), z = F1(), c = F2();
            Id mx = ext(GLSLstd450NMax, t_f32, {ext(GLSLstd450NMin, t_f32, {a, z}), ext(GLSLstd450NMin, t_f32, {ext(GLSLstd450NMax, t_f32, {a, z}), c})});
            return vres_f(i, mx);
        }
        if (n == "v_rcp_f32" || n == "v_rcp_iflag_f32" || n == "v_rcp_legacy_f32") return vres_f(i, b.op(spv::OpFDiv, t_f32, {f(1), F0()}));
        if (n == "v_rsq_f32" || n == "v_rsq_legacy_f32") return vres_f(i, ext(GLSLstd450InverseSqrt, t_f32, {F0()}));
        if (n == "v_sqrt_f32") return vres_f(i, ext(GLSLstd450Sqrt, t_f32, {F0()}));
        if (n == "v_exp_f32" || n == "v_exp_legacy_f32") return vres_f(i, ext(GLSLstd450Exp2, t_f32, {F0()}));
        if (n == "v_log_f32" || n == "v_log_legacy_f32") return vres_f(i, ext(GLSLstd450Log2, t_f32, {F0()}));
        if (n == "v_sin_f32") return vres_f(i, ext(GLSLstd450Sin, t_f32, {b.op(spv::OpFMul, t_f32, {F0(), f(6.28318530718f)})}));
        if (n == "v_cos_f32") return vres_f(i, ext(GLSLstd450Cos, t_f32, {b.op(spv::OpFMul, t_f32, {F0(), f(6.28318530718f)})}));
        if (n == "v_fract_f32") return vres_f(i, ext(GLSLstd450Fract, t_f32, {F0()}));
        if (n == "v_floor_f32") return vres_f(i, ext(GLSLstd450Floor, t_f32, {F0()}));
        if (n == "v_ceil_f32") return vres_f(i, ext(GLSLstd450Ceil, t_f32, {F0()}));
        if (n == "v_trunc_f32") return vres_f(i, ext(GLSLstd450Trunc, t_f32, {F0()}));
        if (n == "v_rndne_f32") return vres_f(i, ext(GLSLstd450RoundEven, t_f32, {F0()}));
        if (n == "v_ldexp_f32") return vres_f(i, ext(GLSLstd450Ldexp, t_f32, {F0(), bc(t_i32, S1())}));
        if (n == "v_cvt_f32_u32") return vres_f(i, b.op(spv::OpConvertUToF, t_f32, {S0()}));
        if (n == "v_cvt_f32_i32") return vres_f(i, b.op(spv::OpConvertSToF, t_f32, {bc(t_i32, S0())}));
        if (n == "v_cvt_u32_f32") return vres_u(i, b.op(spv::OpConvertFToU, t_u32, {ext(GLSLstd450FClamp, t_f32, {F0(), f(0), f(4294967040.0f)})}));
        if (n == "v_cvt_i32_f32") return vres_u(i, tou(b.op(spv::OpConvertFToS, t_i32, {ext(GLSLstd450FClamp, t_f32, {F0(), f(-2147483648.0f), f(2147483520.0f)})})));
        if (n == "v_cvt_flr_i32_f32") return vres_u(i, tou(b.op(spv::OpConvertFToS, t_i32, {ext(GLSLstd450Floor, t_f32, {F0()})})));
        if (n == "v_cvt_rpi_i32_f32") return vres_u(i, tou(b.op(spv::OpConvertFToS, t_i32, {ext(GLSLstd450Floor, t_f32, {b.op(spv::OpFAdd, t_f32, {F0(), f(0.5f)})})})));
        if (n.rfind("v_cvt_f32_ubyte", 0) == 0) {
            uint32_t k = uint32_t(n.back() - '0');
            return vres_f(i, b.op(spv::OpConvertUToF, t_f32, {band(shr(S0(), u(8 * k)), u(0xff))}));
        }
        if (n == "v_cvt_f32_f16") {
            Id h = band(S0(), u(0xffff));
            if (i.opsel & 1) h = shr(S0(), u(16));
            Id v2 = ext(GLSLstd450UnpackHalf2x16, t_v2f, {h});
            return vres_f(i, b.op(spv::OpCompositeExtract, t_f32, {v2, 0}));
        }
        if (n == "v_cvt_f16_f32") {
            Id pk = ext(GLSLstd450PackHalf2x16, t_u32, {b.op(spv::OpCompositeConstruct, t_v2f, {F0(), f(0)})});
            Id old = get_v(i.dst.vgpr_index());
            Id r = (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(pk, u(16))) : bor(band(old, u(0xffff0000)), pk);
            return vres_u(i, r);
        }
        if (n == "v_cvt_pkrtz_f16_f32" || n == "v_cvt_pknorm_u16_f32" || n == "v_cvt_pknorm_i16_f32") {
            Id v2 = b.op(spv::OpCompositeConstruct, t_v2f, {F0(), F1()});
            uint32_t op = n == "v_cvt_pkrtz_f16_f32" ? GLSLstd450PackHalf2x16 : n == "v_cvt_pknorm_u16_f32" ? GLSLstd450PackUnorm2x16 : GLSLstd450PackSnorm2x16;
            return vres_u(i, ext(op, t_u32, {v2}));
        }
        if (n == "v_pack_b32_f16") return vres_u(i, bor(band(S0(), u(0xffff)), shl(S1(), u(16))));
        if (n == "v_cvt_pk_u16_u32") return vres_u(i, bor(band(S0(), u(0xffff)), shl(S1(), u(16))));
        if (n == "v_cndmask_b32") {
            Id mask = i.enc == Enc::VOP3 ? mask_src(i.src[2]) : mask_reg(VCC_LO);
            Id a = S0(), z = S1();
            if (i.enc == Enc::VOP3) {  // float modifiers apply to the selected values
                if (i.src[0].abs || i.src[0].neg) a = tou(F0());
                if (i.src[1].abs || i.src[1].neg) z = tou(F1());
            }
            return vres_u(i, sel(t_u32, mask_bit(mask), z, a));
        }
        if (n == "v_and_b32") return iop(spv::OpBitwiseAnd);
        if (n == "v_or_b32") return iop(spv::OpBitwiseOr);
        if (n == "v_xor_b32") return iop(spv::OpBitwiseXor);
        if (n == "v_not_b32") return vres_u(i, bnot(S0()));
        if (n == "v_xnor_b32") return vres_u(i, bnot(bxor(S0(), S1())));
        if (n == "v_lshlrev_b32") return vres_u(i, shl(S1(), S0()));
        if (n == "v_lshrrev_b32") return vres_u(i, shr(S1(), S0()));
        if (n == "v_ashrrev_i32") return vres_u(i, sar(S1(), S0()));
        if (n == "v_add_nc_u32" || n == "v_add_u32") return iop(spv::OpIAdd);
        if (n == "v_sub_nc_u32" || n == "v_sub_u32") return iop(spv::OpISub);
        if (n == "v_subrev_nc_u32" || n == "v_subrev_u32") return vres_u(i, isub(S1(), S0()));
        if (n == "v_mul_lo_u32" || n == "v_mul_lo_i32") return iop(spv::OpIMul);
        if (n == "v_mul_hi_u32" || n == "v_mul_hi_i32") {
            bool sg = n == "v_mul_hi_i32";
            Id t64 = sg ? b.t_int(64, true) : t_u64;
            Id a = sg ? b.op(spv::OpSConvert, t64, {bc(t_i32, S0())}) : b.op(spv::OpUConvert, t64, {S0()});
            Id z = sg ? b.op(spv::OpSConvert, t64, {bc(t_i32, S1())}) : b.op(spv::OpUConvert, t64, {S1()});
            Id m = b.op(spv::OpIMul, t64, {a, z});
            Id hi = b.op(sg ? spv::OpShiftRightArithmetic : spv::OpShiftRightLogical, t64, {m, b.c_u32(32)});
            return vres_u(i, sg ? tou(b.op(spv::OpSConvert, t_i32, {hi})) : b.op(spv::OpUConvert, t_u32, {hi}));
        }
        if (n == "v_mul_u32_u24") return vres_u(i, b.op(spv::OpIMul, t_u32, {band(S0(), u(0xffffff)), band(S1(), u(0xffffff))}));
        if (n == "v_mul_i32_i24") {
            Id a = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(24)});
            Id z = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S1()), u(0), u(24)});
            return vres_u(i, tou(b.op(spv::OpIMul, t_i32, {a, z})));
        }
        if (n == "v_mad_u32_u24") return vres_u(i, iadd(b.op(spv::OpIMul, t_u32, {band(S0(), u(0xffffff)), band(S1(), u(0xffffff))}), S2()));
        if (n == "v_mad_u32_u16") return vres_u(i, iadd(b.op(spv::OpIMul, t_u32, {band(S0(), u(0xffff)), band(S1(), u(0xffff))}), S2()));
        if (n == "v_bfe_u32") return vres_u(i, b.op(spv::OpBitFieldUExtract, t_u32, {S0(), band(S1(), u(31)), band(S2(), u(31))}));
        if (n == "v_bfe_i32") return vres_u(i, tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), band(S1(), u(31)), band(S2(), u(31))})));
        if (n == "v_bfi_b32") { Id m = S0(); return vres_u(i, bor(band(m, S1()), band(bnot(m), S2()))); }
        if (n == "v_bfm_b32") return vres_u(i, shl(isub(shl(u(1), S0()), u(1)), S1()));
        if (n == "v_lshl_add_u32") return vres_u(i, iadd(shl(S0(), S1()), S2()));
        if (n == "v_add_lshl_u32") return vres_u(i, shl(iadd(S0(), S1()), S2()));
        if (n == "v_lshl_or_b32") return vres_u(i, bor(shl(S0(), S1()), S2()));
        if (n == "v_and_or_b32") return vres_u(i, bor(band(S0(), S1()), S2()));
        if (n == "v_or3_b32") return vres_u(i, bor(bor(S0(), S1()), S2()));
        if (n == "v_xor3_b32") return vres_u(i, bxor(bxor(S0(), S1()), S2()));
        if (n == "v_add3_u32") return vres_u(i, iadd(iadd(S0(), S1()), S2()));
        if (n == "v_xad_u32") return vres_u(i, iadd(bxor(S0(), S1()), S2()));
        if (n == "v_alignbit_b32") {
            Id a = b.op(spv::OpUConvert, t_u64, {S0()}), z = b.op(spv::OpUConvert, t_u64, {S1()});
            Id cat = b.op(spv::OpBitwiseOr, t_u64, {b.op(spv::OpShiftLeftLogical, t_u64, {a, b.c_u32(32)}), z});
            return vres_u(i, b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {cat, band(S2(), u(31))})}));
        }
        if (n == "v_bcnt_u32_b32") return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {S0()}), S1()));
        if (n == "v_ffbl_b32") return vres_u(i, tou(ext(GLSLstd450FindILsb, t_i32, {S0()})));
        if (n == "v_ffbh_u32") {
            Id m = ext(GLSLstd450FindUMsb, t_i32, {S0()});
            return vres_u(i, sel(t_u32, b.op(spv::OpSLessThan, t_bool, {m, b.c_i32(0)}), u(~0u), isub(u(31), tou(m))));
        }
        if (n == "v_bfrev_b32") return vres_u(i, b.op(spv::OpBitReverse, t_u32, {S0()}));
        if (n == "v_min_u32") return vres_u(i, ext(GLSLstd450UMin, t_u32, {S0(), S1()}));
        if (n == "v_max_u32") return vres_u(i, ext(GLSLstd450UMax, t_u32, {S0(), S1()}));
        if (n == "v_min_i32") return vres_u(i, tou(ext(GLSLstd450SMin, t_i32, {bc(t_i32, S0()), bc(t_i32, S1())})));
        if (n == "v_max_i32") return vres_u(i, tou(ext(GLSLstd450SMax, t_i32, {bc(t_i32, S0()), bc(t_i32, S1())})));
        if (n == "v_med3_u32") {
            Id a = S0(), z = S1(), c = S2();
            return vres_u(i, ext(GLSLstd450UMax, t_u32, {ext(GLSLstd450UMin, t_u32, {a, z}), ext(GLSLstd450UMin, t_u32, {ext(GLSLstd450UMax, t_u32, {a, z}), c})}));
        }
        if (n == "v_med3_i32") {
            Id a = bc(t_i32, S0()), z = bc(t_i32, S1()), c = bc(t_i32, S2());
            Id t_i = t_i32;
            return vres_u(i, tou(ext(GLSLstd450SMax, t_i, {ext(GLSLstd450SMin, t_i, {a, z}), ext(GLSLstd450SMin, t_i, {ext(GLSLstd450SMax, t_i, {a, z}), c})})));
        }
        if (n == "v_add_co_u32" || n == "v_sub_co_u32" || n == "v_subrev_co_u32") {
            Id a = S0(), z = S1();
            if (n == "v_subrev_co_u32") std::swap(a, z);
            bool add = n == "v_add_co_u32";
            Id r = add ? iadd(a, z) : isub(a, z);
            Id carry = add ? b.op(spv::OpULessThan, t_bool, {r, a}) : b.op(spv::OpULessThan, t_bool, {a, z});
            vres_u(i, r);
            set_mask(i.enc == Enc::VOP3 ? i.sdst.code : VCC_LO, carry);
            return;
        }
        if (n == "v_add_co_ci_u32" || n == "v_sub_co_ci_u32" || n == "v_subrev_co_ci_u32") {
            Id a = S0(), z = S1();
            if (n == "v_subrev_co_ci_u32") std::swap(a, z);
            Id cin_mask = i.enc == Enc::VOP3 ? mask_src(i.src[2]) : mask_reg(VCC_LO);
            Id cin = b2u(mask_bit(cin_mask));
            bool add = n == "v_add_co_ci_u32";
            Id r = add ? iadd(iadd(a, z), cin) : isub(isub(a, z), cin);
            Id carry = add ? b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpULessThan, t_bool, {r, a}),
                                                            b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {r, a}), ne0(cin)})})
                           : b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpULessThan, t_bool, {a, z}),
                                                            b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {a, z}), ne0(cin)})});
            vres_u(i, r);
            set_mask(i.enc == Enc::VOP3 ? i.sdst.code : VCC_LO, carry);
            return;
        }
        // (Wide mode emits these three once: they read or write both halves themselves.)
        auto shuffle = [&](Id x, Id from) { return b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), x, from}); };
        if (n == "v_readfirstlane_b32") {
            // The first lane in EXEC (every invocation runs every block, so the subgroup's
            // first invocation is not the first active lane).
            auto first_of = [&](Id e) { return sel(t_u32, ne0(e), tou(ext(GLSLstd450FindILsb, t_i32, {e})), u(0)); };
            Id e = get_s(EXEC_LO);
            Id x = shuffle(as_half(0, S0), first_of(e));
            if (wide) x = sel(t_u32, ne0(e), x, shuffle(as_half(1, S0), first_of(get_s(EXEC_HI))));
            return set_s(i.dst.vgpr() ? i.dst.vgpr_index() : i.dst.code, x);  // dst is an SGPR (encoded in vdst)
        }
        if (n == "v_readlane_b32") {
            Id at = S1();
            Id x = shuffle(as_half(0, S0), band(at, u(31)));
            if (wide) x = sel(t_u32, ne0(band(at, u(32))), shuffle(as_half(1, S0), band(at, u(31))), x);
            uint16_t d = i.enc == Enc::VOP3 ? uint16_t(i.dst.code - 256) : i.dst.code;
            return set_s(d, x);
        }
        if (n == "v_writelane_b32") {
            Id at = S1(), val = S0();
            for (int hh = 0; hh < (wide ? 2 : 1); ++hh) {
                Id is = b.op(spv::OpIEqual, t_bool, {lane, band(at, u(31))});
                if (wide) is = b.op(spv::OpLogicalAnd, t_bool, {is, b.op(spv::OpIEqual, t_bool, {band(shr(at, u(5)), u(1)), u(uint32_t(hh))})});
                Id r = vreg_of(hh, i.dst.vgpr_index());
                store(r, sel(t_u32, is, val, load(r, t_u32)));
            }
            return;
        }
        if (n == "v_mbcnt_lo_u32_b32") {
            Id below = isub(shl(u(1), lane), u(1));
            if (wide && h == 1) return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {S0()}), S1()));  // every low lane is below
            return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {band(S0(), below)}), S1()));
        }
        if (n == "v_mbcnt_hi_u32_b32") {
            if (wide) {
                if (h == 0) return vres_u(i, S1());
                Id below = isub(shl(u(1), lane), u(1));
                return vres_u(i, iadd(b.op(spv::OpBitCount, t_u32, {band(S0(), below)}), S1()));
            }
            return vres_u(i, ms_half ? iadd(shl(ms_half, u(5)), S1()) : S1());
        }
        if (n == "v_permlanex16_b32" || n == "v_permlane16_b32") {
            // Lane select from two 32-bit selectors (4 bits per lane of the 16-lane row).
            Id sel_lo = S1(), sel_hi = S2();
            Id idx = band(lane, u(15));
            Id nib = sel(t_u32, b.op(spv::OpULessThan, t_bool, {idx, u(8)}), shr(sel_lo, shl(idx, u(2))), shr(sel_hi, shl(isub(idx, u(8)), u(2))));
            Id src_lane = band(nib, u(15));
            Id row = band(lane, u(16));
            if (n == "v_permlanex16_b32") row = bxor(row, u(16));
            Id from = bor(row, src_lane);
            return vres_u(i, b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), S0(), from}));
        }
        if (n == "v_interp_p1_f32" || n == "v_interp_p2_f32" || n == "v_interp_mov_f32") {
            auto it = ps_inputs.find(i.attr);
            if (it == ps_inputs.end()) return unsupported(i);
            Id val = load(it->second, t_v4f);
            return vres_u(i, tou(b.op(spv::OpCompositeExtract, t_f32, {val, i.attr_chan})));
        }
        if (n == "v_alignbyte_b32") {
            Id a = b.op(spv::OpUConvert, t_u64, {S0()}), z = b.op(spv::OpUConvert, t_u64, {S1()});
            Id cat = b.op(spv::OpBitwiseOr, t_u64, {b.op(spv::OpShiftLeftLogical, t_u64, {a, b.c_u32(32)}), z});
            return vres_u(i, b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {cat, shl(band(S2(), u(3)), u(3))})}));
        }
        if (n == "v_lshlrev_b64" || n == "v_lshrrev_b64" || n == "v_ashrrev_i64") {
            Id x = vraw64(i.src[1]), sh = band(S0(), u(63));
            Id r = n == "v_lshlrev_b64" ? b.op(spv::OpShiftLeftLogical, t_u64, {x, sh})
                 : n == "v_lshrrev_b64" ? b.op(spv::OpShiftRightLogical, t_u64, {x, sh})
                                        : b.op(spv::OpShiftRightArithmetic, t_u64, {x, sh});
            return set_v64(i.dst.vgpr_index(), r);
        }
        if (n == "v_mad_u64_u32") {
            Id m = b.op(spv::OpIMul, t_u64, {b.op(spv::OpUConvert, t_u64, {S0()}), b.op(spv::OpUConvert, t_u64, {S1()})});
            Id c = vraw64(i.src[2]);
            Id r = b.op(spv::OpIAdd, t_u64, {m, c});
            set_v64(i.dst.vgpr_index(), r);
            set_mask(i.sdst.valid ? i.sdst.code : SNULL, b.op(spv::OpULessThan, t_bool, {r, c}));
            return;
        }
        if (n == "v_mad_i32_i24") {
            Id a = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S0()), u(0), u(24)});
            Id z = b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S1()), u(0), u(24)});
            return vres_u(i, iadd(tou(b.op(spv::OpIMul, t_i32, {a, z})), S2()));
        }
        if (n == "v_min3_u32") return vres_u(i, ext(GLSLstd450UMin, t_u32, {ext(GLSLstd450UMin, t_u32, {S0(), S1()}), S2()}));
        if (n == "v_max3_u32") return vres_u(i, ext(GLSLstd450UMax, t_u32, {ext(GLSLstd450UMax, t_u32, {S0(), S1()}), S2()}));
        if (n == "v_perm_b32") {
            // Each result byte k picks byte sel_k of {S0, S1} (8 bytes), or a constant (sel >= 12).
            Id a = b.op(spv::OpUConvert, t_u64, {S0()}), z = b.op(spv::OpUConvert, t_u64, {S1()});
            Id cat = b.op(spv::OpBitwiseOr, t_u64, {b.op(spv::OpShiftLeftLogical, t_u64, {a, b.c_u32(32)}), z});
            Id selw = S2(), r = u(0);
            for (uint32_t k = 0; k < 4; ++k) {
                Id sk = band(shr(selw, u(8 * k)), u(0xff));
                Id byte = band(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpShiftRightLogical, t_u64, {cat, shl(band(sk, u(7)), u(3))})}), u(0xff));
                byte = sel(t_u32, b.op(spv::OpUGreaterThanEqual, t_bool, {sk, u(12)}), sel(t_u32, b.op(spv::OpIEqual, t_bool, {sk, u(12)}), u(0), u(0xff)), byte);
                r = bor(r, shl(byte, u(8 * k)));
            }
            return vres_u(i, r);
        }
        if (n == "v_min_f16" || n == "v_max_f16" || n == "v_min3_f16" || n == "v_max3_f16") {
            bool mn = n[2] == 'm' && n[3] == 'i';
            uint32_t op = mn ? GLSLstd450NMin : GLSLstd450NMax;
            Id a = half(S0(), i.opsel & 1), z = half(S1(), i.opsel & 2);
            Id r = ext(op, t_f32, {a, z});
            if (n.find("3_") != std::string_view::npos) r = ext(op, t_f32, {r, half(S2(), i.opsel & 4)});
            Id old = get_v(i.dst.vgpr_index());
            Id h = band(pack_half(r, f(0)), u(0xffff));
            return vres_u(i, (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(h, u(16))) : bor(band(old, u(0xffff0000)), h));
        }
        if (n == "v_pk_min_f16" || n == "v_pk_max_f16" || n == "v_pk_add_f16" || n == "v_pk_mul_f16" || n == "v_pk_fma_f16") {
            Id a = S0(), z = S1();
            auto lanes = [&](int k) {
                bool hi0 = k ? (i.opsel_hi & 1) : (i.opsel & 1), hi1 = k ? (i.opsel_hi & 2) : (i.opsel & 2);
                Id x = half(a, hi0), y = half(z, hi1);
                if (n == "v_pk_min_f16") return ext(GLSLstd450NMin, t_f32, {x, y});
                if (n == "v_pk_max_f16") return ext(GLSLstd450NMax, t_f32, {x, y});
                if (n == "v_pk_add_f16") return b.op(spv::OpFAdd, t_f32, {x, y});
                if (n == "v_pk_mul_f16") return b.op(spv::OpFMul, t_f32, {x, y});
                bool hi2 = k ? (i.opsel_hi & 4) : (i.opsel & 4);
                return ext(GLSLstd450Fma, t_f32, {x, y, half(S2(), hi2)});
            };
            return vres_u(i, pack_half(lanes(0), lanes(1)));
        }
        // ---- mixed precision: each source f32, or the low or high f16 of a dword (op_sel_hi says
        // f16, op_sel which half); neg_lo negates, neg_hi takes the absolute value ----
        if (n == "v_fma_mix_f32" || n == "v_fma_mixlo_f16" || n == "v_fma_mixhi_f16" || n == "v_mad_mix_f32" || n == "v_mad_mixlo_f16" || n == "v_mad_mixhi_f16") {
            static const float fl[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 0.15915494f};
            auto mix = [&](int k) {
                const Operand& o = i.src[k];
                Id x;
                if ((i.opsel_hi >> k) & 1) {
                    if (o.code >= 240 && o.code <= 248) x = f(fl[o.code - 240]);  // inline constants are the value itself
                    else x = half(ov[k] ? ov[k] : raw(o), (i.opsel >> k) & 1);
                } else {
                    x = tof(ov[k] ? ov[k] : raw(o));
                }
                if ((i.neg_hi >> k) & 1) x = ext(GLSLstd450FAbs, t_f32, {x});
                if (o.neg) x = b.op(spv::OpFNegate, t_f32, {x});
                return x;
            };
            Id r = ext(GLSLstd450Fma, t_f32, {mix(0), mix(1), mix(2)});
            if (i.clamp) r = ext(GLSLstd450FClamp, t_f32, {r, f(0), f(1)});
            if (n.find("mix_f32") != std::string_view::npos) return vres_u(i, tou(r));
            Id old = get_v(i.dst.vgpr_index());
            Id hv = band(pack_half(r, f(0)), u(0xffff));
            bool hi = n.find("mixhi") != std::string_view::npos;
            return vres_u(i, hi ? bor(band(old, u(0xffff)), shl(hv, u(16))) : bor(band(old, u(0xffff0000)), hv));
        }
        // ---- 16-bit float arithmetic: low (or op_sel) halves in, the low (or op_sel[3]) half out,
        // the other half kept ----
        if (n == "v_add_f16" || n == "v_sub_f16" || n == "v_subrev_f16" || n == "v_mul_f16" || n == "v_fma_f16" || n == "v_fmac_f16" ||
            n == "v_mad_f16" || n == "v_mac_f16" || n == "v_rcp_f16" || n == "v_sqrt_f16" || n == "v_rsq_f16" || n == "v_ldexp_f16" ||
            n == "v_floor_f16" || n == "v_ceil_f16" || n == "v_trunc_f16" || n == "v_fract_f16" || n == "v_rndne_f16" || n == "v_exp_f16" ||
            n == "v_log_f16" || n == "v_sin_f16" || n == "v_cos_f16") {
            auto hs = [&](int k) {
                Id x = half(ov[k] ? ov[k] : raw(i.src[k]), (i.opsel >> k) & 1);
                if (i.src[k].abs) x = ext(GLSLstd450FAbs, t_f32, {x});
                if (i.src[k].neg) x = b.op(spv::OpFNegate, t_f32, {x});
                return x;
            };
            Id old = get_v(i.dst.vgpr_index());
            bool dst_hi = (i.opsel & 8) != 0;
            Id r;
            if (n == "v_add_f16") r = b.op(spv::OpFAdd, t_f32, {hs(0), hs(1)});
            else if (n == "v_sub_f16") r = b.op(spv::OpFSub, t_f32, {hs(0), hs(1)});
            else if (n == "v_subrev_f16") r = b.op(spv::OpFSub, t_f32, {hs(1), hs(0)});
            else if (n == "v_mul_f16") r = b.op(spv::OpFMul, t_f32, {hs(0), hs(1)});
            else if (n == "v_fma_f16" || n == "v_mad_f16") r = ext(GLSLstd450Fma, t_f32, {hs(0), hs(1), hs(2)});
            else if (n == "v_fmac_f16" || n == "v_mac_f16") r = ext(GLSLstd450Fma, t_f32, {hs(0), hs(1), half(old, dst_hi)});
            else if (n == "v_rcp_f16") r = b.op(spv::OpFDiv, t_f32, {f(1), hs(0)});
            else if (n == "v_sqrt_f16") r = ext(GLSLstd450Sqrt, t_f32, {hs(0)});
            else if (n == "v_rsq_f16") r = ext(GLSLstd450InverseSqrt, t_f32, {hs(0)});
            else if (n == "v_ldexp_f16") r = ext(GLSLstd450Ldexp, t_f32, {hs(0), bc(t_i32, b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, S1()), u(0), u(16)}))});
            else if (n == "v_floor_f16") r = ext(GLSLstd450Floor, t_f32, {hs(0)});
            else if (n == "v_ceil_f16") r = ext(GLSLstd450Ceil, t_f32, {hs(0)});
            else if (n == "v_trunc_f16") r = ext(GLSLstd450Trunc, t_f32, {hs(0)});
            else if (n == "v_fract_f16") r = ext(GLSLstd450Fract, t_f32, {hs(0)});
            else if (n == "v_rndne_f16") r = ext(GLSLstd450RoundEven, t_f32, {hs(0)});
            else if (n == "v_exp_f16") r = ext(GLSLstd450Exp2, t_f32, {hs(0)});
            else if (n == "v_log_f16") r = ext(GLSLstd450Log2, t_f32, {hs(0)});
            else if (n == "v_sin_f16") r = ext(GLSLstd450Sin, t_f32, {b.op(spv::OpFMul, t_f32, {hs(0), f(6.2831853f)})});
            else r = ext(GLSLstd450Cos, t_f32, {b.op(spv::OpFMul, t_f32, {hs(0), f(6.2831853f)})});
            if (i.clamp) r = ext(GLSLstd450FClamp, t_f32, {r, f(0), f(1)});
            Id hv = band(pack_half(r, f(0)), u(0xffff));
            return vres_u(i, dst_hi ? bor(band(old, u(0xffff)), shl(hv, u(16))) : bor(band(old, u(0xffff0000)), hv));
        }
        if (n == "v_lshrrev_b16" || n == "v_lshlrev_b16" || n == "v_ashrrev_i16" || n == "v_add_u16" || n == "v_sub_u16" || n == "v_mul_lo_u16") {
            Id a = band(S0(), u(0xffff)), z = band(S1(), u(0xffff)), r;
            if (n == "v_lshrrev_b16") r = shr(z, band(a, u(15)));
            else if (n == "v_lshlrev_b16") r = shl(z, band(a, u(15)));
            else if (n == "v_ashrrev_i16") r = tou(b.op(spv::OpShiftRightArithmetic, t_i32, {b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, z), u(0), u(16)}), band(a, u(15))}));
            else if (n == "v_add_u16") r = iadd(a, z);
            else if (n == "v_sub_u16") r = isub(a, z);
            else r = b.op(spv::OpIMul, t_u32, {a, z});
            Id old = get_v(i.dst.vgpr_index());
            return vres_u(i, (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(r, u(16))) : bor(band(old, u(0xffff0000)), band(r, u(0xffff))));
        }
        // ---- cube map coordinates: face id, s and t on the face, twice the major axis ----
        if (n == "v_cubeid_f32" || n == "v_cubesc_f32" || n == "v_cubetc_f32" || n == "v_cubema_f32") {
            Id x = F0(), y = F1(), z = F2();
            auto fabs_ = [&](Id a) { return ext(GLSLstd450FAbs, t_f32, {a}); };
            auto neg = [&](Id a) { return b.op(spv::OpFNegate, t_f32, {a}); };
            auto lt0 = [&](Id a) { return b.op(spv::OpFOrdLessThan, t_bool, {a, f(0)}); };
            auto ge = [&](Id a, Id c) { return b.op(spv::OpFOrdGreaterThanEqual, t_bool, {a, c}); };
            Id ax = fabs_(x), ay = fabs_(y), az = fabs_(z);
            Id zmaj = b.op(spv::OpLogicalAnd, t_bool, {ge(az, ax), ge(az, ay)});
            Id ymaj = b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpLogicalNot, t_bool, {zmaj}), ge(ay, ax)});
            auto pick = [&](Id zv, Id yv, Id xv) { return sel(t_f32, zmaj, zv, sel(t_f32, ymaj, yv, xv)); };
            Id r;
            if (n == "v_cubeid_f32")
                r = pick(sel(t_f32, lt0(z), f(5), f(4)), sel(t_f32, lt0(y), f(3), f(2)), sel(t_f32, lt0(x), f(1), f(0)));
            else if (n == "v_cubesc_f32")
                r = pick(sel(t_f32, lt0(z), neg(x), x), x, sel(t_f32, lt0(x), z, neg(z)));
            else if (n == "v_cubetc_f32")
                r = pick(neg(y), sel(t_f32, lt0(y), neg(z), z), neg(y));
            else
                r = b.op(spv::OpFMul, t_f32, {f(2), pick(z, y, x)});
            return vres_f(i, r);
        }
        // ---- VGPR indexed by M0 ----
        if (n == "v_movrels_b32" || n == "v_movreld_b32" || n == "v_movrelsd_b32") {
            Id m0 = get_s(M0);
            uint32_t base_s = i.src[0].vgpr() ? i.src[0].vgpr_index() : 0, base_d = i.dst.vgpr_index();
            uint32_t limit = std::max<uint32_t>(p.vgprs, 1);
            if (n == "v_movreld_b32") {  // VGPR[dst + M0] = src0
                Id val = S0(), act = active();
                for (uint32_t k = 0; base_d + k < limit && base_d + k < VGPRS; ++k) {
                    Id r = vreg(uint16_t(base_d + k));
                    Id hit = b.op(spv::OpLogicalAnd, t_bool, {act, b.op(spv::OpIEqual, t_bool, {m0, u(k)})});
                    store(r, sel(t_u32, hit, val, load(r, t_u32)));
                }
                return;
            }
            Id x = u(0);  // VGPR[src0 + M0]
            for (uint32_t k = 0; base_s + k < limit && base_s + k < VGPRS; ++k)
                x = sel(t_u32, b.op(spv::OpIEqual, t_bool, {m0, u(k)}), get_v(uint16_t(base_s + k)), x);
            if (n == "v_movrels_b32") return vres_u(i, x);
            Id act = active();  // movrelsd: VGPR[dst + M0] = VGPR[src0 + M0]
            for (uint32_t k = 0; base_d + k < limit && base_d + k < VGPRS; ++k) {
                Id r = vreg(uint16_t(base_d + k));
                Id hit = b.op(spv::OpLogicalAnd, t_bool, {act, b.op(spv::OpIEqual, t_bool, {m0, u(k)})});
                store(r, sel(t_u32, hit, x, load(r, t_u32)));
            }
            return;
        }
        if (n == "v_swap_b32") {
            uint16_t a = i.dst.vgpr_index(), z = i.src[0].vgpr_index();
            Id va = get_v(a), vz = get_v(z), act = active();
            store(vreg(a), sel(t_u32, act, vz, va));
            store(vreg(z), sel(t_u32, act, va, vz));
            return;
        }
        if (n == "v_cvt_pk_u8_f32") {
            Id byte = b.op(spv::OpConvertFToU, t_u32, {ext(GLSLstd450FClamp, t_f32, {ext(GLSLstd450RoundEven, t_f32, {F0()}), f(0), f(255)})});
            Id sh = shl(band(S1(), u(3)), u(3));
            return vres_u(i, bor(band(S2(), bnot(shl(u(0xff), sh))), shl(byte, sh)));
        }
        // ---- 16-bit interpolation (attribute in src0: bits 0-5, channel in bits 6-7). The host
        // interpolates the attribute row as floats; p1ll is the first half of the pair. ----
        if (n == "v_interp_p1ll_f16" || n == "v_interp_p1lv_f16") return vres_u(i, u(0));
        if (n == "v_interp_p2_f16") {
            uint32_t attr = i.src[0].code & 63, chan = (i.src[0].code >> 6) & 3;
            auto it = ps_inputs.find(uint8_t(attr));
            if (it == ps_inputs.end()) return unsupported(i);
            // A compressed export (f16 pairs) arrives here unpacked into four floats: channel c's
            // low and high halves are the row's components 2c and 2c + 1.
            uint32_t comp = std::min<uint32_t>(2 * chan + (i.opsel & 1), 3);
            Id val = b.op(spv::OpCompositeExtract, t_f32, {load(it->second, t_v4f), comp});
            Id old = get_v(i.dst.vgpr_index());
            Id hv = band(pack_half(val, f(0)), u(0xffff));
            return vres_u(i, (i.opsel & 8) ? bor(band(old, u(0xffff)), shl(hv, u(16))) : bor(band(old, u(0xffff0000)), hv));
        }
        if (n == "v_cmp_class_f32") return unsupported(i);
        return unsupported(i);
    }

    // Vector memory.
    void vmem(const Inst& i) {
        std::string_view n = i.name;
        int count = 1;
        if (n.find("x2") != std::string_view::npos || n.find("_xy") != std::string_view::npos) count = 2;
        if (n.find("x3") != std::string_view::npos || n.find("_xyz") != std::string_view::npos) count = 3;
        if (n.find("x4") != std::string_view::npos || n.find("_xyzw") != std::string_view::npos) count = 4;
        if (i.enc == Enc::MUBUF || i.enc == Enc::MTBUF) {
            bool load_op = n.find("load") != std::string_view::npos, store_op = n.find("store") != std::string_view::npos;
            bool dword = n.find("dword") != std::string_view::npos || n.find("format") != std::string_view::npos;
            bool atomic = n.rfind("buffer_atomic_", 0) == 0;
            bool small = n.find("byte") != std::string_view::npos || n.find("short") != std::string_view::npos;
            if (!(load_op || store_op || atomic) || !(dword || small || atomic)) return unsupported(i);
            Id base = vsharp_base(i.srsrc);
            Id stride = band(shr(get_s(uint16_t(i.srsrc + 1)), u(16)), u(0x3fff));
            Id records = get_s(uint16_t(i.srsrc + 2));
            Id off = u(uint32_t(i.mem_offset));
            if (i.soffset != SNULL) off = iadd(off, raw(Operand{i.soffset, 0, false, false, true}));
            uint8_t va = i.vaddr;
            Id index = u(0);
            if (i.idxen) index = get_v(va++);
            if (i.offen) off = iadd(off, get_v(va));
            Id byte = iadd(b.op(spv::OpIMul, t_u32, {index, stride}), off);
            // Raw buffers (stride 0): bound by bytes; structured: by index.
            Id in_bounds = sel(t_bool, b.op(spv::OpIEqual, t_bool, {stride, u(0)}), b.op(spv::OpULessThan, t_bool, {off, records}),
                               b.op(spv::OpULessThan, t_bool, {index, records}));
            Id addr = add64(base, byte);
            if (atomic) return buffer_atomic(i, n, addr, in_bounds, i.vdata);
            if (small) {
                bool is_byte = n.find("byte") != std::string_view::npos, sext = n.find("sbyte") != std::string_view::npos || n.find("sshort") != std::string_view::npos;
                uint32_t width = is_byte ? 8 : 16;
                Id aligned = b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(~3ull)});
                Id shift = shl(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(3)})}), u(3));
                Id safe = b.op(spv::OpSelect, t_u64, {in_bounds, aligned, b.c_u64(0)});
                Id word = mem_load(safe);
                if (load_op) {
                    Id v_ = sext ? tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, word), shift, u(width)}))
                                 : b.op(spv::OpBitFieldUExtract, t_u32, {word, shift, u(width)});
                    set_v(i.vdata, sel(t_u32, in_bounds, v_, u(0)));
                } else {
                    Id merged = b.op(spv::OpBitFieldInsert, t_u32, {word, get_v(i.vdata), shift, u(width)});
                    Id wsafe = b.op(spv::OpSelect, t_u64, {b.op(spv::OpLogicalAnd, t_bool, {in_bounds, active()}), aligned, b.c_u64(0)});
                    mem_store(wsafe, merged);
                }
                return;
            }
            if (load_op) {
                // Out of bounds reads zero: guard by an address of our zero page (page table entry 0).
                auto words = mem_load_n(b.op(spv::OpSelect, t_u64, {in_bounds, addr, b.c_u64(0)}), count);
                for (int k = 0; k < count; ++k) set_v(uint16_t(i.vdata + k), sel(t_u32, in_bounds, words[k], u(0)));
                return;
            }
            std::vector<Id> vals;
            for (int k = 0; k < count; ++k) vals.push_back(get_v(uint16_t(i.vdata + k)));
            mem_store_n(addr, b.op(spv::OpLogicalAnd, t_bool, {in_bounds, active()}), vals);
            return;
        }
        if (i.enc == Enc::GLOBAL || i.enc == Enc::FLAT) {
            bool load_op = n.find("load") != std::string_view::npos, store_op = n.find("store") != std::string_view::npos;
            bool atomic = n.find("_atomic_") != std::string_view::npos;
            bool small = n.find("byte") != std::string_view::npos || n.find("short") != std::string_view::npos;
            if (!atomic && (!(load_op || store_op) || (n.find("dword") == std::string_view::npos && !small))) return unsupported(i);
            Id addr;
            if (i.enc == Enc::GLOBAL && i.saddr != 0x7d && i.saddr != 0x7f) {
                Operand sa;
                sa.code = i.saddr;
                addr = add64(raw64(sa), get_v(i.vaddr));
            } else {
                Id lo = b.op(spv::OpUConvert, t_u64, {get_v(i.vaddr)});
                Id hi = b.op(spv::OpUConvert, t_u64, {get_v(uint16_t(i.vaddr + 1))});
                addr = b.op(spv::OpBitwiseOr, t_u64, {lo, b.op(spv::OpShiftLeftLogical, t_u64, {hi, b.c_u32(32)})});
            }
            addr = add64s(addr, u(uint32_t(i.mem_offset)));
            if (atomic) return buffer_atomic(i, n, addr, b.c_bool(true), i.vdata);
            if (small) {  // 8 / 16 bits: through the aligned dword
                bool is_byte = n.find("byte") != std::string_view::npos;
                bool sext = n.find("sbyte") != std::string_view::npos || n.find("sshort") != std::string_view::npos;
                uint32_t width = is_byte ? 8 : 16;
                Id aligned = b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(~3ull)});
                Id shift = shl(b.op(spv::OpUConvert, t_u32, {b.op(spv::OpBitwiseAnd, t_u64, {addr, b.c_u64(3)})}), u(3));
                Id word = mem_load(aligned);
                if (load_op) {
                    Id v_ = sext ? tou(b.op(spv::OpBitFieldSExtract, t_i32, {bc(t_i32, word), shift, u(width)}))
                                 : b.op(spv::OpBitFieldUExtract, t_u32, {word, shift, u(width)});
                    return set_v(i.dst.vgpr_index(), v_);
                }
                Id merged = b.op(spv::OpBitFieldInsert, t_u32, {word, get_v(i.vdata), shift, u(width)});
                mem_store_n(aligned, active(), {merged});
                return;
            }
            uint16_t data = load_op ? i.dst.vgpr_index() : i.vdata;
            if (load_op) {
                auto words = mem_load_n(addr, count);
                for (int k = 0; k < count; ++k) set_v(uint16_t(data + k), words[k]);
                return;
            }
            std::vector<Id> vals;
            for (int k = 0; k < count; ++k) vals.push_back(get_v(uint16_t(data + k)));
            mem_store_n(addr, active(), vals);
            return;
        }
        if (i.enc == Enc::DS && lds) {
            const uint32_t last = (p.lds_bytes + 3) / 4 - 1;
            auto at = [&](Id byte) {
                return b.op(spv::OpAccessChain, lds_ptr, {lds, ext(GLSLstd450UMin, t_u32, {shr(byte, u(2)), u(last)})});
            };
            Id addr = iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff)));
            if (n == "ds_read_b32" || n == "ds_read_b64" || n == "ds_read_b128" || n == "ds_read_b96") {
                int c = n == "ds_read_b32" ? 1 : n == "ds_read_b64" ? 2 : n == "ds_read_b96" ? 3 : 4;
                for (int k = 0; k < c; ++k) set_v(uint16_t(i.dst.vgpr_index() + k), load(at(iadd(addr, u(4 * k))), t_u32));
                return;
            }
            if (n == "ds_write_b32" || n == "ds_write_b64" || n == "ds_write_b128" || n == "ds_write_b96") {
                int c = n == "ds_write_b32" ? 1 : n == "ds_write_b64" ? 2 : n == "ds_write_b96" ? 3 : 4;
                for (int k = 0; k < c; ++k) {
                    Id p = at(iadd(addr, u(4 * k)));
                    store(p, sel(t_u32, active(), get_v(uint16_t(i.data0 + k)), load(p, t_u32)));
                }
                return;
            }
            if (n == "ds_read2_b32" || n == "ds_write2_b32" || n == "ds_read2st64_b32" || n == "ds_write2st64_b32") {
                Id base = get_v(i.vaddr);
                uint32_t scale = n.find("st64") != std::string_view::npos ? 256 : 4;
                Id a0 = iadd(base, u(uint32_t(i.offset0) * scale)), a1 = iadd(base, u(uint32_t(i.offset1) * scale));
                if (n.rfind("ds_read2", 0) == 0) {
                    set_v(i.dst.vgpr_index(), load(at(a0), t_u32));
                    set_v(uint16_t(i.dst.vgpr_index() + 1), load(at(a1), t_u32));
                } else {
                    Id p0 = at(a0), p1 = at(a1);
                    store(p0, sel(t_u32, active(), get_v(i.data0), load(p0, t_u32)));
                    store(p1, sel(t_u32, active(), get_v(i.data1), load(p1, t_u32)));
                }
                return;
            }
        }
        if (i.enc == Enc::DS && lds && (n == "ds_write_b16" || n == "ds_write_b8")) {
            Id addr = iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff)));
            Id ptr = b.op(spv::OpAccessChain, lds_ptr, {lds, ext(GLSLstd450UMin, t_u32, {shr(addr, u(2)), u((p.lds_bytes + 3) / 4 - 1)})});
            Id shift = shl(band(addr, u(3)), u(3));
            Id merged = b.op(spv::OpBitFieldInsert, t_u32, {load(ptr, t_u32), get_v(i.data0), shift, u(n == "ds_write_b16" ? 16 : 8)});
            store(ptr, sel(t_u32, active(), merged, load(ptr, t_u32)));
            return;
        }
        // LDS atomics: ds_<op>[_rtn]_<u32|i32|b32>; inactive lanes leave memory alone.
        if (i.enc == Enc::DS && lds && n.size() > 7 && (n.substr(n.size() - 4) == "_u32" || n.substr(n.size() - 4) == "_i32" || n.substr(n.size() - 4) == "_b32")) {
            std::string_view body = n.substr(3, n.size() - 7);
            bool rtn = body.size() > 4 && body.substr(body.size() - 4) == "_rtn";
            if (rtn) body = body.substr(0, body.size() - 4);
            bool sg = n.substr(n.size() - 4) == "_i32";
            spv::Op op = body == "add" ? spv::OpAtomicIAdd : body == "sub" ? spv::OpAtomicISub : body == "or" ? spv::OpAtomicOr
                       : body == "and" ? spv::OpAtomicAnd : body == "xor" ? spv::OpAtomicXor
                       : body == "max" ? (sg ? spv::OpAtomicSMax : spv::OpAtomicUMax) : body == "min" ? (sg ? spv::OpAtomicSMin : spv::OpAtomicUMin)
                       : body == "wrxchg" ? spv::OpAtomicExchange : body == "inc" ? spv::OpAtomicIIncrement : spv::OpNop;
            if (op != spv::OpNop && op != spv::OpAtomicIIncrement) {
                Id addr = iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff)));
                Id ptr = b.op(spv::OpAccessChain, lds_ptr, {lds, ext(GLSLstd450UMin, t_u32, {shr(addr, u(2)), u((p.lds_bytes + 3) / 4 - 1)})});
                Id act = active(), val = get_v(i.data0), old;
                if (op == spv::OpAtomicExchange) {
                    Id res = b.local(p_fn_u32, u(0));
                    store(res, load(ptr, t_u32));
                    if_then(act, [&] { store(res, b.op(spv::OpAtomicExchange, t_u32, {ptr, u(spv::ScopeWorkgroup), u(0), val})); });
                    old = load(res, t_u32);
                } else {
                    Id neutral = body == "and" || (body == "min" && !sg) ? u(~0u) : body == "min" ? u(0x7fffffff) : body == "max" && sg ? u(0x80000000) : u(0);
                    old = b.op(op, t_u32, {ptr, u(spv::ScopeWorkgroup), u(0), sel(t_u32, act, val, neutral)});
                }
                if (rtn) set_v(i.dst.vgpr_index(), old);
                return;
            }
        }
        if (i.enc == Enc::DS && n == "ds_bpermute_b32") {
            auto shuf = [&](Id x, Id from) { return b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), x, from}); };
            if (!wide) {
                Id from = band(shr(iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff))), u(2)), u(31));
                return set_v(i.dst.vgpr_index(), shuf(get_v(i.data0), from));
            }
            // Wide (emitted once): every lane's result from both halves' sources, then the writes.
            Id src[2] = {as_half(0, [&] { return get_v(i.data0); }), as_half(1, [&] { return get_v(i.data0); })};
            Id res[2];
            for (int hh = 0; hh < 2; ++hh) {
                Id idx = as_half(hh, [&] { return band(shr(iadd(get_v(i.vaddr), u(uint32_t(i.mem_offset & 0xffff))), u(2)), u(63)); });
                Id from = band(idx, u(31));
                res[hh] = sel(t_u32, ne0(band(idx, u(32))), shuf(src[1], from), shuf(src[0], from));
            }
            for (int hh = 0; hh < 2; ++hh) as_half(hh, [&] { set_v(i.dst.vgpr_index(), res[hh]); return 0; });
            return;
        }
        if (i.enc == Enc::DS && n == "ds_swizzle_b32") {
            // offset bit 15: quad-perm mode (offset[7:0] = 4 x 2-bit lanes); else bitmask mode (and/or/xor on 5 bits).
            uint32_t off = uint32_t(i.mem_offset) & 0xffff;
            Id from;
            if (off & 0x8000) {
                Id q = band(lane, u(3));
                Id selbits = shr(u(off & 0xff), shl(q, u(1)));
                from = bor(band(lane, u(~3u)), band(selbits, u(3)));
            } else {
                uint32_t and_m = off & 0x1f, or_m = (off >> 5) & 0x1f, xor_m = (off >> 10) & 0x1f;
                from = bxor(bor(band(lane, u(and_m | ~31u)), u(or_m)), u(xor_m));
            }
            return set_v(i.dst.vgpr_index(), b.op(spv::OpGroupNonUniformShuffle, t_u32, {u(spv::ScopeSubgroup), get_v(i.data0), band(from, u(31))}));
        }
        if (i.enc == Enc::MIMG) return image(i);
        return unsupported(i);
    }

    // Atomics on title memory (buffer_atomic_* / global_atomic_*). Inactive lanes and
    // out-of-bounds ones operate on the zero page with a neutral value.
    void buffer_atomic(const Inst& i, std::string_view n, Id addr, Id in_bounds, uint16_t data) {
        Id go = b.op(spv::OpLogicalAnd, t_bool, {in_bounds, active()});
        Id a = b.op(spv::OpSelect, t_u64, {go, addr, b.c_u64(0)});
        Id bad = guarded(page_entry(a));
        Id dev = translate_addr(b.op(spv::OpSelect, t_u64, {bad, b.c_u64(0), a}));
        Id ptr = b.op(spv::OpConvertUToPtr, p_psb_u32, {dev});
        Id v_ = get_v(data);
        Id old;
        auto is = [&](const char* k) { return n.find(k) != std::string_view::npos; };
        if (is("cmpswap")) {
            // data = new value, data + 1 = the value memory must hold; returns the old value.
            Id cmp = get_v(uint16_t(data + 1));
            Id res = b.local(p_fn_u32, u(0));
            if_then(go, [&] { store(res, b.op(spv::OpAtomicCompareExchange, t_u32, {ptr, u(spv::ScopeDevice), u(0), u(0), v_, cmp})); });
            if (i.glc) set_v(data, load(res, t_u32));
            return;
        }
        if (is("fmin") || is("fmax")) {
            // Float min/max: compare-and-swap loop.
            bool mn = is("fmin");
            Id hdr = b.id(), body_ = b.id(), cont_ = b.id(), merge_ = b.id();
            Id prev_var = b.local(p_fn_u32, u(0)), tries_var = b.local(p_fn_u32, u(0));
            store(prev_var, b.op(spv::OpAtomicLoad, t_u32, {ptr, u(spv::ScopeDevice), u(0)}));
            store(tries_var, u(0));
            b.op_void(spv::OpBranch, {hdr});
            b.label(hdr);
            b.op_void(spv::OpLoopMerge, {merge_, cont_, 0});
            b.op_void(spv::OpBranch, {body_});
            b.label(body_);
            Id prev = load(prev_var, t_u32);
            Id want = tou(ext(mn ? GLSLstd450NMin : GLSLstd450NMax, t_f32, {tof(prev), tof(v_)}));
            want = sel(t_u32, go, want, prev);
            Id got = b.op(spv::OpAtomicCompareExchange, t_u32, {ptr, u(spv::ScopeDevice), u(0), u(0), want, prev});
            store(prev_var, got);
            Id tries = b.op(spv::OpIAdd, t_u32, {load(tries_var, t_u32), u(1)});
            store(tries_var, tries);
            b.op_void(spv::OpBranch, {cont_});
            b.label(cont_);
            Id done = b.op(spv::OpLogicalOr, t_bool, {b.op(spv::OpIEqual, t_bool, {got, prev}), b.op(spv::OpUGreaterThanEqual, t_bool, {tries, u(256)})});
            b.op_void(spv::OpBranchConditional, {done, merge_, hdr});
            b.label(merge_);
            old = load(prev_var, t_u32);
        } else {
            spv::Op op = is("_add") ? spv::OpAtomicIAdd : is("_sub") ? spv::OpAtomicISub : is("_or") ? spv::OpAtomicOr
                       : is("_and") ? spv::OpAtomicAnd : is("_xor") ? spv::OpAtomicXor : is("_smax") ? spv::OpAtomicSMax
                       : is("_smin") ? spv::OpAtomicSMin : is("_umax") ? spv::OpAtomicUMax : is("_umin") ? spv::OpAtomicUMin
                       : is("_swap") ? spv::OpAtomicExchange : spv::OpNop;
            if (op == spv::OpNop) return unsupported(i);
            Id neutral = is("_and") || is("_umin") ? u(~0u) : is("_smin") ? u(0x7fffffff) : is("_smax") ? u(0x80000000) : u(0);
            if (op == spv::OpAtomicExchange) {
                old = b.op(spv::OpAtomicLoad, t_u32, {ptr, u(spv::ScopeDevice), u(0)});
                Id yes = b.id(), after = b.id();
                b.op_void(spv::OpSelectionMerge, {after, 0});
                b.op_void(spv::OpBranchConditional, {go, yes, after});
                b.label(yes);
                b.op(spv::OpAtomicExchange, t_u32, {ptr, u(spv::ScopeDevice), u(0), v_});
                b.op_void(spv::OpBranch, {after});
                b.label(after);
            } else {
                old = b.op(op, t_u32, {ptr, u(spv::ScopeDevice), u(0), sel(t_u32, go, v_, neutral)});
            }
        }
        if (i.glc) set_v(data, old);  // glc: return the pre-op value
    }

    // Images: T# (8 SGPRs) / S# (4 SGPRs) -> bindless slots through the texture table.
    //   table: u32 mask, then entries of {u32 key_lo, u32 key_hi, u32 slot, u32 pad}
    // A loop that stops at the first hit (usually the first probe): the unrolled form cost the
    // driver seconds per shader.
    struct SlotMemo { Id lo, hi, slot; uint32_t block; };
    std::vector<SlotMemo> slot_memo;
    Id slot_of(Id key_lo, Id key_hi) {
        for (auto& m : slot_memo)
            if (m.lo == key_lo && m.hi == key_hi && m.block == b.labels) return m.slot;
        Id tt = load(b.op(spv::OpAccessChain, p_pc_u64, {push, u(2)}), t_u64);
        Id mask = b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {tt}), uint32_t(spv::MemoryAccessAlignedMask), 4});
        // texture_hash, in SPIR-V.
        Id hx = bxor(b.op(spv::OpIMul, t_u32, {key_lo, u(0x9e3779b1u)}), b.op(spv::OpIMul, t_u32, {key_hi, u(0x85ebca6bu)}));
        hx = b.op(spv::OpIMul, t_u32, {bxor(hx, shr(hx, u(16))), u(0x7feb352du)});
        Id h = band(bxor(hx, shr(hx, u(15))), mask);
        Id probe = b.local(p_fn_u32, u(0)), result = b.local(p_fn_u32, u(0));
        b.op_void(spv::OpStore, {probe, u(0)});
        b.op_void(spv::OpStore, {result, u(0)});
        Id header = b.id(), body = b.id(), cont = b.id(), merge = b.id();
        b.op_void(spv::OpBranch, {header});
        b.label(header);
        b.op_void(spv::OpLoopMerge, {merge, cont, 0});
        b.op_void(spv::OpBranch, {body});
        b.label(body);
        Id pv = b.op(spv::OpLoad, t_u32, {probe});
        Id idx = band(iadd(h, pv), mask);
        Id e = b.op(spv::OpIAdd, t_u64, {tt, b.op(spv::OpUConvert, t_u64, {iadd(u(16), shl(idx, u(4)))})});
        auto word = [&](uint32_t k) {
            return b.op(spv::OpLoad, t_u32, {b.op(spv::OpConvertUToPtr, p_psb_u32, {b.op(spv::OpIAdd, t_u64, {e, b.c_u64(4 * k)})}),
                                             uint32_t(spv::MemoryAccessAlignedMask), 4});
        };
        Id hit = b.op(spv::OpLogicalAnd, t_bool, {b.op(spv::OpIEqual, t_bool, {word(0), key_lo}), b.op(spv::OpIEqual, t_bool, {word(1), key_hi})});
        b.op_void(spv::OpStore, {result, sel(t_u32, hit, word(2), u(0))});
        Id last = b.op(spv::OpUGreaterThanEqual, t_bool, {pv, u(TEXTURE_TABLE_PROBES - 1)});
        b.op_void(spv::OpBranchConditional, {b.op(spv::OpLogicalOr, t_bool, {hit, last}), merge, cont});
        b.label(cont);
        b.op_void(spv::OpStore, {probe, iadd(pv, u(1))});
        b.op_void(spv::OpBranch, {header});
        b.label(merge);
        Id slot = b.op(spv::OpLoad, t_u32, {result});
        slot_memo.push_back({key_lo, key_hi, slot, b.labels});
        return slot;
    }
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

    // ------------------------------------------------------------ body ----
    // Control flow: `pc` holds the next block's number. Blocks are emitted in program order,
    // each run when pc names it, so forward branches need no loop. Backward branches make
    // loop regions (overlapping ones merged), each a structured loop that runs again while pc
    // stays inside it. Ending the program sets pc past the last block.
    struct Region {
        uint32_t lo, hi;
        bool active = false;
    };
    std::vector<Region> regions;
    std::vector<size_t> first_inst;
    Id trips = 0;
    uint32_t end_block = 0;

    uint32_t trip_limit() const { return p.stage == HwStage::Ps ? 512 : p.stage == HwStage::Vs || p.stage == HwStage::Ms ? 1024 : 8192; }

    void find_regions() {
        std::vector<std::pair<uint32_t, uint32_t>> r;
        for (auto& i : insts) {
            uint32_t target;
            if (!gfx10::branch_target(i, target)) continue;
            auto tb = block_of.find(target);
            if (tb == block_of.end()) continue;
            uint32_t src = std::prev(block_of.upper_bound(i.offset))->second;
            if (tb->second <= src) r.push_back({tb->second, src});
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t x = 0; x < r.size() && !changed; ++x)
                for (size_t y = 0; y < r.size() && !changed; ++y) {
                    if (x == y) continue;
                    auto& a = r[x];
                    auto& z = r[y];
                    bool same = a == z;
                    bool crossing = a.first < z.first && z.first <= a.second && a.second < z.second;
                    if (!same && !crossing) continue;
                    a = {std::min(a.first, z.first), std::max(a.second, z.second)};
                    r.erase(r.begin() + y);
                    changed = true;
                }
        }
        for (auto& [lo, hi] : r) regions.push_back({lo, hi});
    }

    Id pc_in(uint32_t lo, uint32_t hi) {
        return b.op(spv::OpULessThanEqual, t_bool, {isub(load(pc, t_u32), u(lo)), u(hi - lo)});
    }

    void block(uint32_t bi, uint32_t start) {
        size_t ii = first_inst[bi];
        for (; ii < insts.size(); ++ii) {
            const Inst& i = insts[ii];
            auto next_block = block_of.find(i.offset);
            if (i.offset != start && next_block != block_of.end()) {
                store(pc, u(next_block->second));
                return;
            }
            uint32_t target;
            if (gfx10::ends_program(i)) {
                store(pc, u(end_block));
                return;
            }
            if (gfx10::branch_target(i, target)) {
                auto tb = block_of.find(target);
                auto fb = block_of.find(i.offset + i.size);
                uint32_t tn = tb != block_of.end() ? tb->second : end_block, fn_ = fb != block_of.end() ? fb->second : end_block;
                std::string_view n = i.name;
                if (n == "s_branch") return store(pc, u(tn));
                Id c = 0;
                if (n == "s_cbranch_scc0") c = b.op(spv::OpLogicalNot, t_bool, {get_scc()});
                else if (n == "s_cbranch_scc1") c = get_scc();
                else if (n == "s_cbranch_vccz") c = b.op(spv::OpIEqual, t_bool, {mask64(VCC_LO), u(0)});
                else if (n == "s_cbranch_vccnz") c = ne0(mask64(VCC_LO));
                else if (n == "s_cbranch_execz") c = b.op(spv::OpIEqual, t_bool, {mask64(EXEC_LO), u(0)});
                else if (n == "s_cbranch_execnz") c = ne0(mask64(EXEC_LO));
                else {
                    unsupported(i);
                    c = b.c_bool(false);
                }
                return store(pc, sel(t_u32, c, u(tn), u(fn_)));
            }
            emit(i);
        }
        store(pc, u(end_block));
    }

    void if_then(Id cond, const std::function<void()>& f) {
        Id yes = b.id(), after = b.id();
        b.op_void(spv::OpSelectionMerge, {after, 0});
        b.op_void(spv::OpBranchConditional, {cond, yes, after});
        b.label(yes);
        f();
        b.op_void(spv::OpBranch, {after});
        b.label(after);
    }

    void loop(Region& r, const std::vector<uint32_t>& starts) {
        loop(r, [&] { range(r.lo, r.hi, starts, false); });
    }

    void loop(Region& r, const std::function<void()>& inner) {
        r.active = true;
        if_then(pc_in(r.lo, r.hi), [&] {
            Id header = b.id(), body_ = b.id(), cont = b.id(), merge = b.id();
            // Each loop counts its own trips: with one shared counter an inner loop reset the
            // outer one's on every pass, so an outer loop around an inner one had no cap at all.
            Id trips = b.local(p_fn_u32, u(0));
            store(trips, u(0));
            b.op_void(spv::OpBranch, {header});
            b.label(header);
            b.op_void(spv::OpLoopMerge, {merge, cont, 0});
            b.op_void(spv::OpBranch, {body_});
            b.label(body_);
            inner();
            b.op_void(spv::OpBranch, {cont});
            b.label(cont);
            Id t = iadd(load(trips, t_u32), u(1));
            store(trips, t);
            Id again = b.op(spv::OpLogicalAnd, t_bool, {pc_in(r.lo, r.hi), b.op(spv::OpULessThan, t_bool, {t, u(trip_limit())})});
            b.op_void(spv::OpBranchConditional, {again, header, merge});
            b.label(merge);
            // A loop stopped by its cap ends the program (as VIctory's forks do): carrying on let an
            // outer loop enter it again with a fresh count, so nested caps multiplied (8192 x 8192
            // trips) and a loop spinning on bad data held the GPU until Windows reset it.
            Id capped = b.op(spv::OpUGreaterThanEqual, t_bool, {load(trips, t_u32), u(trip_limit())});
            store(pc, sel(t_u32, capped, u(end_block), load(pc, t_u32)));
        });
        r.active = false;
    }

    std::vector<cfg::Term> terms() const {
        std::vector<cfg::Term> out(end_block);
        for (uint32_t bi = 0; bi < end_block; ++bi) {
            cfg::Term& t = out[bi];
            t.kind = cfg::Exit::End;
            for (size_t ii = first_inst[bi]; ii < insts.size(); ++ii) {
                const Inst& i = insts[ii];
                if (ii != first_inst[bi] && block_of.count(i.offset)) {
                    t.kind = cfg::Exit::Next;
                    break;
                }
                if (gfx10::ends_program(i)) break;
                uint32_t target;
                if (!gfx10::branch_target(i, target)) continue;
                auto tb = block_of.find(target);
                auto fb = block_of.find(i.offset + i.size);
                t.t = tb != block_of.end() ? tb->second : end_block;
                t.f = fb != block_of.end() ? fb->second : end_block;
                t.kind = std::string_view(i.name) == "s_branch" ? cfg::Exit::Jump : cfg::Exit::Branch;
                break;
            }
        }
        return out;
    }

    void if_else(Id cond, const std::function<void()>& f, const std::function<void()>& g) {
        Id yes = b.id(), no = b.id(), after = b.id();
        b.op_void(spv::OpSelectionMerge, {after, 0});
        b.op_void(spv::OpBranchConditional, {cond, yes, no});
        b.label(yes);
        f();
        b.op_void(spv::OpBranch, {after});
        b.label(no);
        g();
        b.op_void(spv::OpBranch, {after});
        b.label(after);
    }

    void count_plan(const std::vector<cfg::Node>& nodes) {
        out.blocks = end_block;
        std::function<void(const std::vector<cfg::Node>&)> walk = [&](const std::vector<cfg::Node>& ns) {
            for (auto& x : ns) {
                if (x.kind == cfg::Node::Range) out.guarded_blocks += x.b - x.a + 1;
                walk(x.body);
                walk(x.other);
            }
        };
        walk(nodes);
    }

    Id pc_is(uint32_t k) { return b.op(spv::OpIEqual, t_bool, {load(pc, t_u32), u(k)}); }
    // A guard in front of block a; b > a when loops start at a (pc may name any block in them).
    Id pc_at(const cfg::Node& x) { return x.b > x.a ? pc_in(x.a, x.b) : pc_is(x.a); }

    void run(const std::vector<cfg::Node>& nodes, const std::vector<uint32_t>& starts) {
        for (auto& x : nodes) {
            switch (x.kind) {
                case cfg::Node::Block: block(x.a, starts[x.a]); break;
                case cfg::Node::If: if_then(pc_at(x), [&] { run(x.body, starts); }); break;
                case cfg::Node::IfElse: if_else(pc_at(x), [&] { run(x.body, starts); }, [&] { run(x.other, starts); }); break;
                case cfg::Node::Range: range(x.a, x.b, starts, false); break;
                case cfg::Node::Loop: loop(regions[x.a], [&] { run(x.body, starts); }); break;
            }
        }
    }

    void range(uint32_t lo, uint32_t hi, const std::vector<uint32_t>& starts, bool top) {
        for (uint32_t k = lo; k <= hi;) {
            Region* outer = nullptr;
            for (auto& r : regions)
                if (!r.active && r.lo == k && r.hi <= hi && (!outer || r.hi > outer->hi)) outer = &r;
            if (outer) {
                uint32_t next = outer->hi + 1;
                loop(*outer, starts);
                k = next;
                continue;
            }
            if (top && k == 0) block(0, starts[0]);
            else if_then(b.op(spv::OpIEqual, t_bool, {load(pc, t_u32), u(k)}), [&] { block(k, starts[k]); });
            ++k;
        }
    }

    bool body() {
        Id fn = b.id();
        b.begin_function(fn, t_void, t_fn);
        b.label();
        scc = b.local(p_fn_bool, b.c_bool(false));
        registers.insert(scc);
        pc = b.local(p_fn_u32, u(0));
        trips = b.local(p_fn_u32, u(0));
        prologue();
        std::vector<uint32_t> starts;
        for (auto& [off, n] : block_of) {
            starts.push_back(off);
            size_t ii = 0;
            while (ii < insts.size() && insts[ii].offset < off) ++ii;
            first_inst.push_back(ii);
        }
        end_block = uint32_t(starts.size());
        find_regions();
        std::vector<cfg::Region> cr;
        for (auto& r : regions) cr.push_back({r.lo, r.hi});
        auto nodes = cfg::plan(terms(), cr, &out.plan_note);
        count_plan(nodes);
        run(nodes, starts);
        if (p.stage == HwStage::Ms) mesh_epilogue();
        b.op_void(spv::OpReturn, {});
        b.end_function();
        // Entry point.
        spv::ExecutionModel model = p.stage == HwStage::Vs ? spv::ExecutionModelVertex
                                  : p.stage == HwStage::Ps ? spv::ExecutionModelFragment
                                  : p.stage == HwStage::Ms ? spv::ExecutionModelMeshEXT
                                                           : spv::ExecutionModelGLCompute;
        b.entry(model, fn, "main", interface);
        if (p.stage == HwStage::Ps) b.mode(fn, spv::ExecutionModeOriginUpperLeft);
        if (p.stage == HwStage::Ms) {
            out.local_size[0] = wide ? (p.ms_threads + 63) / 64 * 32 : p.ms_threads;  // wide: two console threads per invocation
            b.mode(fn, spv::ExecutionModeLocalSize, {out.local_size[0], 1, 1});
            b.mode(fn, spv::ExecutionModeOutputVertices, {ms_vertices()});
            b.mode(fn, spv::ExecutionModeOutputPrimitivesEXT, {ms_primitives()});
            b.mode(fn, ms_lines() ? spv::ExecutionModeOutputLinesEXT : spv::ExecutionModeOutputTrianglesEXT);
        }
        if (p.stage == HwStage::Cs) {
            for (int k = 0; k < 3; ++k) out.local_size[k] = p.cs_threads[k] ? p.cs_threads[k] : 1;
            if (wide) {  // half as many invocations, each running two lanes, in whole 32-lane waves
                uint32_t threads = out.local_size[0] * out.local_size[1] * out.local_size[2];
                out.local_size[0] = (threads + 63) / 64 * 32, out.local_size[1] = out.local_size[2] = 1;
            }
            b.mode(fn, spv::ExecutionModeLocalSize, {out.local_size[0], out.local_size[1], out.local_size[2]});
        }
        return true;
    }

    void emit(const Inst& i) {
        if (wide && vector_inst(i)) {
            std::string_view n = i.name;
            bool once = n == "v_readfirstlane_b32" || n == "v_readlane_b32" || n == "v_writelane_b32" || n == "ds_bpermute_b32";
            for (h = 0; h < (once ? 1 : 2); ++h) emit_one(i);
            h = 0;
            return;
        }
        emit_one(i);
    }
    static bool vector_inst(const Inst& i) {
        switch (i.enc) {
            case Enc::VOP2: case Enc::VOP1: case Enc::VOPC: case Enc::VOP3: case Enc::VOP3P: case Enc::VINTRP: case Enc::DS:
            case Enc::MUBUF: case Enc::MTBUF: case Enc::MIMG: case Enc::FLAT: case Enc::GLOBAL: case Enc::SCRATCH: case Enc::EXP:
                return true;
            default: return false;
        }
    }
    void emit_one(const Inst& i) {
        switch (i.enc) {
            case Enc::MUBUF: case Enc::MTBUF: case Enc::GLOBAL: case Enc::FLAT: case Enc::SCRATCH: case Enc::DS: case Enc::MIMG:
                return vmem(i);
            case Enc::EXP: return export_(i);
            default: return alu(i);
        }
    }
};

}  // namespace

uint32_t code_id(const Program& p) {
    uint32_t h = 2166136261u;
    for (uint8_t c : p.code) h = (h ^ c) * 16777619u;
    return h;
}

uint64_t translator_build() {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const char* s : {__DATE__ " " __TIME__, cfg::build_stamp(), gfx10::build_stamp()})
        for (; *s; ++s) h = (h ^ uint8_t(*s)) * 0x100000001b3ull;
    return h;
}

bool translate(const Program& p, Translation& out, std::string& error, const Options* opt) {
    if (p.stage == HwStage::Ms && (!p.ms_threads || p.ms_threads > 128 || p.ms_max_vertices > 256 || p.ms_max_primitives > 256)) {
        error = "mesh program with " + std::to_string(p.ms_threads) + " threads, " + std::to_string(p.ms_max_vertices) + " vertices, " +
                std::to_string(p.ms_max_primitives) + " primitives (128 threads and 256 of each run here)";
        return false;
    }
    if (p.stage != HwStage::Vs && p.stage != HwStage::Ps && p.stage != HwStage::Cs && p.stage != HwStage::Ms) {
        error = std::string("stage ") + stage_name(p.stage) + " not translated yet";
        return false;
    }
    T t(p, out, opt);
    if (!t.decode_all()) {
        error = t.error;
        return false;
    }
    t.globals();
    if (!t.body()) {
        error = t.error;
        return false;
    }
    out.spirv = t.b.finish();
    return true;
}

}  // namespace kl::shader
