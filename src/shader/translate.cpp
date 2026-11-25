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
#include <bitset>
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

    #include "shader/translate/helpers.inc"
    #include "shader/translate/memory.inc"
    #include "shader/translate/setup.inc"
    #include "shader/translate/results.inc"
    #include "shader/translate/alu.inc"
    #include "shader/translate/vmem.inc"
    #include "shader/translate/atomics.inc"
    #include "shader/translate/image.inc"
    #include "shader/translate/export.inc"
    #include "shader/translate/control.inc"
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

// The translated-shader cache (cache\spirv, and the driver's pipelines.bin behind it) is keyed by
// this, not by the build: rebuilding keylime, or changing anything that does not change the SPIR-V
// a shader becomes, keeps every translated and compiled shader. Raise it only for a change that
// must reach shaders already in the cache (a translation bug that breaks them), since every shader
// is then translated and compiled again.
constexpr uint32_t TRANSLATOR_VERSION = 1;

uint64_t translator_build() {
    uint64_t h = 0xcbf29ce484222325ull;
    for (const char* s = "keylime-translator"; *s; ++s) h = (h ^ uint8_t(*s)) * 0x100000001b3ull;
    return (h ^ TRANSLATOR_VERSION) * 0x100000001b3ull;
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
    const Translation start = out;
    std::set<uint32_t> distrust;
    for (int pass = 0;; ++pass) {
        out = start;
        T t(p, out, opt);
        t.distrust = distrust;
        if (pass == 8) t.no_lane_facts = true;
        if (!t.decode_all()) {
            error = t.error;
            return false;
        }
        t.globals();
        if (!t.body()) {
            error = t.error;
            return false;
        }
        t.check_assumptions();
        if (!t.broken.empty() && pass < 8) {
            distrust.insert(t.broken.begin(), t.broken.end());
            continue;
        }
        out.spirv = t.b.finish();
        return true;
    }
}

}  // namespace kl::shader
