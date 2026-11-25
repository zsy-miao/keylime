// keylime shaders: the console's shader containers. An Xbox Series shader is
// a DXBC container; its DXIL part has an XDAT index after the program header
// listing blobs around the (usually stripped) bitcode. Kind 0x0d blobs are
// "XBCH" zlib streams unpacking to "FSSM": per hardware stage, a record with
// the GFX10 machine code, wave size, register counts, where root-signature
// parameters arrive (user SGPRs), and the stage's inputs/outputs.
// Format as VIctory observed it (native.rs).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace kl::shader {

enum class HwStage : uint8_t { Ls, Hs, Es, Gs, Vs, Ps, Ms, Cs };
const char* stage_name(HwStage s);

enum class RootType : uint8_t { Constants, Descriptor, Table, ImmediateSampler, ResourceHeap, SamplerHeap, Other };
enum class ResourceType : uint8_t { Srv, Sampler, Cbv, Uav, Other };

struct RootParam {
    RootType root;
    uint32_t table_offset_dwords, sgprs, base_register;
    uint8_t user_data, slot;
};
struct UsedResource {
    ResourceType kind;
    RootType root;
    uint32_t table_offset_dwords, lower_bound, stride;
    uint8_t dwords, user_data, slot;
};

// Semantic usage: 6 generic, 7 clip distance, others raw.
struct PsInput {
    uint8_t usage, row, mask, attr;
    bool flat, fp16, pack2;
    uint8_t row2;
};
struct VsInput {
    uint8_t usage, row, mask, vgpr;
};
struct VsOutput {
    uint8_t usage, row, param;
    bool fp16, pack2;
};
struct ExportPatch {
    uint8_t mrt;
    uint32_t offset;  // dwords into the code
    std::vector<uint32_t> code;
};

struct Program {
    HwStage stage;
    bool wave64;
    uint16_t vgprs;
    std::vector<RootParam> layout;
    std::vector<UsedResource> used;
    std::vector<uint8_t> code;  // GFX10 machine code
    // Stage I/O.
    std::vector<PsInput> ps_inputs;
    uint32_t ps_input_addr = 0, db_shader_control = 0;
    std::vector<ExportPatch> patches;
    std::vector<VsInput> vs_inputs;
    std::vector<VsOutput> vs_outputs;
    uint32_t cs_rsrc2 = 0, cs_threads[3] = {1, 1, 1};
    uint32_t lds_bytes = 0;
    // The code with each render target's export patched for its full 32-bit format.
    std::vector<uint8_t> patched_code() const;
};

struct Container {
    uint32_t program_version = 0;  // kind << 16 | major << 4 | minor
    bool has_bitcode = false;
    std::vector<Program> programs;
};

// Parses a DXBC container; false with `error` on malformed input.
bool parse(const uint8_t* data, size_t size, Container& out, std::string& error);

}  // namespace kl::shader
