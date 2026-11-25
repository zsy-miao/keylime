// keylime shaders: GFX10 program -> SPIR-V (see docs/translator.md).
//
// Runtime contract (what the GPU side provides):
//   push constants (32 bytes, every stage):
//     uint64 draw_data         this draw's record: user SGPRs of each stage (32 dwords at
//                              Options::sgpr_offset)
//     uint64 page_table        title address -> device address (2 levels, 1 GiB / 64 KiB)
//     uint64 texture_table     T#/S# hash -> bindless slot
//     uint misc[2]             (reserved)
//   set 0: binding 0 sampled images[], binding 1 samplers[], binding 2 storage images[]
//   VS: inputs at location = signature row; outputs at location = param row.
//   PS: inputs at location = signature row; colour outputs at location = MRT index.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "shader/container.h"

namespace kl::shader {

struct Translation {
    std::vector<uint32_t> spirv;
    std::map<std::string, size_t> unsupported;  // instruction name -> count (emitted as no-ops)
    uint32_t local_size[3] = {1, 1, 1};
    std::vector<uint8_t> vs_input_rows, vs_output_rows, ps_input_rows;
    uint8_t mrt_mask = 0;
    bool writes_depth = false, discards = false;
};

// Host choices for one translation.
struct Options {
    std::map<uint8_t, uint8_t> ps_location;  // PS input row -> location (the linked VS output row)
    uint32_t sgpr_offset = 0;                // dwords into the draw record where this stage's SGPRs start
};

bool translate(const Program& p, Translation& out, std::string& error, const Options* opt = nullptr);

}  // namespace kl::shader
