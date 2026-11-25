// keylime shaders: GFX10 program -> SPIR-V (see docs/translator.md).
//
// Runtime contract (what the GPU side provides):
//   push constants (std430 block, 256 bytes):
//     uint user_sgpr[32]       the user SGPRs (root signature values)
//     uint64 page_table        title address -> device address (2 levels, 1 GiB / 64 KiB)
//     uint64 texture_table     T#/S# hash -> bindless slot
//     uint misc[4]             (reserved)
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

bool translate(const Program& p, Translation& out, std::string& error);

}  // namespace kl::shader
