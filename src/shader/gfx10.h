// keylime shaders: GFX10 (RDNA 2) instruction decoder. Encodings from AMD's
// "RDNA 2 Instruction Set Architecture" reference; opcode names from Mesa.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace kl::shader::gfx10 {

enum class Enc : uint8_t {
    SOP2, SOPK, SOP1, SOPC, SOPP, SMEM, VOP2, VOP1, VOPC, VOP3, VOP3P, VINTRP, DS, MUBUF, MTBUF, MIMG, FLAT, GLOBAL,
    SCRATCH, EXP, Unknown
};

// A 9-bit (or 8-bit VGPR / 7-bit SGPR) operand as encoded, with modifiers.
struct Operand {
    uint16_t code = 0;  // 0-255 scalar space (see kinds below), 256-511 VGPR v0-v255
    uint32_t literal = 0;  // when code == 255
    bool neg = false, abs = false;
    bool valid = false;

    bool vgpr() const { return code >= 256; }
    uint16_t vgpr_index() const { return uint16_t(code - 256); }
    bool sgpr() const { return code <= 105; }
    bool literal_const() const { return code == 255; }
    bool inline_const() const { return (code >= 128 && code <= 208) || (code >= 240 && code <= 248); }
};

struct Inst {
    Enc enc = Enc::Unknown;
    uint16_t op = 0;
    const char* name = "?";
    uint32_t offset = 0;   // byte offset in the code
    uint32_t size = 4;     // bytes (with literal)
    uint32_t words[3] = {};

    // ALU
    Operand dst, sdst, src[3];  // sdst: VOP3B carry-out / VOPC-in-VOP3 destination
    uint8_t opsel = 0, opsel_hi = 0, neg_hi = 0, omod = 0;
    bool clamp = false;
    // DPP / SDWA (src0 escape codes 0xfa / 0xe9 / 0xea / 0xf9)
    bool dpp = false, dpp8 = false, sdwa = false;
    uint32_t dpp_ctrl = 0, dpp_row_mask = 0xf, dpp_bank_mask = 0xf, dpp_sel8 = 0;
    bool dpp_bound_ctrl = false, dpp_fi = false;
    uint8_t sdwa_dst_sel = 6, sdwa_dst_unused = 0, sdwa_src_sel[2] = {6, 6};
    bool sdwa_sext[2] = {};
    // Scalar
    int32_t simm16 = 0;  // SOPK / SOPP
    // SMEM
    uint8_t sbase = 0, sdata = 0;
    int32_t smem_offset = 0;
    uint8_t soffset = 0x7d;  // 125 = null
    bool glc = false, dlc = false, slc = false;
    // Memory (DS / MUBUF / MTBUF / FLAT / MIMG)
    uint8_t vaddr = 0, vdata = 0, srsrc = 0, ssamp = 0, saddr = 0x7d, data0 = 0, data1 = 0, offset0 = 0, offset1 = 0;
    int32_t mem_offset = 0;
    bool offen = false, idxen = false, gds = false, lds = false, tfe = false, d16 = false, a16 = false, unorm = false;
    uint8_t dmask = 0, dim = 0, nsa = 0, dfmt = 0;  // MTBUF format in dfmt
    uint8_t nsa_addr[13] = {};
    // Export
    uint8_t exp_target = 0, exp_en = 0, exp_src[4] = {};
    bool exp_compr = false, exp_done = false, exp_vm = false;
    // VINTRP
    uint8_t attr = 0, attr_chan = 0;
};

// Decodes one instruction at `at` (bytes); false at the end or on garbage.
bool decode(const uint8_t* code, size_t size, uint32_t at, Inst& out);
std::string format(const Inst& i);  // assembly-like text
std::string operand_text(const Operand& o);

// True for scalar branches (SOPP s_branch / s_cbranch_*); target in bytes.
bool branch_target(const Inst& i, uint32_t& target);
bool ends_program(const Inst& i);  // s_endpgm*

}  // namespace kl::shader::gfx10
