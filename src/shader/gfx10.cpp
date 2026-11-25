#include "shader/gfx10.h"

#include <cstdio>
#include <cstring>

#include "shader/gfx10_opcodes.h"

namespace kl::shader::gfx10 {
namespace {

template <size_t N>
const char* lookup(const OpName (&t)[N], uint16_t op) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t m = (lo + hi) / 2;
        if (t[m].op < op) lo = m + 1;
        else hi = m;
    }
    return lo < N && t[lo].op == op ? t[lo].name : nullptr;
}

uint32_t bits(uint32_t w, int lo, int n) { return uint32_t((uint64_t(w) >> lo) & ((1ull << n) - 1)); }

Operand src9(uint32_t code) {
    Operand o;
    o.code = uint16_t(code);
    o.valid = true;
    return o;
}
Operand vgpr8(uint32_t r) {
    Operand o;
    o.code = uint16_t(256 + r);
    o.valid = true;
    return o;
}
Operand sgpr7(uint32_t r) {
    Operand o;
    o.code = uint16_t(r);
    o.valid = true;
    return o;
}

// A literal follows when any 9-bit source is 255.
bool take_literal(Inst& i, const uint8_t* code, size_t size) {
    bool need = false;
    for (auto& s : i.src)
        if (s.valid && s.code == 255) need = true;
    if (!need) return true;
    if (i.offset + i.size + 4 > size) return false;
    uint32_t lit;
    memcpy(&lit, code + i.offset + i.size, 4);
    for (auto& s : i.src)
        if (s.valid && s.code == 255) s.literal = lit;
    i.size += 4;
    return true;
}

bool second(Inst& i, const uint8_t* code, size_t size) {
    if (i.offset + 8 > size) return false;
    memcpy(&i.words[1], code + i.offset + 4, 4);
    i.size = 8;
    return true;
}

// src0 escapes on VOP1/VOP2/VOPC: 0xfa DPP16, 0xe9/0xea DPP8, 0xf9 SDWA. Extra dword follows.
bool vop_extension(Inst& i, const uint8_t* code, size_t size) {
    uint16_t s0 = i.src[0].code;
    if (s0 != 0xfa && s0 != 0xe9 && s0 != 0xea && s0 != 0xf9) return true;
    if (i.offset + 8 > size) return false;
    uint32_t w;
    memcpy(&w, code + i.offset + 4, 4);
    i.words[1] = w;
    i.size = 8;
    if (s0 == 0xfa) {
        i.dpp = true;
        i.src[0] = vgpr8(w & 0xff);
        i.dpp_ctrl = (w >> 8) & 0x1ff;
        i.dpp_fi = (w >> 18) & 1;
        i.dpp_bound_ctrl = (w >> 19) & 1;
        i.src[0].neg = (w >> 20) & 1, i.src[0].abs = (w >> 21) & 1;
        i.src[1].neg = (w >> 22) & 1, i.src[1].abs = (w >> 23) & 1;
        i.dpp_bank_mask = (w >> 24) & 0xf;
        i.dpp_row_mask = (w >> 28) & 0xf;
    } else if (s0 == 0xe9 || s0 == 0xea) {
        i.dpp = i.dpp8 = true;
        i.dpp_fi = s0 == 0xea;
        i.src[0] = vgpr8(w & 0xff);
        i.dpp_sel8 = w >> 8;
    } else {
        i.sdwa = true;
        uint32_t s0r = w & 0xff;
        bool s0_sgpr = (w >> 23) & 1;
        i.src[0] = s0_sgpr ? src9(s0r) : vgpr8(s0r);
        i.sdwa_dst_sel = (w >> 8) & 7;
        i.sdwa_dst_unused = (w >> 11) & 3;
        i.clamp = (w >> 13) & 1;
        i.omod = (w >> 14) & 3;
        i.sdwa_src_sel[0] = (w >> 16) & 7;
        i.sdwa_sext[0] = (w >> 19) & 1;
        i.src[0].neg = (w >> 20) & 1, i.src[0].abs = (w >> 21) & 1;
        i.sdwa_src_sel[1] = (w >> 24) & 7;
        i.sdwa_sext[1] = (w >> 27) & 1;
        i.src[1].neg = (w >> 28) & 1, i.src[1].abs = (w >> 29) & 1;
        if ((w >> 31) & 1) i.src[1] = src9(i.src[1].vgpr() ? i.src[1].vgpr_index() : i.src[1].code);  // src1 is an SGPR
    }
    return true;
}

}  // namespace

const char* build_stamp() { return __DATE__ " " __TIME__; }

bool decode(const uint8_t* code, size_t size, uint32_t at, Inst& i) {
    if (at + 4 > size) return false;
    i = Inst{};
    i.offset = at;
    uint32_t w;
    memcpy(&w, code + at, 4);
    i.words[0] = w;
    i.size = 4;

    if ((w >> 31) == 0 && (w >> 25) != 0x3f && (w >> 25) != 0x3e) {  // VOP2 (VOP1/VOPC share its space)
        i.enc = Enc::VOP2;
        i.op = uint16_t(bits(w, 25, 6));
        i.name = lookup(VOP2, i.op);
        i.dst = vgpr8(bits(w, 17, 8));
        i.src[0] = src9(bits(w, 0, 9));
        i.src[1] = vgpr8(bits(w, 9, 8));
        // v_fmamk/fmaak/madmk/madak carry a literal K.
        if (i.name && (strstr(i.name, "mk_") || strstr(i.name, "ak_"))) {
            if (at + 8 > size) return false;
            memcpy(&i.src[2].literal, code + at + 4, 4);
            i.src[2].code = 255, i.src[2].valid = true;
            i.size = 8;
            return i.name != nullptr;
        }
        if (!vop_extension(i, code, size)) return false;
        return take_literal(i, code, size) && i.name;
    }
    uint32_t top2 = w >> 30, top4 = w >> 28, top6 = w >> 26, top7 = w >> 25, top9 = w >> 23;
    if (top9 == 0x17d) {  // SOP1 101111101
        i.enc = Enc::SOP1;
        i.op = uint16_t(bits(w, 8, 8));
        i.name = lookup(SOP1, i.op);
        i.dst = sgpr7(bits(w, 16, 7));
        i.src[0] = src9(bits(w, 0, 8));
        return take_literal(i, code, size) && i.name;
    }
    if (top9 == 0x17e) {  // SOPC
        i.enc = Enc::SOPC;
        i.op = uint16_t(bits(w, 16, 7));
        i.name = lookup(SOPC, i.op);
        i.src[0] = src9(bits(w, 0, 8));
        i.src[1] = src9(bits(w, 8, 8));
        return take_literal(i, code, size) && i.name;
    }
    if (top9 == 0x17f) {  // SOPP
        i.enc = Enc::SOPP;
        i.op = uint16_t(bits(w, 16, 7));
        i.name = lookup(SOPP, i.op);
        i.simm16 = int16_t(w & 0xffff);
        return i.name != nullptr;
    }
    if (top4 == 0xb) {  // SOPK 1011
        i.enc = Enc::SOPK;
        i.op = uint16_t(bits(w, 23, 5));
        i.name = lookup(SOPK, i.op);
        i.dst = sgpr7(bits(w, 16, 7));
        i.simm16 = int16_t(w & 0xffff);
        if (i.name && !strcmp(i.name, "s_setreg_imm32_b32")) {  // 32-bit literal follows
            if (at + 8 > size) return false;
            memcpy(&i.src[0].literal, code + at + 4, 4);
            i.src[0].code = 255, i.src[0].valid = true;
            i.size = 8;
        }
        return i.name != nullptr;
    }
    if (top2 == 2) {  // SOP2 10
        i.enc = Enc::SOP2;
        i.op = uint16_t(bits(w, 23, 7));
        i.name = lookup(SOP2, i.op);
        i.dst = sgpr7(bits(w, 16, 7));
        i.src[0] = src9(bits(w, 0, 8));
        i.src[1] = src9(bits(w, 8, 8));
        return take_literal(i, code, size) && i.name;
    }
    if (top7 == 0x3f) {  // VOP1 0111111
        i.enc = Enc::VOP1;
        i.op = uint16_t(bits(w, 9, 8));
        i.name = lookup(VOP1, i.op);
        i.dst = vgpr8(bits(w, 17, 8));
        i.src[0] = src9(bits(w, 0, 9));
        if (!vop_extension(i, code, size)) return false;
        return take_literal(i, code, size) && i.name;
    }
    if (top7 == 0x3e) {  // VOPC 0111110
        i.enc = Enc::VOPC;
        i.op = uint16_t(bits(w, 17, 8));
        i.name = lookup(VOPC, i.op);
        i.src[0] = src9(bits(w, 0, 9));
        i.src[1] = vgpr8(bits(w, 9, 8));
        if (!vop_extension(i, code, size)) return false;
        return take_literal(i, code, size) && i.name;
    }
    switch (top6) {
        case 0x3d: {  // SMEM 111101
            if (!second(i, code, size)) return false;
            i.enc = Enc::SMEM;
            i.op = uint16_t(bits(w, 18, 8));
            i.name = lookup(SMEM, i.op);
            i.sbase = uint8_t(bits(w, 0, 6) << 1);
            i.sdata = uint8_t(bits(w, 6, 7));
            i.dlc = bits(w, 14, 1), i.glc = bits(w, 16, 1);
            uint32_t w1 = i.words[1];
            i.smem_offset = int32_t(w1 << 11) >> 11;  // 21-bit signed
            i.soffset = uint8_t(w1 >> 25);
            return i.name != nullptr;
        }
        case 0x35: {  // VOP3 110101
            if (!second(i, code, size)) return false;
            i.enc = Enc::VOP3;
            i.op = uint16_t(bits(w, 16, 10));
            uint32_t w1 = i.words[1];
            if (i.op < 0x100) i.name = lookup(VOPC, i.op);
            else if (i.op < 0x140) i.name = lookup(VOP2, uint16_t(i.op - 0x100));
            else if (i.op >= 0x180 && i.op < 0x200) i.name = lookup(VOP1, uint16_t(i.op - 0x180));
            else i.name = lookup(VOP3, i.op);
            i.dst = vgpr8(bits(w, 0, 8));
            i.src[0] = src9(bits(w1, 0, 9));
            i.src[1] = src9(bits(w1, 9, 9));
            i.src[2] = src9(bits(w1, 18, 9));
            for (int k = 0; k < 3; ++k) i.src[k].neg = (w1 >> (29 + k)) & 1;
            i.omod = uint8_t(bits(w1, 27, 2));
            i.clamp = bits(w, 15, 1);
            // VOP3B (carry out in sdst): v_add_co / v_sub_co / div_scale / mad_u64 ...
            bool vop3b = i.name && (strstr(i.name, "_co_") || strstr(i.name, "div_scale") || strstr(i.name, "mad_u64") ||
                                    strstr(i.name, "mad_i64"));
            if (vop3b) {
                i.sdst = sgpr7(bits(w, 8, 7));
                i.clamp = bits(w, 15, 1);
            } else {
                for (int k = 0; k < 3; ++k) i.src[k].abs = (w >> (8 + k)) & 1;
                i.opsel = uint8_t(bits(w, 11, 4));
            }
            if (i.op < 0x100) i.sdst = sgpr7(bits(w, 0, 8)), i.dst = Operand{};  // VOPC in VOP3: sdst in vdst field
            return take_literal(i, code, size) && i.name;
        }
        case 0x33: {  // VOP3P 110011
            if (!second(i, code, size)) return false;
            i.enc = Enc::VOP3P;
            i.op = uint16_t(bits(w, 16, 7));
            i.name = lookup(VOP3P, i.op);
            uint32_t w1 = i.words[1];
            i.dst = vgpr8(bits(w, 0, 8));
            i.src[0] = src9(bits(w1, 0, 9));
            i.src[1] = src9(bits(w1, 9, 9));
            i.src[2] = src9(bits(w1, 18, 9));
            i.neg_hi = uint8_t(bits(w, 8, 3));
            i.opsel = uint8_t(bits(w, 11, 3));
            i.opsel_hi = uint8_t(bits(w1, 27, 2) | (bits(w, 14, 1) << 2));
            for (int k = 0; k < 3; ++k) i.src[k].neg = (w1 >> (29 + k)) & 1;
            i.clamp = bits(w, 15, 1);
            return take_literal(i, code, size) && i.name;
        }
        case 0x32: {  // VINTRP 110010
            i.enc = Enc::VINTRP;
            i.op = uint16_t(bits(w, 16, 2));
            i.name = lookup(VINTRP, i.op);
            i.dst = vgpr8(bits(w, 18, 8));
            i.src[0] = vgpr8(bits(w, 0, 8));
            i.attr_chan = uint8_t(bits(w, 8, 2));
            i.attr = uint8_t(bits(w, 10, 6));
            return i.name != nullptr;
        }
        case 0x36: {  // DS 110110
            if (!second(i, code, size)) return false;
            i.enc = Enc::DS;
            i.op = uint16_t(bits(w, 18, 8));
            i.name = lookup(DS, i.op);
            i.offset0 = uint8_t(bits(w, 0, 8)), i.offset1 = uint8_t(bits(w, 8, 8));
            i.mem_offset = int32_t(bits(w, 0, 16));
            i.gds = bits(w, 17, 1);
            uint32_t w1 = i.words[1];
            i.vaddr = uint8_t(bits(w1, 0, 8)), i.data0 = uint8_t(bits(w1, 8, 8)), i.data1 = uint8_t(bits(w1, 16, 8));
            i.dst = vgpr8(bits(w1, 24, 8));
            return i.name != nullptr;
        }
        case 0x38:  // MUBUF 111000
        case 0x3a: {  // MTBUF 111010
            if (!second(i, code, size)) return false;
            bool mt = top6 == 0x3a;
            i.enc = mt ? Enc::MTBUF : Enc::MUBUF;
            if (mt) {
                i.op = uint16_t(bits(w, 16, 3) | (bits(i.words[1], 21, 1) << 3));
                i.dfmt = uint8_t(bits(w, 19, 7));
                i.name = lookup(MTBUF, i.op);
            } else {
                i.op = uint16_t(bits(w, 18, 7) | (bits(i.words[1], 21, 1) << 7));
                i.name = lookup(MUBUF, i.op);
                i.lds = bits(w, 16, 1);
                i.slc = bits(w, 17, 1);
            }
            i.mem_offset = int32_t(bits(w, 0, 12));
            i.offen = bits(w, 12, 1), i.idxen = bits(w, 13, 1), i.glc = bits(w, 14, 1), i.dlc = bits(w, 15, 1);
            uint32_t w1 = i.words[1];
            i.vaddr = uint8_t(bits(w1, 0, 8)), i.vdata = uint8_t(bits(w1, 8, 8));
            i.srsrc = uint8_t(bits(w1, 16, 5) << 2);
            if (mt) i.slc = bits(w1, 22, 1);
            i.tfe = bits(w1, 23, 1);
            i.soffset = uint8_t(bits(w1, 24, 8));
            return i.name != nullptr;
        }
        case 0x3c: {  // MIMG 111100
            if (!second(i, code, size)) return false;
            i.enc = Enc::MIMG;
            i.nsa = uint8_t(bits(w, 1, 2));
            i.dim = uint8_t(bits(w, 3, 3));
            i.dlc = bits(w, 7, 1);
            i.dmask = uint8_t(bits(w, 8, 4));
            i.unorm = bits(w, 12, 1);
            i.glc = bits(w, 13, 1);
            i.op = uint16_t(bits(w, 18, 7) | (bits(w, 0, 1) << 7));
            i.slc = bits(w, 25, 1);
            i.name = lookup(MIMG, i.op);
            uint32_t w1 = i.words[1];
            i.vaddr = uint8_t(bits(w1, 0, 8)), i.vdata = uint8_t(bits(w1, 8, 8));
            i.srsrc = uint8_t(bits(w1, 16, 5) << 2);
            i.ssamp = uint8_t(bits(w1, 21, 5) << 2);
            i.a16 = bits(w1, 30, 1), i.d16 = bits(w1, 31, 1);
            i.tfe = bits(w1, 22, 1);
            i.lds = bits(w1, 23, 1);  // lwe
            if (i.nsa) {  // extra address dwords: one VGPR byte each
                size_t extra = size_t(i.nsa) * 4;
                if (at + 8 + extra > size) return false;
                for (size_t k = 0; k < extra && k < 13; ++k) i.nsa_addr[k] = code[at + 8 + k];
                i.size = uint32_t(8 + extra);
            }
            return i.name != nullptr;
        }
        case 0x37: {  // FLAT / GLOBAL / SCRATCH 110111
            if (!second(i, code, size)) return false;
            uint32_t seg = bits(w, 14, 2);
            i.enc = seg == 0 ? Enc::FLAT : seg == 1 ? Enc::SCRATCH : Enc::GLOBAL;
            i.op = uint16_t(bits(w, 18, 7));
            i.name = seg == 0 ? lookup(FLAT, i.op) : seg == 1 ? lookup(SCRATCH, i.op) : lookup(GLOBAL, i.op);
            i.mem_offset = int32_t(bits(w, 0, 12) << 20) >> 20;  // 12-bit signed (flat: unsigned in practice)
            i.dlc = bits(w, 12, 1), i.lds = bits(w, 13, 1), i.glc = bits(w, 16, 1), i.slc = bits(w, 17, 1);
            uint32_t w1 = i.words[1];
            i.vaddr = uint8_t(bits(w1, 0, 8)), i.vdata = uint8_t(bits(w1, 8, 8));
            i.saddr = uint8_t(bits(w1, 16, 7));
            i.dst = vgpr8(bits(w1, 24, 8));
            return i.name != nullptr;
        }
        case 0x3e: {  // EXP 111110
            if (!second(i, code, size)) return false;
            i.enc = Enc::EXP;
            i.name = "exp";
            i.exp_en = uint8_t(bits(w, 0, 4));
            i.exp_target = uint8_t(bits(w, 4, 6));
            i.exp_compr = bits(w, 10, 1), i.exp_done = bits(w, 11, 1), i.exp_vm = bits(w, 12, 1);
            for (int k = 0; k < 4; ++k) i.exp_src[k] = uint8_t(i.words[1] >> (8 * k));
            return true;
        }
        default: break;
    }
    i.enc = Enc::Unknown;
    return false;
}

std::string operand_text(const Operand& o) {
    char b[48];
    uint16_t c = o.code;
    if (c >= 256) snprintf(b, sizeof b, "v%u", c - 256);
    else if (c <= 105) snprintf(b, sizeof b, "s%u", c);
    else if (c == 106) return "vcc_lo";
    else if (c == 107) return "vcc_hi";
    else if (c >= 108 && c <= 123) snprintf(b, sizeof b, "ttmp%u", c - 108);
    else if (c == 124) return "m0";
    else if (c == 125) return "null";
    else if (c == 126) return "exec_lo";
    else if (c == 127) return "exec_hi";
    else if (c >= 128 && c <= 192) snprintf(b, sizeof b, "%d", c - 128);
    else if (c >= 193 && c <= 208) snprintf(b, sizeof b, "%d", -(int(c) - 192));
    else if (c == 240) return "0.5";
    else if (c == 241) return "-0.5";
    else if (c == 242) return "1.0";
    else if (c == 243) return "-1.0";
    else if (c == 244) return "2.0";
    else if (c == 245) return "-2.0";
    else if (c == 246) return "4.0";
    else if (c == 247) return "-4.0";
    else if (c == 248) return "0.15915494";
    else if (c == 251) return "vccz";
    else if (c == 252) return "execz";
    else if (c == 253) return "scc";
    else if (c == 254) return "lds_direct";
    else if (c == 255) snprintf(b, sizeof b, "0x%x", o.literal);
    else snprintf(b, sizeof b, "src%u", c);
    std::string s = b;
    if (o.abs) s = "|" + s + "|";
    if (o.neg) s = "-" + s;
    return s;
}

std::string format(const Inst& i) {
    std::string s = i.name;
    auto add = [&](const std::string& t) { s += (s.size() == strlen(i.name) ? " " : ", ") + t; };
    switch (i.enc) {
        case Enc::SOPP: {
            char b[24];
            snprintf(b, sizeof b, "%d", i.simm16);
            add(b);
            break;
        }
        case Enc::SOPK: {
            char b[24];
            add(operand_text(i.dst));
            snprintf(b, sizeof b, "0x%x", uint16_t(i.simm16));
            add(b);
            break;
        }
        case Enc::SMEM: {
            char b[64];
            snprintf(b, sizeof b, "s[%u], s[%u:%u], 0x%x", i.sdata, i.sbase, i.sbase + 1, i.smem_offset);
            add(b);
            if (i.soffset != 125) add("s" + std::to_string(i.soffset));
            break;
        }
        case Enc::MUBUF: case Enc::MTBUF: case Enc::MIMG: case Enc::FLAT: case Enc::GLOBAL: case Enc::SCRATCH: {
            char b[96];
            snprintf(b, sizeof b, "v%u, v%u, s[%u:%u] off 0x%x%s%s", i.vdata, i.vaddr, i.srsrc, i.srsrc + 3, i.mem_offset,
                     i.offen ? " offen" : "", i.idxen ? " idxen" : "");
            if (i.enc == Enc::MIMG) snprintf(b, sizeof b, "v%u, v%u, s[%u], s[%u] dmask 0x%x dim %u", i.vdata, i.vaddr, i.srsrc, i.ssamp, i.dmask, i.dim);
            if (i.enc == Enc::GLOBAL || i.enc == Enc::FLAT) snprintf(b, sizeof b, "v%u, v[%u], saddr s%u off %d", i.vdata, i.vaddr, i.saddr, i.mem_offset);
            add(b);
            break;
        }
        case Enc::DS: {
            char b[64];
            snprintf(b, sizeof b, "%s, v%u, v%u, v%u off %d", operand_text(i.dst).c_str(), i.vaddr, i.data0, i.data1, i.mem_offset);
            add(b);
            break;
        }
        case Enc::EXP: {
            char b[80];
            snprintf(b, sizeof b, "target %u en 0x%x v%u v%u v%u v%u%s%s", i.exp_target, i.exp_en, i.exp_src[0], i.exp_src[1],
                     i.exp_src[2], i.exp_src[3], i.exp_done ? " done" : "", i.exp_compr ? " compr" : "");
            add(b);
            break;
        }
        case Enc::VINTRP: {
            char b[48];
            snprintf(b, sizeof b, "%s, v%u, attr%u.%c", operand_text(i.dst).c_str(), i.src[0].vgpr_index(), i.attr, "xyzw"[i.attr_chan]);
            add(b);
            break;
        }
        default:
            if (i.dst.valid) add(operand_text(i.dst));
            if (i.sdst.valid) add(operand_text(i.sdst));
            for (auto& o : i.src)
                if (o.valid) add(operand_text(o));
            if (i.dpp) add(i.dpp8 ? "dpp8" : "dpp");
            if (i.sdwa) add("sdwa");
            break;
    }
    return s;
}

bool branch_target(const Inst& i, uint32_t& target) {
    if (i.enc != Enc::SOPP) return false;
    if (strncmp(i.name, "s_branch", 8) && strncmp(i.name, "s_cbranch_", 10)) return false;
    if (!strcmp(i.name, "s_cbranch_join")) return false;
    target = uint32_t(int64_t(i.offset) + 4 + int64_t(i.simm16) * 4);
    return true;
}

bool ends_program(const Inst& i) { return i.enc == Enc::SOPP && !strncmp(i.name, "s_endpgm", 8); }

}  // namespace kl::shader::gfx10
