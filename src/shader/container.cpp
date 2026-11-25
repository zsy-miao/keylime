#include "shader/container.h"

#include <cstring>

#include "miniz.h"

namespace kl::shader {
namespace {

constexpr size_t XDAT_OFFSET = 0x18, MAX_XDAT = 64;
constexpr uint32_t KIND_NATIVE = 0x0d, XBCH_VERSION = 3, FSSM_VERSION = 0x101;
constexpr size_t XBCH_HEADER = 0x18, FSSM_HEADER = 0x10;
constexpr uint64_t ABSENT = 0xDEADBEEFDEADBEEFull, RECORD_SIZE = 0x54, ENTRY_SIZE = 20;
constexpr size_t RECORD_FLAGS = 0x14, RECORD_REGS = 0x50, RECORD_LAYOUT = 0x28, RECORD_USED = 0x30, RECORD_COUNTS = 0x38,
                 RECORD_LDS = 0x4c;
constexpr uint32_t FLAG_WAVE32 = 1 << 5;
constexpr size_t PS_INPUTS = 0x24, PS_INPUT_COUNT = 0x124, PS_INPUT_ADDR = 0x134, PS_DB = 0x138, PS_PATCH_SIZE = 0x148,
                 PS_PATCHES = 0x14c, PATCH_SLOT_DWORDS = 10, PATCH_FULL = 9, PATCH_BYTES = 4 + 10 * PATCH_SLOT_DWORDS * 4, MRTS = 8;
constexpr size_t VS_INPUTS = 0x24, VS_INPUT_COUNT = 0xc4, VS_OUTPUTS = 0xc8, VS_OUTPUT_COUNT = 0x188;
constexpr size_t CS_RSRC2 = 0x24, CS_THREADS = 0x28;
constexpr uint32_t MAX_SEMANTICS = 32;
constexpr uint64_t MAX_UNPACKED = 16 << 20;

const std::pair<HwStage, size_t> SLOTS[8] = {{HwStage::Ls, 0x08}, {HwStage::Hs, 0x10}, {HwStage::Es, 0x18}, {HwStage::Gs, 0x20},
                                              {HwStage::Vs, 0x28}, {HwStage::Ps, 0x30}, {HwStage::Ms, 0x48}, {HwStage::Cs, 0x60}};

struct View {
    const uint8_t* p;
    size_t n;
    bool u32(size_t at, uint32_t& v) const { return at + 4 <= n ? (memcpy(&v, p + at, 4), true) : false; }
    bool u64(size_t at, uint64_t& v) const { return at + 8 <= n ? (memcpy(&v, p + at, 8), true) : false; }
    bool sub(uint64_t off, uint64_t len, View& out) const {
        if (off > n || len > n - off) return false;
        out = {p + off, size_t(len)};
        return true;
    }
};

RootType root_type(uint32_t r) {
    switch (r) {
        case 0: return RootType::Constants;
        case 1: return RootType::Descriptor;
        case 3: return RootType::Table;
        case 4: return RootType::ImmediateSampler;
        case 5: return RootType::ResourceHeap;
        case 6: return RootType::SamplerHeap;
        default: return RootType::Other;
    }
}

bool programs(const std::vector<uint8_t>& fssm, Container& out, std::string& err) {
    View f{fssm.data(), fssm.size()};
    uint32_t version;
    if (fssm.size() < FSSM_HEADER || memcmp(fssm.data(), "FSSM", 4)) return err = "no FSSM magic", false;
    if (!f.u32(4, version) || version != FSSM_VERSION) return err = "unknown FSSM version", false;
    View payload{fssm.data() + FSSM_HEADER, fssm.size() - FSSM_HEADER};
    uint64_t size, slots;
    if (!f.u64(8, size) || size != payload.n) return err = "FSSM payload size disagrees", false;
    if (!payload.u64(0, slots) || slots != 6) return err = "unexpected FSSM slot count", false;
    for (auto [stage, at] : SLOTS) {
        uint64_t start;
        if (!payload.u64(at, start)) return err = "FSSM truncated", false;
        if (start == ABSENT) continue;
        View rec;
        if (!payload.sub(start, RECORD_SIZE, rec)) return err = "FSSM record out of bounds", false;
        uint64_t code_off, code_len;
        rec.u64(0, code_off), rec.u64(8, code_len);
        View code;
        if (!payload.sub(code_off, code_len, code)) return err = "native code out of bounds", false;
        Program p{};
        p.stage = stage;
        uint32_t flags, regs, lds;
        rec.u32(RECORD_FLAGS, flags), rec.u32(RECORD_REGS, regs), rec.u32(RECORD_LDS, lds);
        p.wave64 = !(flags & FLAG_WAVE32);
        p.vgprs = uint16_t(regs & 0x3ff);
        p.lds_bytes = lds;
        p.code.assign(code.p, code.p + code.n);
        uint32_t n_layout, n_used;
        rec.u32(RECORD_COUNTS, n_layout), rec.u32(RECORD_COUNTS + 4, n_used);
        uint64_t layout_off, used_off;
        rec.u64(RECORD_LAYOUT, layout_off), rec.u64(RECORD_USED, used_off);
        View t;
        if (n_layout && payload.sub(layout_off, ENTRY_SIZE * n_layout, t))
            for (uint32_t i = 0; i < n_layout; ++i) {
                const uint8_t* e = t.p + i * ENTRY_SIZE;
                RootParam r{};
                uint32_t v[4];
                memcpy(v, e, 16);
                r.root = root_type(v[0]), r.table_offset_dwords = v[1], r.sgprs = v[2], r.base_register = v[3];
                r.user_data = e[16], r.slot = e[17];
                p.layout.push_back(r);
            }
        if (n_used && payload.sub(used_off, ENTRY_SIZE * n_used, t))
            for (uint32_t i = 0; i < n_used; ++i) {
                const uint8_t* e = t.p + i * ENTRY_SIZE;
                UsedResource u{};
                u.kind = e[0] <= 3 ? ResourceType(e[0]) : ResourceType::Other;
                u.root = root_type(e[1]);
                memcpy(&u.table_offset_dwords, e + 4, 4), memcpy(&u.lower_bound, e + 8, 4), memcpy(&u.stride, e + 12, 4);
                u.dwords = e[16], u.user_data = e[17], u.slot = e[18];
                p.used.push_back(u);
            }
        View sd{payload.p + start + RECORD_SIZE, payload.n - size_t(start + RECORD_SIZE)};
        auto count = [&](size_t at, uint32_t& c) { return sd.u32(at, c) && c <= MAX_SEMANTICS; };
        if (stage == HwStage::Ps) {
            uint32_t c;
            if (!count(PS_INPUT_COUNT, c)) return err = "PS stage data", false;
            for (uint32_t i = 0; i < c; ++i) {
                const uint8_t* e = sd.p + PS_INPUTS + 8 * i;
                p.ps_inputs.push_back({e[0], e[1], uint8_t(e[2] & 0xf), e[3], (e[7] & 1) != 0, (e[7] & 8) != 0, (e[7] & 0x10) != 0, e[5]});
            }
            sd.u32(PS_INPUT_ADDR, p.ps_input_addr), sd.u32(PS_DB, p.db_shader_control);
            uint32_t psize;
            sd.u32(PS_PATCH_SIZE, psize);
            if (psize > PATCH_SLOT_DWORDS) return err = "implausible export patch size", false;
            for (size_t mrt = 0; mrt < MRTS; ++mrt) {
                size_t a = PS_PATCHES + mrt * PATCH_BYTES;
                uint32_t off;
                if (!sd.u32(a, off) || off == ~0u) continue;
                if ((size_t(off) + psize) * 4 > p.code.size()) return err = "export patch outside the code", false;
                ExportPatch ep{uint8_t(mrt), off, {}};
                for (size_t i = 0; i < psize; ++i) {
                    uint32_t w;
                    sd.u32(a + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4 + 4 * i, w);
                    ep.code.push_back(w);
                }
                p.patches.push_back(ep);
            }
        } else if (stage == HwStage::Vs || stage == HwStage::Ms) {
            uint32_t c;
            if (stage == HwStage::Vs) {
                if (!count(VS_INPUT_COUNT, c)) return err = "VS stage data", false;
                for (uint32_t i = 0; i < c; ++i) {
                    const uint8_t* e = sd.p + VS_INPUTS + 5 * i;
                    p.vs_inputs.push_back({e[0], e[1], uint8_t(e[2] & 0xf), e[3]});
                }
            }
            if (!count(VS_OUTPUT_COUNT, c)) return err = "VS outputs", false;
            for (uint32_t i = 0; i < c; ++i) {
                const uint8_t* e = sd.p + VS_OUTPUTS + 6 * i;
                p.vs_outputs.push_back({e[0], e[1], e[2], (e[5] & 1) != 0, (e[5] & 2) != 0});
            }
        } else if (stage == HwStage::Cs) {
            sd.u32(CS_RSRC2, p.cs_rsrc2);
            for (int i = 0; i < 3; ++i) sd.u32(CS_THREADS + 4 * i, p.cs_threads[i]);
        }
        out.programs.push_back(std::move(p));
    }
    return true;
}

bool unpack(View blob, std::vector<uint8_t>& out, std::string& err) {
    uint32_t version;
    uint64_t packed, unpacked;
    if (blob.n < XBCH_HEADER || memcmp(blob.p, "XBCH", 4)) return err = "no XBCH magic", false;
    if (!blob.u32(4, version) || version != XBCH_VERSION) return err = "unknown XBCH version", false;
    blob.u64(8, packed), blob.u64(16, unpacked);
    if (unpacked > MAX_UNPACKED) return err = "implausible XBCH size", false;
    View stream;
    if (!blob.sub(XBCH_HEADER, packed, stream)) return err = "XBCH stream out of bounds", false;
    out.resize(size_t(unpacked));
    size_t n = tinfl_decompress_mem_to_mem(out.data(), out.size(), stream.p, stream.n, TINFL_FLAG_PARSE_ZLIB_HEADER);
    if (n != unpacked) return err = "XBCH stream does not inflate to its size", false;
    return true;
}

}  // namespace

const char* stage_name(HwStage s) {
    static const char* n[] = {"ls", "hs", "es", "gs", "vs", "ps", "ms", "cs"};
    return n[size_t(s)];
}

std::vector<uint8_t> Program::patched_code() const {
    std::vector<uint8_t> c = code;
    for (auto& p : patches) memcpy(c.data() + size_t(p.offset) * 4, p.code.data(), p.code.size() * 4);
    return c;
}

bool parse(const uint8_t* data, size_t size, Container& out, std::string& err) {
    View c{data, size};
    uint32_t parts;
    if (size < 32 || memcmp(data, "DXBC", 4) || !c.u32(28, parts) || parts > 64) return err = "not a DXBC container", false;
    for (uint32_t i = 0; i < parts; ++i) {
        uint32_t off, len;
        if (!c.u32(32 + 4 * i, off) || !c.u32(off + 4, len)) return err = "truncated part table", false;
        View part;
        if (!c.sub(uint64_t(off) + 8, len, part)) return err = "part out of bounds", false;
        if (!memcmp(data + off, "ISG1", 4) || !memcmp(data + off, "OSG1", 4)) {
            auto& sig = data[off] == 'I' ? out.inputs : out.outputs;
            uint32_t n = 0;
            part.u32(0, n);
            for (uint32_t e = 0; e < n && e < 64; ++e) {
                uint32_t w[8];
                if (8 + 32 * (e + 1) > part.n) break;
                memcpy(w, part.p + 8 + 32 * e, 32);
                Signature s{};
                s.index = w[2], s.system_value = w[3], s.component_type = w[4], s.reg = w[5];
                s.mask = uint8_t(w[6] & 0xff);
                for (size_t k = w[1]; k < part.n && part.p[k]; ++k) s.name.push_back(char(part.p[k]));
                sig.push_back(std::move(s));
            }
            continue;
        }
        if (memcmp(data + off, "DXIL", 4)) continue;
        part.u32(0, out.program_version);
        uint32_t bc_off, bc_len;
        if (part.u32(16, bc_off) && part.u32(20, bc_len))
            out.has_bitcode = bc_len > 64 && 8 + bc_off + 4 <= part.n && !memcmp(part.p + 8 + bc_off, "BC\xc0\xde", 4);
        if (part.n < XDAT_OFFSET + 12 || memcmp(part.p + XDAT_OFFSET, "XDAT", 4)) continue;
        uint32_t count;
        part.u32(XDAT_OFFSET + 4, count);
        if (count > MAX_XDAT) return err = "implausible XDAT count", false;
        for (uint32_t k = 0; k < count; ++k) {
            size_t at = XDAT_OFFSET + 12 + 12 * k;
            uint32_t kind, boff, bsize;
            if (!part.u32(at, kind) || !part.u32(at + 4, boff) || !part.u32(at + 8, bsize)) return err = "XDAT truncated", false;
            if (kind != KIND_NATIVE) continue;
            View blob;
            if (!part.sub(boff, bsize, blob)) return err = "XDAT entry out of bounds", false;
            std::vector<uint8_t> fssm;
            if (!unpack(blob, fssm, err) || !programs(fssm, out, err)) return false;
        }
    }
    return true;
}

}  // namespace kl::shader
