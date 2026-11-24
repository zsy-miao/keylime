#include "gpu/texture.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#include "addrinterface.h"
#include "log.h"

namespace kl::gpu::texture {
namespace {

// Family/revision of the Navi 2x (GFX10.3) part the console's GPU is.
constexpr UINT_32 FAMILY_NV = 143;   // amdgpu_asic_addr.h FAMILY_NV
constexpr UINT_32 NAVI21_A0 = 0x28;  // NV_SIENNA_CICHLID_P_A0
// GB_ADDR_CONFIG: 16 pipes, 256 B pipe interleave, 16 packers (Navi 21-class).
// ASSUMPTION: the console's exact pipe/packer setup is unknown; sizes of the
// non-XOR modes do not depend on it, XOR'd addressing does.
constexpr UINT_32 GB_ADDR_CONFIG = 0x00000444;

void* ADDR_API alloc_cb(const ADDR_ALLOCSYSMEM_INPUT* in) { return malloc(in->sizeInBytes); }
ADDR_E_RETURNCODE ADDR_API free_cb(const ADDR_FREESYSMEM_INPUT* in) {
    free(in->pVirtAddr);
    return ADDR_OK;
}

ADDR_HANDLE handle() {
    static ADDR_HANDLE h = [] {
        ADDR_CREATE_INPUT in{};
        ADDR_CREATE_OUTPUT out{};
        in.size = sizeof in;
        out.size = sizeof out;
        in.chipEngine = 0x0000000D;  // CIASICIDGFXENGINE_ARCTICISLAND
        in.chipFamily = FAMILY_NV;
        in.chipRevision = NAVI21_A0;
        in.callbacks.allocSysMem = alloc_cb;
        in.callbacks.freeSysMem = free_cb;
        in.regValue.gbAddrConfig = GB_ADDR_CONFIG;
        in.createFlags.useTileIndex = 0;
        if (AddrCreate(&in, &out) != ADDR_OK) {
            KL_ERROR("gpu", "addrlib: AddrCreate failed");
            return ADDR_HANDLE(nullptr);
        }
        return out.hLib;
    }();
    return h;
}

}  // namespace

bool element(uint32_t f, uint32_t& bytes, uint32_t& block) {
    block = 1;
    switch (f) {
        case 1: case 2: case 3: case 4: bytes = 16; return true;          // R32G32B32A32
        case 5: case 6: case 7: case 8: bytes = 12; return true;          // R32G32B32
        case 9: case 10: case 11: case 12: case 13: case 14:              // R16G16B16A16
        case 15: case 16: case 17: case 18:                               // R32G32
        case 19: case 20: case 21: case 22: bytes = 8; return true;       // R32G8X24 family
        case 23: case 24: case 25: case 26: case 27: case 28: case 29: case 30: case 31: case 32:
        case 33: case 34: case 35: case 36: case 37: case 38: case 39: case 40: case 41: case 42:
        case 43: case 44: case 45: case 46: case 47: bytes = 4; return true;  // 32-bit formats
        case 48: case 49: case 50: case 51: case 52: case 53: case 54: case 55: case 56: case 57:
        case 58: case 59: bytes = 2; return true;                         // R8G8 / R16 / D16
        case 60: case 61: case 62: case 63: case 64: case 65: bytes = 1; return true;  // R8 / A8
        case 67: case 68: case 69: bytes = 4; return true;                // RGB9E5, packed
        case 70: case 71: case 72: case 79: case 80: case 81: bytes = 8, block = 4; return true;  // BC1, BC4
        case 73: case 74: case 75: case 76: case 77: case 78: case 82: case 83: case 84:
            bytes = 16, block = 4; return true;                           // BC2, BC3, BC5
        case 85: case 86: case 87: case 88: case 89: case 90: case 91: case 92: case 93:
            bytes = f <= 86 ? 2 : 4; return true;                         // B5G6R5, BGRA, ...
        case 94: case 95: case 96: case 97: case 98: case 99:
            bytes = 16, block = 4; return true;                           // BC6H, BC7
        case 115: bytes = 2; return true;                                 // B4G4R4A4
        case 116: case 117: bytes = 4; return true;                       // console 10-bit float color
        case 118: case 119: case 120: bytes = 3; return true;             // console D16S8 (two planes)
        case 189: bytes = 4; return true;                                 // console signed 10-bit color
        case 190: bytes = 1; return true;                                 // console two 4-bit channels
        case 191: bytes = 4; return true;                                 // console 24/8 color
        default: return false;
    }
}

uint32_t full_mip_count(uint32_t w, uint32_t h, uint32_t d) {
    uint32_t m = w > h ? w : h;
    if (d > m) m = d;
    uint32_t n = 1;
    while (m > 1) m >>= 1, ++n;
    return n;
}

uint32_t default_swizzle(Dim dim, bool depth, bool rt) {
    if (dim == Dim::Tex1d) return ADDR_SW_LINEAR;
    if (depth) return ADDR_SW_64KB_Z_X;
    if (rt) return ADDR_SW_64KB_R_X;
    return ADDR_SW_4KB_S;
}

bool layout(const Desc& d, Layout& out) {
    ADDR_HANDLE h = handle();
    if (!h) return false;
    uint32_t bytes, block;
    if (!element(d.format, bytes, block)) return false;
    out = {};
    out.bytes_per_element = bytes;
    out.block = block;
    uint32_t mips = d.mips ? d.mips : full_mip_count(d.width, d.height, d.dim == Dim::Tex3d ? d.depth_or_array : 1);

    ADDR2_COMPUTE_SURFACE_INFO_INPUT in{};
    ADDR2_COMPUTE_SURFACE_INFO_OUTPUT o{};
    std::vector<ADDR2_MIP_INFO> mi(mips);
    in.size = sizeof in;
    o.size = sizeof o;
    o.pMipInfo = mi.data();
    in.resourceType = d.dim == Dim::Tex1d ? ADDR_RSRC_TEX_1D : d.dim == Dim::Tex3d ? ADDR_RSRC_TEX_3D : ADDR_RSRC_TEX_2D;
    in.swizzleMode = AddrSwizzleMode(d.swizzle);
    // Block formats: addrlib works in elements; pass the element size and block-divided dims.
    in.bpp = bytes == 3 ? 24 : bytes * 8;
    in.format = ADDR_FMT_INVALID;
    in.width = (d.width + block - 1) / block;
    in.height = d.dim == Dim::Tex1d ? 1 : (d.height + block - 1) / block;
    in.numSlices = d.depth_or_array ? d.depth_or_array : 1;
    in.numMipLevels = mips;
    in.numSamples = d.samples ? d.samples : 1;
    in.numFrags = in.numSamples;
    in.flags.texture = 1;
    in.flags.view3dAs2dArray = 0;
    if (Addr2ComputeSurfaceInfo(h, &in, &o) != ADDR_OK) return false;
    out.size = o.surfSize;
    out.alignment = o.baseAlign;
    out.slice_size = o.sliceSize;
    out.mips.resize(mips);
    for (uint32_t i = 0; i < mips; ++i) {
        out.mips[i].offset = mi[i].offset;
        out.mips[i].pitch = mi[i].pitch;
        out.mips[i].height = mi[i].height;
        out.mips[i].depth = mi[i].depth;
        out.mips[i].in_tail = o.mipChainInTail || (o.firstMipIdInTail <= i);
    }
    return true;
}

}  // namespace kl::gpu::texture
