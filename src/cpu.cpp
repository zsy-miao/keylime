// keylime: AMD-only instructions on Intel CPUs. The console's CPU is Zen 2;
// the title uses instructions Intel lacks. A first vectored handler does
// their job: clzero (0f 01 fc) zeroes the 64-byte line at rax; monitorx /
// mwaitx (0f 01 fa / fb) are wait hints (mwaitx yields); SSE4a extrq /
// insertq (66/f2 0f 78/79) are computed on the saved xmm registers.
// (After VIctory crash.rs; counted, so a hot one shows and can be patched.)
#include <windows.h>

#include <atomic>
#include <cstring>

#include "log.h"

namespace kl::cpu {
namespace {

std::atomic<uint64_t> g_clzero{0}, g_wait{0}, g_sse4a{0};

void note(std::atomic<uint64_t>& c, const char* what) {
    uint64_t n = ++c;
    if (n == 1 || n == 1000 || n == 100000 || n == 10000000) KL_INFO("cpu", "emulated %s x%llu", what, (unsigned long long)n);
}

size_t sse4a(const uint8_t* code, M128A* xmm) {
    auto b = [&](size_t i) { return code[i]; };
    uint8_t prefix = b(0), rex = 0;
    size_t at = 1;
    if ((b(1) & 0xf0) == 0x40) rex = b(1), at = 2;
    if ((prefix != 0x66 && prefix != 0xf2) || b(at) != 0x0f || (b(at + 1) != 0x78 && b(at + 1) != 0x79)) return 0;
    uint8_t modrm = b(at + 2);
    if ((modrm >> 6) != 3) return 0;
    unsigned dst = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    unsigned src = (modrm & 7) | ((rex & 1) << 3);
    bool imm = b(at + 1) == 0x78;
    uint8_t f0, f1;
    size_t len;
    if (imm) {
        f0 = b(at + 3), f1 = b(at + 4), len = at + 5;
    } else {
        uint64_t f = prefix == 0x66 ? xmm[src].Low : uint64_t(xmm[src].High);
        f0 = uint8_t(f), f1 = uint8_t(f >> 8), len = at + 3;
    }
    unsigned n = f0 & 63, bit = f1 & 63;
    unsigned bits = n ? n : 64;
    uint64_t mask = bits == 64 ? ~0ull : (1ull << bits) - 1;
    if (prefix == 0x66) {  // extrq
        if (imm && ((modrm >> 3) & 7) != 0) return 0;
        unsigned t = imm ? src : dst;
        xmm[t].Low = (xmm[t].Low >> bit) & mask;
    } else {  // insertq
        uint64_t v = (xmm[src].Low & mask) << bit;
        xmm[dst].Low = (xmm[dst].Low & ~(mask << bit)) | v;
    }
    return len;
}

LONG CALLBACK handler(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION) return EXCEPTION_CONTINUE_SEARCH;
    auto* c = ep->ContextRecord;
    auto* code = reinterpret_cast<const uint8_t*>(c->Rip);
    if (code[0] == 0x0f && code[1] == 0x01) {
        if (code[2] == 0xfc) {
            memset(reinterpret_cast<void*>(c->Rax & ~63ull), 0, 64);
            note(g_clzero, "clzero");
            c->Rip += 3;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (code[2] == 0xfa || code[2] == 0xfb) {
            if (code[2] == 0xfb) SwitchToThread();
            note(g_wait, "monitorx/mwaitx");
            c->Rip += 3;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if (size_t len = sse4a(code, c->FltSave.XmmRegisters)) {
        note(g_sse4a, "sse4a");
        c->Rip += len;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

}  // namespace

void install() { AddVectoredExceptionHandler(1, handler); }

}  // namespace kl::cpu
