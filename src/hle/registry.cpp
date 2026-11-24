#include "hle/hle.h"

#include <windows.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>

#include "log.h"

namespace kl::hle {

std::string lower(std::string s) {
    for (auto& c : s) c = char(tolower((unsigned char)c));
    return s;
}

static bool match(const std::string& pattern, const std::string& dll) {
    if (!pattern.empty() && pattern.back() == '*')
        return dll.compare(0, pattern.size() - 1, pattern, 0, pattern.size() - 1) == 0;
    return dll == pattern;
}

bool Module::serves(const std::string& dll) const {
    auto d = lower(dll);
    for (auto& p : dlls)
        if (match(p, d)) return true;
    return false;
}

static std::vector<Module*>& registry() {
    static std::vector<Module*> r;
    return r;
}

Registrar::Registrar(Module* m) { registry().push_back(m); }

const std::vector<Module*>& modules() { return registry(); }

void* lookup(const std::string& dll, const std::string& name) {
    for (auto* m : registry())
        if (m->serves(dll))
            for (auto& e : m->exports)
                if (name == e.name) return e.address;
    return nullptr;
}

bool serves(const std::string& dll) {
    for (auto* m : registry())
        if (m->serves(dll)) return true;
    return false;
}

// ---------------------------------------------------------------- host ----

// Windows API sets and DLLs that behave the same on console and PC. Anything
// whose meaning differs (memory, files, loader) is overridden by a module,
// and modules always win.
static const char* kHostAllow[] = {
    "api-ms-win-core-*", "api-ms-win-eventing-*", "api-ms-win-security-*", "api-ms-win-crt-*",
    "kernel32.dll", "kernelbase.dll", "ntdll.dll", "ucrtbase.dll", "vcruntime140*", "msvcp140*",
    "ws2_32.dll", "bcrypt.dll", "crypt32.dll", "ncrypt.dll", "winhttp.dll", "sspicli.dll",
    "cabinet.dll", "avrt.dll", "ext-ms-win-rtcore-ntuser-*", "api-ms-win-shcore-*",
    "msvcp_win.dll", "ole32.dll", "advapi32.dll", "iphlpapi.dll", "dbghelp.dll", "xaudio2_9.dll",
    "mfplat.dll", "mfreadwrite.dll",
};

bool host_allows(const std::string& dll) {
    auto d = lower(dll);
    for (auto* p : kHostAllow)
        if (match(p, d)) return true;
    return false;
}

static HMODULE host_module(const std::string& dll) {
    static std::mutex lock;
    static std::map<std::string, HMODULE> cache;
    std::lock_guard g(lock);
    auto key = lower(dll);
    if (auto it = cache.find(key); it != cache.end()) return it->second;
    // System32 only: never a same-named DLL from the game folder or PATH.
    HMODULE h = LoadLibraryExA(key.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    cache[key] = h;
    return h;
}

void* host_resolve(const std::string& dll, const std::string& name, uint16_t ordinal) {
    if (!host_allows(dll)) return nullptr;
    HMODULE m = host_module(dll);
    if (!m) return nullptr;
    FARPROC p = name.empty() ? GetProcAddress(m, MAKEINTRESOURCEA(ordinal)) : GetProcAddress(m, name.c_str());
    return reinterpret_cast<void*>(p);
}

void* host_function(const char* dll, const char* name) {
    HMODULE m = host_module(dll);
    return m ? reinterpret_cast<void*>(GetProcAddress(m, name)) : nullptr;
}

// --------------------------------------------------------------- stubs ----

namespace {
struct Site {
    std::string dll, symbol;
    std::atomic<uint64_t> calls{0};
};

extern "C" uint64_t kl_unimplemented(Site* site) {
    uint64_t n = ++site->calls;
    if (n <= 3 || (n & (n - 1)) == 0)
        KL_WARN("stub", "unimplemented %s!%s (call %llu) -> 0", site->dll.c_str(), site->symbol.c_str(),
                (unsigned long long)n);
    return 0;
}

struct StubPage {
    uint8_t* base = nullptr;
    size_t used = 0;
};
}  // namespace

void* make_stub(const std::string& dll, const std::string& symbol) {
    static std::mutex lock;
    static StubPage page;
    static std::map<std::string, void*> made;
    std::lock_guard g(lock);
    std::string key = lower(dll) + "!" + symbol;
    if (auto it = made.find(key); it != made.end()) return it->second;
    constexpr size_t kPage = 64 * 1024, kStub = 32;
    if (!page.base || page.used + kStub > kPage) {
        page.base = static_cast<uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
        page.used = 0;
        if (!page.base) return nullptr;
    }
    auto* site = new Site{dll, symbol};
    uint8_t* p = page.base + page.used;
    page.used += kStub;
    // mov rcx, site ; mov rax, kl_unimplemented ; jmp rax  (title's return address stays on the stack)
    uint64_t s = reinterpret_cast<uint64_t>(site), h = reinterpret_cast<uint64_t>(&kl_unimplemented);
    p[0] = 0x48; p[1] = 0xB9; std::memcpy(p + 2, &s, 8);
    p[10] = 0x48; p[11] = 0xB8; std::memcpy(p + 12, &h, 8);
    p[20] = 0xFF; p[21] = 0xE0;
    FlushInstructionCache(GetCurrentProcess(), p, kStub);
    made[key] = p;
    return p;
}

}  // namespace kl::hle
