// keylime: loader-facing kernel APIs whose console meaning differs.
//  - GetModuleHandle(NULL) / GetProcAddress(NULL) mean the title's exe, not keylime.exe.
//  - Console-only DLLs we implement (xgameruntime.dll, ...) load as pseudo handles.
//  - The title's own DLLs load through our loader (mapped + bound by us).
#include <windows.h>

#include <string>

#include "hle/hle.h"
#include "loader.h"
#include "log.h"
#include "runtime.h"

namespace kl { LPTOP_LEVEL_EXCEPTION_FILTER title_unhandled_filter(LPTOP_LEVEL_EXCEPTION_FILTER f); }
namespace kl::hle {
namespace {

std::string narrow(const char* s) { return s ? std::string(s) : std::string(); }
std::string narrow(const wchar_t* s) {
    if (!s) return {};
    int n = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
    std::string out(size_t(n > 0 ? n - 1 : 0), '\0');
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, s, -1, out.data(), n, nullptr, nullptr);
    return out;
}

// "C:\x\Foo.DLL" -> "foo.dll"; adds ".dll" when there is no extension.
std::string file_key(std::string name) {
    auto slash = name.find_last_of("\\/");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    name = lower(name);
    if (name.find('.') == std::string::npos) name += ".dll";
    return name;
}

HMODULE main_module() { return loader().main() ? loader().main()->module : nullptr; }

// A pseudo handle: the Module object itself (never a valid PE base).
Module* pseudo_module(HMODULE h) {
    for (auto* m : modules())
        if (reinterpret_cast<HMODULE>(m) == h) return m;
    return nullptr;
}

HMODULE find_module(const std::string& raw, bool load) {
    auto key = file_key(raw);
    if (auto* img = loader().find(key)) return img->module;
    for (auto* m : modules())
        if (!host_allows(key) && m->serves(key)) return reinterpret_cast<HMODULE>(m);
    if (load)
        if (HMODULE h = loader().load_late(key)) return h;
    return nullptr;
}

bool runs_code(DWORD flags) { return (flags & (0x1 | 0x2 | 0x20 | 0x40)) == 0; }

HMODULE WINAPI kGetModuleHandleW(LPCWSTR name) {
    if (!name) return main_module();
    if (HMODULE h = find_module(narrow(name), false)) return h;
    return GetModuleHandleW(name);
}
HMODULE WINAPI kGetModuleHandleA(LPCSTR name) {
    if (!name) return main_module();
    if (HMODULE h = find_module(narrow(name), false)) return h;
    return GetModuleHandleA(name);
}
BOOL WINAPI kGetModuleHandleExW(DWORD flags, LPCWSTR name, HMODULE* out) {
    if (!(flags & GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS)) {
        if (!name) return *out = main_module(), TRUE;
        if (HMODULE h = find_module(narrow(name), false)) return *out = h, TRUE;
    }
    return GetModuleHandleExW(flags, name, out);
}

const LoadedImage* image_of(HMODULE h) {
    if (!h) return loader().main();
    for (auto& i : loader().images())
        if (i->module == h) return i.get();
    return nullptr;
}

DWORD WINAPI kGetModuleFileNameW(HMODULE h, LPWSTR out, DWORD size) {
    auto* img = image_of(h);
    if (!img) return GetModuleFileNameW(h, out, size);
    // The original file, not our cache copy.
    std::wstring path = (runtime().title_dir / runtime().file_of(img)).wstring();
    if (!size) return SetLastError(ERROR_INSUFFICIENT_BUFFER), 0;
    size_t n = path.size() < size - 1 ? path.size() : size - 1;
    wmemcpy(out, path.c_str(), n);
    out[n] = 0;
    if (n < path.size()) return SetLastError(ERROR_INSUFFICIENT_BUFFER), size;
    return DWORD(n);
}
DWORD WINAPI kGetModuleFileNameA(HMODULE h, LPSTR out, DWORD size) {
    if (!image_of(h)) return GetModuleFileNameA(h, out, size);
    wchar_t w[MAX_PATH * 2];
    DWORD n = kGetModuleFileNameW(h, w, MAX_PATH * 2);
    if (!n) return 0;
    std::string s = narrow(w);
    if (!size) return SetLastError(ERROR_INSUFFICIENT_BUFFER), 0;
    size_t c = s.size() < size - 1 ? s.size() : size - 1;
    memcpy(out, s.c_str(), c);
    out[c] = 0;
    return c < s.size() ? (SetLastError(ERROR_INSUFFICIENT_BUFFER), size) : DWORD(c);
}

HMODULE WINAPI kLoadLibraryExW(LPCWSTR name, HANDLE file, DWORD flags) {
    if (name)
        if (HMODULE h = find_module(narrow(name), runs_code(flags))) {
            KL_INFO("kernel", "LoadLibrary(%s) -> title/emulated %p", narrow(name).c_str(), (void*)h);
            return h;
        }
    HMODULE h = LoadLibraryExW(name, file, flags);
    KL_INFO("kernel", "LoadLibrary(%s) -> host %p", narrow(name).c_str(), (void*)h);
    return h;
}
HMODULE WINAPI kLoadLibraryExA(LPCSTR name, HANDLE file, DWORD flags) {
    if (name)
        if (HMODULE h = find_module(name, runs_code(flags))) {
            KL_INFO("kernel", "LoadLibrary(%s) -> title/emulated %p", name, (void*)h);
            return h;
        }
    HMODULE h = LoadLibraryExA(name, file, flags);
    KL_INFO("kernel", "LoadLibrary(%s) -> host %p", name ? name : "", (void*)h);
    return h;
}
HMODULE WINAPI kLoadLibraryW(LPCWSTR name) { return kLoadLibraryExW(name, nullptr, 0); }
HMODULE WINAPI kLoadLibraryA(LPCSTR name) { return kLoadLibraryExA(name, nullptr, 0); }
HMODULE WINAPI kLoadPackagedLibrary(LPCWSTR name, DWORD) { return kLoadLibraryExW(name, nullptr, 0); }

FARPROC WINAPI kGetProcAddress(HMODULE h, LPCSTR name) {
    if (!h) h = main_module();
    const bool by_ordinal = reinterpret_cast<uintptr_t>(name) < 0x10000;
    std::string sym = by_ordinal ? "#" + std::to_string(reinterpret_cast<uintptr_t>(name)) : std::string(name);
    if (Module* m = pseudo_module(h)) {
        for (auto& e : m->exports)
            if (sym == e.name) return reinterpret_cast<FARPROC>(e.address);
        KL_WARN("kernel", "GetProcAddress(%s, %s): not implemented -> stub", m->name, sym.c_str());
        return reinterpret_cast<FARPROC>(make_stub(m->dlls.empty() ? m->name : m->dlls[0], sym));
    }
    // A host DLL whose function we override must not be reachable around us.
    if (!image_of(h) || h != main_module()) {
        wchar_t path[MAX_PATH];
        if (GetModuleFileNameW(h, path, MAX_PATH)) {
            auto file = lower(std::filesystem::path(path).filename().string());
            if (void* p = lookup(file, sym)) return reinterpret_cast<FARPROC>(p);
        }
    }
    FARPROC p = GetProcAddress(h, name);
    if (!p) KL_TRACE("kernel", "GetProcAddress(%p, %s) -> null", (void*)h, sym.c_str());
    return p;
}

BOOL WINAPI kFreeLibrary(HMODULE h) {
    if (pseudo_module(h) || image_of(h)) return TRUE;
    return FreeLibrary(h);
}

void WINAPI kOutputDebugStringA(LPCSTR s) {
    if (!s) return;
    std::string t(s);
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
    KL_INFO("title", "%s", t.c_str());
}
void WINAPI kOutputDebugStringW(LPCWSTR s) {
    if (!s) return;
    kOutputDebugStringA(narrow(s).c_str());
}

void WINAPI kExitProcess(UINT code) {
    KL_WARN("kernel", "title ExitProcess(%u) from %s", code, describe(uintptr_t(_ReturnAddress())).c_str());
    ExitProcess(code);
}
BOOL WINAPI kTerminateProcess(HANDLE p, UINT code) {
    KL_WARN("kernel", "title TerminateProcess(%u) from %s", code, describe(uintptr_t(_ReturnAddress())).c_str());
    return TerminateProcess(p, code);
}

LPTOP_LEVEL_EXCEPTION_FILTER WINAPI kSetUnhandledExceptionFilter(LPTOP_LEVEL_EXCEPTION_FILTER f) {
    return kl::title_unhandled_filter(f);
}

}  // namespace

KL_MODULE("kernel",
          (std::vector<std::string>{"kernel32.dll", "kernelbase.dll", "api-ms-win-core-libraryloader-*",
                                    "api-ms-win-core-processthreads-*", "api-ms-win-core-debug-*",
                                    "api-ms-win-core-errorhandling-*"}),
          (std::vector<Export>{
              KL_EXPORT_AS("GetModuleHandleW", kGetModuleHandleW), KL_EXPORT_AS("GetModuleHandleA", kGetModuleHandleA),
              KL_EXPORT_AS("GetModuleHandleExW", kGetModuleHandleExW),
              KL_EXPORT_AS("GetModuleFileNameW", kGetModuleFileNameW), KL_EXPORT_AS("GetModuleFileNameA", kGetModuleFileNameA),
              KL_EXPORT_AS("LoadLibraryExW", kLoadLibraryExW), KL_EXPORT_AS("LoadLibraryExA", kLoadLibraryExA),
              KL_EXPORT_AS("LoadLibraryW", kLoadLibraryW), KL_EXPORT_AS("LoadLibraryA", kLoadLibraryA),
              KL_EXPORT_AS("LoadPackagedLibrary", kLoadPackagedLibrary),
              KL_EXPORT_AS("GetProcAddress", kGetProcAddress), KL_EXPORT_AS("FreeLibrary", kFreeLibrary),
              KL_EXPORT_AS("OutputDebugStringA", kOutputDebugStringA), KL_EXPORT_AS("OutputDebugStringW", kOutputDebugStringW),
              KL_EXPORT_AS("ExitProcess", kExitProcess), KL_EXPORT_AS("TerminateProcess", kTerminateProcess),
              KL_EXPORT_AS("SetUnhandledExceptionFilter", kSetUnhandledExceptionFilter),
          }));

}  // namespace kl::hle
