// keylime: high-level emulation registry. Each module lists the DLL names it
// answers for and the functions it implements; anything not listed falls
// through to the host (allow-listed Windows DLLs) or to a logging stub.
#pragma once
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace kl::hle {

struct Export {
    const char* name;  // imported name, or "#<ordinal>"
    void* address;
};

struct Module {
    const char* name;
    std::vector<std::string> dlls;  // lower case; trailing '*' = prefix
    std::vector<Export> exports;
    bool serves(const std::string& dll) const;
};

struct Registrar {
    Registrar(Module* m);
};

const std::vector<Module*>& modules();
void* lookup(const std::string& dll, const std::string& name);
bool serves(const std::string& dll);
std::string lower(std::string s);

// Host pass-through ------------------------------------------------------
bool host_allows(const std::string& dll);
void* host_resolve(const std::string& dll, const std::string& name, uint16_t ordinal);
void* host_function(const char* dll, const char* name);

// Unimplemented calls: a 32-byte stub per symbol that logs and returns 0.
void* make_stub(const std::string& dll, const std::string& symbol);

}  // namespace kl::hle

#define KL_CAT2(a, b) a##b
#define KL_CAT(a, b) KL_CAT2(a, b)
#define KL_EXPORT(fn) ::kl::hle::Export{#fn, reinterpret_cast<void*>(&fn)}
#define KL_EXPORT_AS(name, fn) ::kl::hle::Export{name, reinterpret_cast<void*>(&fn)}
// KL_MODULE("kernel", {"kernel32.dll", "api-ms-win-core-*"}, {KL_EXPORT(Foo), ...});
#define KL_MODULE(label, dll_list, export_list)                                          \
    static ::kl::hle::Module KL_CAT(kl_module_, __LINE__){label, dll_list, export_list}; \
    static ::kl::hle::Registrar KL_CAT(kl_registrar_, __LINE__){&KL_CAT(kl_module_, __LINE__)}
