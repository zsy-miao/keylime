// keylime: xgameruntime.dll exports and the QueryApiImpl dispatch (see runtime.h).
#include "hle/xgr/xgr.h"

#include <atomic>
#include <mutex>
#include <set>

#include "hle/hle.h"
#include "log.h"
#include "loader.h"
#include <intrin.h>

namespace kl::xgr {
namespace {

std::mutex g_lock;
std::vector<Interface>& interfaces() {
    static std::vector<Interface> v;
    return v;
}

std::string guid_text(const GUID& g) {
    char b[64];
    snprintf(b, sizeof b, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g.Data1, g.Data2, g.Data3, g.Data4[0], g.Data4[1],
             g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
    return b;
}

ULONG WINAPI add_ref(Object*) { return 1; }
HRESULT WINAPI query_interface(Object* o, const GUID*, void** out) {
    if (out) *out = nullptr;
    KL_WARN("xgr", "%s: QueryInterface on a runtime object", o->area);
    return E_NOINTERFACE;
}

// Logging stubs, one per slot (the slot is in the function).
template <size_t S>
uint64_t WINAPI missing(Object* o) {
    static std::atomic<int> times{0};
    if (times++ < 4)
        KL_WARN("xgr", "runtime method not implemented: %s +0x%zx (called from %s)", o ? o->area : "?", S * 8,
                describe(uintptr_t(_ReturnAddress())).c_str());
    return 0xDEAD000080004001ull;
}
template <size_t... I>
void* const* stubs(std::index_sequence<I...>) {
    static void* const t[] = {reinterpret_cast<void*>(&missing<I>)...};
    return t;
}

HRESULT WINAPI XGameRuntimeInitialize() {
    KL_INFO("xgr", "XGameRuntimeInitialize");
    return S_OK;
}
void WINAPI XGameRuntimeUninitialize() {}
HRESULT WINAPI InitializeApiImpl(uint64_t v, uint64_t b) {
    KL_INFO("xgr", "InitializeApiImpl(%llu, %llu)", (unsigned long long)v, (unsigned long long)b);
    return S_OK;
}
HRESULT WINAPI InitializeApiImplEx(uint64_t v, uint64_t b, uint64_t) { return InitializeApiImpl(v, b); }
HRESULT WINAPI InitializeApiImplEx2(uint64_t v, uint64_t b, uint64_t, const void*) { return InitializeApiImpl(v, b); }
void WINAPI UninitializeApiImpl() {}

HRESULT WINAPI QueryApiImpl(const GUID* provider, const GUID* iid, void** out) {
    if (!provider || !iid) return NOT_SUPPORTED;
    {
        std::lock_guard g(g_lock);
        for (auto& i : interfaces())
            if (IsEqualGUID(i.provider, *provider) && IsEqualGUID(i.iid, *iid)) {
                if (!out) return E_INVALIDARG_;
                *out = i.object;
                return S_OK;
            }
        static std::set<std::string> seen;
        std::string key = guid_text(*provider) + "/" + guid_text(*iid);
        if (seen.insert(key).second) KL_WARN("xgr", "runtime API area not implemented: provider %s", key.c_str());
    }
    return NOT_SUPPORTED;
}

void WINAPI XErrorReport(HRESULT hr, const char* message) {
    KL_WARN("xgr", "XErrorReport %08lx: %s", hr, message ? message : "");
}

}  // namespace

void** make_table(const char* area, size_t n) {
    (void)area;
    static void* const* s = stubs(std::make_index_sequence<128>{});
    auto** t = new void*[n < 128 ? 128 : n];
    for (size_t i = 0; i < 128; ++i) t[i] = s[i];
    t[0] = reinterpret_cast<void*>(&query_interface);
    t[1] = t[2] = reinterpret_cast<void*>(&add_ref);
    return t;
}
void set(void** t, size_t offset, void* fn) { t[offset / 8] = fn; }

void add_interfaces(std::initializer_list<Interface> list) {
    std::lock_guard g(g_lock);
    for (auto& i : list) interfaces().push_back(i);
}

HRESULT put_string(const char* text, size_t size, char* buffer, size_t* used) {
    size_t len = strlen(text) + 1;
    if (!buffer || size < len) return E_NOT_SUFFICIENT_BUFFER_;
    memcpy(buffer, text, len);
    if (used) *used = len;
    return S_OK;
}

uint64_t token() {
    static std::atomic<uint64_t> next{1};
    return next++;
}

KL_MODULE("xgameruntime", (std::vector<std::string>{"xgameruntime.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(XGameRuntimeInitialize), KL_EXPORT(XGameRuntimeUninitialize), KL_EXPORT(InitializeApiImpl),
                                    KL_EXPORT(InitializeApiImplEx), KL_EXPORT(InitializeApiImplEx2), KL_EXPORT(UninitializeApiImpl),
                                    KL_EXPORT(QueryApiImpl), KL_EXPORT(XErrorReport)}));

}  // namespace kl::xgr
