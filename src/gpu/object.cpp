#include "gpu/object.h"

#include <intrin.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "gpu/device.h"
#include "loader.h"
#include "log.h"

namespace kl::gpu {

std::string guid_text(const Iid* g) {
    if (!g) return "null";
    char b[64];
    snprintf(b, sizeof b, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2, g->Data3,
             g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6], g->Data4[7]);
    return b;
}

const char* kind_name(Kind k) {
    static const char* names[] = {"device", "dxgi", "queue", "allocator", "list", "heap", "resource", "fence",
                                  "descriptor heap", "root signature", "pipeline", "query heap",
                                  "command signature", "swap chain", "object"};
    return names[size_t(k)];
}

// ------------------------------------------------------ missing slots ----

namespace {
std::mutex g_seen_lock;
std::map<std::pair<int, int>, int> g_seen;

// One generated thunk per (kind, slot): mov ecx, kind; mov edx, slot; mov r8, [rsp]; jmp missing.
extern "C" HRESULT kl_gpu_missing(uint32_t kind, uint32_t slot, uintptr_t caller) {
    {
        std::lock_guard g(g_seen_lock);
        if (g_seen[{int(kind), int(slot)}]++ != 0) return E_NOTIMPL;
    }
    KL_WARN("gpu", "%s method 0x%x not implemented (called from %s)", kind_name(Kind(kind)), slot * 8,
            describe(caller).c_str());
    return E_NOTIMPL;
}

void* missing_thunk(Kind kind, size_t slot) {
    static std::mutex lock;
    static uint8_t* page = nullptr;
    static size_t used = 0;
    static std::map<std::pair<int, int>, void*> made;
    std::lock_guard g(lock);
    auto key = std::make_pair(int(kind), int(slot));
    if (auto it = made.find(key); it != made.end()) return it->second;
    constexpr size_t kPage = 64 << 10, kSize = 32;
    if (!page || used + kSize > kPage) {
        page = static_cast<uint8_t*>(VirtualAlloc(nullptr, kPage, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
        used = 0;
    }
    uint8_t* p = page + used;
    used += kSize;
    uint32_t k = uint32_t(kind), s = uint32_t(slot);
    uint64_t target = reinterpret_cast<uint64_t>(&kl_gpu_missing);
    size_t i = 0;
    p[i++] = 0xB9; memcpy(p + i, &k, 4); i += 4;                     // mov ecx, kind
    p[i++] = 0xBA; memcpy(p + i, &s, 4); i += 4;                     // mov edx, slot
    p[i++] = 0x4C; p[i++] = 0x8B; p[i++] = 0x04; p[i++] = 0x24;      // mov r8, [rsp]
    p[i++] = 0x48; p[i++] = 0xB8; memcpy(p + i, &target, 8); i += 8;  // mov rax, target
    p[i++] = 0xFF; p[i++] = 0xE0;                                    // jmp rax
    FlushInstructionCache(GetCurrentProcess(), p, kSize);
    made[key] = p;
    return p;
}

// ------------------------------------------------------- private data ----

std::mutex g_data_lock;
std::map<std::pair<const void*, std::string>, std::vector<uint8_t>> g_data;

std::string key(const Iid* g) { return guid_text(g); }

HRESULT WINAPI get_private_data(Object* o, const Iid* id, uint32_t* size, void* data) {
    if (!size) return E_INVALIDARG;
    std::lock_guard g(g_data_lock);
    auto it = g_data.find({o, key(id)});
    if (it == g_data.end()) {
        *size = 0;
        return DXGI_ERROR_NOT_FOUND;
    }
    if (!data) {
        *size = uint32_t(it->second.size());
        return S_OK;
    }
    if (*size < it->second.size()) {
        *size = uint32_t(it->second.size());
        return DXGI_ERROR_MORE_DATA;
    }
    *size = uint32_t(it->second.size());
    memcpy(data, it->second.data(), it->second.size());
    return S_OK;
}
HRESULT WINAPI set_private_data(Object* o, const Iid* id, uint32_t size, const void* data) {
    std::lock_guard g(g_data_lock);
    if (!data) {
        g_data.erase({o, key(id)});
        return S_OK;
    }
    auto* b = static_cast<const uint8_t*>(data);
    g_data[{o, key(id)}] = std::vector<uint8_t>(b, b + size);
    return S_OK;
}
HRESULT WINAPI set_private_interface(Object*, const Iid*, const void*) { return S_OK; }
HRESULT WINAPI set_name(Object* o, const wchar_t* name) {
    KL_TRACE("gpu", "object %p named %ls", (void*)o, name ? name : L"");
    return S_OK;
}
HRESULT WINAPI get_device(Object*, const Iid* iid, void** out) { return device_query(iid, out); }

}  // namespace

void** make_table(Kind kind, size_t n) {
    auto** t = new void*[n];
    for (size_t i = 0; i < n; ++i) t[i] = missing_thunk(kind, i);
    return t;
}

uint32_t WINAPI add_ref(Object* o) { return ++o->refs; }
uint32_t WINAPI release_static(Object* o) {
    uint32_t r = o->refs.load();
    if (r) r = --o->refs;
    return r;
}
HRESULT WINAPI ignore() { return S_OK; }

void fill_object(void** t, void* query_interface, void* release) {
    t[0] = query_interface;
    t[1] = reinterpret_cast<void*>(&add_ref);
    t[2] = release ? release : reinterpret_cast<void*>(&release_static);
    t[3] = reinterpret_cast<void*>(&get_private_data);
    t[4] = reinterpret_cast<void*>(&set_private_data);
    t[5] = reinterpret_cast<void*>(&set_private_interface);
    t[6] = reinterpret_cast<void*>(&set_name);
}

void fill_child(void** t) { t[7] = reinterpret_cast<void*>(&get_device); }

HRESULT answer(Kind kind, const Iid* iid, void* found, void** out) {
    if (!out) return E_POINTER;
    if (!found) {
        KL_WARN("gpu", "%s: no interface %s", kind_name(kind), guid_text(iid).c_str());
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ++static_cast<Object*>(found)->refs;
    *out = found;
    return S_OK;
}

}  // namespace kl::gpu
