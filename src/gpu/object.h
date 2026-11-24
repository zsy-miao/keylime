// keylime GPU: what every console graphics object starts with (as the
// title's inlined D3D12.X code reads it): a function table pointer, a word
// of bookkeeping bits (device index in bits 0-2; all zero) and a reference
// count. Tables begin with IUnknown's three slots, then (for D3D12 objects)
// GetPrivateData, SetPrivateData, SetPrivateDataInterface, SetName, and for
// device children GetDevice at 0x38.
//
// Every table is longer than the interface we know and every slot exists:
// an unwritten one logs its offset and the caller, then returns E_NOTIMPL,
// so a run shows exactly what to write next.
#pragma once
#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>

namespace kl::gpu {

struct Object {
    const void* const* functions;
    uint32_t bits;
    std::atomic<uint32_t> refs;
};
static_assert(sizeof(Object) == 16);

using Iid = GUID;
constexpr Iid guid(uint32_t a, uint16_t b, uint16_t c, std::initializer_list<uint8_t> d) {
    Iid g{a, b, c, {}};
    int i = 0;
    for (uint8_t x : d) g.Data4[i++] = x;
    return g;
}
inline bool same(const Iid* a, const Iid& b) { return a && memcmp(a, &b, sizeof b) == 0; }
std::string guid_text(const Iid* g);

// The kind of object a table belongs to, for the "not implemented" log.
enum class Kind : uint8_t {
    Device, Dxgi, Queue, Allocator, List, Heap, Resource, Fence, DescriptorHeap, RootSignature,
    Pipeline, QuerySet, CommandSignature, SwapChain, Other, Count
};
const char* kind_name(Kind k);

constexpr size_t kMaxSlots = 192;

// A table of `n` slots, all logging stubs for `kind`; the caller fills what it has.
// Tables are built once and never freed.
void** make_table(Kind kind, size_t n);
// Fills IUnknown (refcount on Object, never frees unless `release` is given) and the
// D3D12 object methods (private data + name) into slots 0..6.
void fill_object(void** t, void* query_interface, void* release = nullptr);
// GetDevice at 0x38 for device children.
void fill_child(void** t);

// Default IUnknown pieces.
uint32_t WINAPI add_ref(Object* o);
uint32_t WINAPI release_static(Object* o);
HRESULT WINAPI ignore();  // accepts and ignores

// Calls slot `offset` of an object (for host code and tests).
template <class F>
F slot(const void* obj, size_t offset) {
    auto* o = static_cast<const Object*>(obj);
    return reinterpret_cast<F>(const_cast<void*>(o->functions[offset / 8]));
}

// Answers a QueryInterface-like call: `found` (AddRef'd) or null + E_NOINTERFACE.
HRESULT answer(Kind kind, const Iid* iid, void* found, void** out);

}  // namespace kl::gpu
