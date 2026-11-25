// keylime: xgameruntime.dll, the console game runtime (XTaskQueue, XAsync,
// XUser, XSystem, XGame, XPackage, saves, store, ...), as an offline retail
// console with one signed-in user.
//
// Titles load the DLL dynamically (a pseudo module here), look up the
// *ApiImpl functions and ask QueryApiImpl for a COM-style object per area
// (provider GUID, interface GUID); each API call is one method of that
// object. Areas, GUIDs and slot maps were reverse engineered by VIctory
// (GPL-3.0, crates/victory-hle/src/modules/x*.rs); this is a C++ port of
// that work.
#pragma once
#include <windows.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace kl::xgr {

constexpr HRESULT E_PENDING_ = HRESULT(0x8000000A), E_ABORT_ = HRESULT(0x80004004), E_NOTIMPL_ = HRESULT(0x80004001),
                  E_INVALIDARG_ = HRESULT(0x80070057), E_NOT_SUFFICIENT_BUFFER_ = HRESULT(0x8007007A),
                  E_ILLEGAL_METHOD_CALL_ = HRESULT(0x8000000E), NOT_SUPPORTED = HRESULT(0x80070032);

// A runtime object as title code sees it: a method table pointer first.
struct Object {
    void* const* table;
    const char* area;
};

// A table of `n` slots: QueryInterface (logs), AddRef/Release (no-ops), the rest logging stubs.
void** make_table(const char* area, size_t n);
void set(void** t, size_t offset, void* fn);  // by byte offset

struct Interface {
    GUID provider, iid;
    Object* object;
};
void add_interfaces(std::initializer_list<Interface> list);

template <class T>
HRESULT put(T* out, T value) {
    if (!out) return E_INVALIDARG_;
    *out = value;
    return S_OK;
}
HRESULT put_string(const char* text, size_t size, char* buffer, size_t* used);
uint64_t token();

// ---- async (xasync.cpp) ----
struct AsyncBlock {
    void* queue;
    void* context;
    void(WINAPI* callback)(AsyncBlock*);
    uintptr_t internal[4];  // ours: [unused, status, result size, unused]
};
struct ProviderData {
    AsyncBlock* block;
    size_t buffer_size;
    void* buffer;
    void* context;
};
using Provider = HRESULT(WINAPI*)(uint32_t op, const ProviderData* data);
constexpr uint32_t OP_BEGIN = 0, OP_DO_WORK = 1, OP_GET_RESULT = 2, OP_CANCEL = 3, OP_CLEANUP = 4;

HRESULT async_begin(AsyncBlock* b, void* context, const void* identity, Provider p);
HRESULT async_schedule(AsyncBlock* b, uint32_t delay_ms);
void async_complete(AsyncBlock* b, HRESULT result, size_t required);
HRESULT async_status(AsyncBlock* b, bool wait);
HRESULT async_result_size(AsyncBlock* b, size_t* out);
HRESULT async_result(AsyncBlock* b, const void* identity, size_t size, void* buffer, size_t* used);
// A call that is already over: completes with `hr` and these result bytes.
HRESULT finished(AsyncBlock* b, const void* identity, HRESULT hr, std::vector<uint8_t> bytes = {});
// A call whose result `work` makes on a thread of ours.
HRESULT pending(AsyncBlock* b, const void* identity, std::function<HRESULT(std::vector<uint8_t>&)> work);
template <class T>
std::vector<uint8_t> bytes(const T& v) {
    std::vector<uint8_t> b(sizeof v);
    memcpy(b.data(), &v, sizeof v);
    return b;
}
template <class T>
HRESULT result_as(AsyncBlock* b, const void* identity, T* out) {
    return async_result(b, identity, sizeof(T), out, nullptr);
}
// Submits `f` to the queue's work (0) or completion (1) port.
HRESULT queue_submit(void* queue, uint32_t port, uint32_t delay_ms, void* context, void(WINAPI* callback)(void*, bool));

}  // namespace kl::xgr
