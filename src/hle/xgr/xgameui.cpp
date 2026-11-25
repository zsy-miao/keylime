// keylime: system dialogs (XGameUi), after VIctory's xgameui.rs. A profile
// card has nothing to show and closes at once; text entry is cancelled for
// now (a host dialog comes later).
#include "hle/xgr/xgr.h"
#include "log.h"

namespace kl::xgr {
namespace {

const GUID PROVIDER = {0xdfcd4649, 0x4ff8, 0x4043, {0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0}};
const GUID IID_2026 = {0xeaf669df, 0x5542, 0x4590, {0x99, 0xa3, 0x8d, 0xc0, 0x61, 0xf8, 0x37, 0xcc}};
using This = Object*;
char PROFILE, TEXT;

HRESULT WINAPI profile_async(This, AsyncBlock* b, const void*, uint64_t) { return finished(b, &PROFILE, S_OK); }
HRESULT WINAPI profile_result(This, AsyncBlock* b) { return async_result(b, &PROFILE, 0, nullptr, nullptr); }
HRESULT WINAPI text_async(This, AsyncBlock* b, const char* title, const char*, const char*, uint32_t, uint32_t) {
    KL_INFO("xgr", "text entry (%s): cancelled (no host dialog yet)", title ? title : "");
    return finished(b, &TEXT, E_ABORT_);
}
HRESULT WINAPI text_size(This, AsyncBlock* b, uint32_t* out) {
    size_t n = 0;
    HRESULT hr = async_result_size(b, &n);
    if (FAILED(hr)) return hr;
    return put(out, uint32_t(n));
}
HRESULT WINAPI text_result(This, AsyncBlock* b, uint32_t size, char* buf, uint32_t* used) {
    size_t w = 0;
    HRESULT hr = async_result(b, &TEXT, size, buf, &w);
    if (FAILED(hr) || !used) return hr;
    return put(used, uint32_t(w));
}

struct Init {
    Init() {
        void** t = make_table("game ui", 20);
        auto s = [&](size_t off, auto fn) { set(t, off, reinterpret_cast<void*>(fn)); };
        s(0x38, &profile_async), s(0x40, &profile_result), s(0x88, &text_async), s(0x90, &text_size), s(0x98, &text_result);
        static Object o{t, "game ui"};
        add_interfaces({{PROVIDER, IID_2026, &o}});
    }
} g_init;

}  // namespace
}  // namespace kl::xgr
