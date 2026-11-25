// keylime: XUser: one offline local user, signed in, never a guest, every
// privilege granted (port of VIctory's xuser.rs).
#include "hle/xgr/xgr.h"
#include "log.h"

namespace kl::xgr {
namespace {

const GUID PROVIDER = {0x01acd177, 0x91f9, 0x4763, {0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0}};
const GUID IID_USER = {0xeb9bf948, 0x18dc, 0x4d82, {0xbb, 0xcc, 0x40, 0xe0, 0xa8, 0x09, 0xc4, 0xc0}};
const GUID GAMERTAG_IID = {0xcef4fac0, 0x7676, 0x4a94, {0xa1, 0x19, 0x4c, 0x43, 0xf9, 0xeb, 0x5b, 0x74}};
const GUID DEVICE_PROVIDER = {0x7d824997, 0x10dc, 0x45ab, {0x86, 0xb7, 0x27, 0x37, 0x76, 0x7c, 0x0b, 0xf1}};
const GUID DEVICE_IID = {0x0cc6a956, 0xe7e1, 0x4fdf, {0x93, 0x41, 0x9d, 0x5d, 0xa6, 0x49, 0xeb, 0xc8}};

constexpr HRESULT USER_NOT_FOUND = HRESULT(0x89245104), FAILED_TO_GET_TOKEN = HRESULT(0x89245111);
constexpr uint64_t XUID = 0x0009000000000001ull, LOCAL_ID = 1;

char g_user;  // the only user; handles are its address
void* user() { return &g_user; }

using This = Object*;
HRESULT WINAPI duplicate(This, void*, void** out) { return put(out, user()); }
void WINAPI close_(This, void*) {}
int32_t WINAPI compare(This, void* a, void* b) { return a != b; }

char ADD, RESOLVE, TOKEN;
HRESULT WINAPI add_provider(uint32_t op, const ProviderData* d) {
    if (op == OP_BEGIN) async_complete(d->block, S_OK, sizeof(void*));
    if (op == OP_GET_RESULT) return put(static_cast<void**>(d->buffer), user());
    return S_OK;
}
HRESULT WINAPI add_async(This, uint32_t options, AsyncBlock* b) {
    KL_INFO("xgr", "XUserAddAsync(options %u): the offline local user signs in", options);
    return async_begin(b, nullptr, &ADD, &add_provider);
}
HRESULT WINAPI add_result(This, AsyncBlock* b, void** out) { return async_result(b, &ADD, sizeof(void*), out, nullptr); }
HRESULT WINAPI get_local_id(This, void*, uint64_t* out) { return put(out, LOCAL_ID); }
HRESULT WINAPI find_by_local_id(This, uint64_t id, void** out) { return id == LOCAL_ID ? put(out, user()) : USER_NOT_FOUND; }
HRESULT WINAPI get_id(This, void*, uint64_t* out) { return put(out, XUID); }
HRESULT WINAPI find_by_id(This, uint64_t id, void** out) { return id == XUID ? put(out, user()) : USER_NOT_FOUND; }
HRESULT WINAPI get_is_guest(This, void*, bool* out) { return put(out, false); }
HRESULT WINAPI get_state(This, void*, uint32_t* out) { return put(out, 0u); }  // signed in
HRESULT WINAPI get_gamertag(This, void*, uint32_t component, size_t size, char* buf, size_t* used) {
    return put_string(component == 2 ? "" : "Player", size, buf, used);  // 2: modern suffix (none)
}
HRESULT WINAPI get_age_group(This, void*, uint32_t* out) { return put(out, 3u); }  // adult
HRESULT WINAPI check_privilege(This, void*, uint32_t, uint32_t, bool* has, uint32_t* reason) {
    if (reason) *reason = 0;
    return put(has, true);
}
HRESULT WINAPI resolve_async(This, void*, uint32_t, uint32_t, AsyncBlock* b) { return finished(b, &RESOLVE, S_OK); }
HRESULT WINAPI resolve_result(This, AsyncBlock* b) { return async_result(b, &RESOLVE, 0, nullptr, nullptr); }
HRESULT WINAPI token_async(This, void*, uint32_t, const char*, const char* url, size_t, const void*, size_t, const void*, AsyncBlock* b) {
    KL_INFO("xgr", "user token for %s: offline", url ? url : "");
    return finished(b, &TOKEN, FAILED_TO_GET_TOKEN);
}
HRESULT WINAPI token_size(This, AsyncBlock* b, size_t* out) { return async_result_size(b, out); }
HRESULT WINAPI token_result(This, AsyncBlock* b, size_t size, void* buf, const void**, size_t* used) {
    return async_result(b, &TOKEN, size, buf, used);
}
HRESULT WINAPI register_(This, void*, void*, void*, uint64_t* out) { return put(out, token()); }
bool WINAPI unregister(This, uint64_t, bool) { return true; }
HRESULT WINAPI find_for_device(This, const void*, void**) { return USER_NOT_FOUND; }

struct Init {
    Init() {
        void** t = make_table("user", 35);
        auto s = [&](void** tb, size_t off, auto fn) { set(tb, off, reinterpret_cast<void*>(fn)); };
        s(t, 0x18, &duplicate), s(t, 0x20, &close_), s(t, 0x28, &compare), s(t, 0x38, &add_async), s(t, 0x40, &add_result);
        s(t, 0x48, &get_local_id), s(t, 0x50, &find_by_local_id), s(t, 0x58, &get_id), s(t, 0x60, &find_by_id);
        s(t, 0x68, &get_is_guest), s(t, 0x70, &get_state), s(t, 0x78, &get_gamertag), s(t, 0x98, &get_age_group);
        s(t, 0xa0, &check_privilege), s(t, 0xa8, &resolve_async), s(t, 0xb0, &resolve_result), s(t, 0xb8, &token_async);
        s(t, 0xc0, &token_size), s(t, 0xc8, &token_result), s(t, 0x108, &register_), s(t, 0x110, &unregister);
        void** g = make_table("user gamertag", 4);
        s(g, 0x18, &get_gamertag);
        void** d = make_table("user devices", 6);
        s(d, 0x18, &find_for_device), s(d, 0x20, &register_), s(d, 0x28, &unregister);
        static Object user_o{t, "user"}, tag_o{g, "user gamertag"}, dev_o{d, "user devices"};
        add_interfaces({{PROVIDER, IID_USER, &user_o}, {PROVIDER, PROVIDER, &user_o}, {PROVIDER, GAMERTAG_IID, &tag_o},
                        {DEVICE_PROVIDER, DEVICE_IID, &dev_o}});
    }
} g_init;

}  // namespace
}  // namespace kl::xgr
