// keylime: XStore as an offline console answers it (port of VIctory's
// xstore.rs): the game's own licence is local and active (a full, owned
// game), the user owns no add-ons, and what needs the store service fails
// with the documented store network error.
#include <fstream>
#include <sstream>

#include "hle/xgr/xgr.h"
#include "log.h"
#include "runtime.h"

namespace kl::xgr {
namespace {

const GUID PROVIDER = {0x0dd112ac, 0x7c24, 0x448c, {0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c}};
const GUID IID_2026 = {0xb09d803c, 0x2414, 0x4a05, {0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44}};
const GUID DURABLES_IID = {0x2d42fea5, 0xe71d, 0x4b76, {0x97, 0xcd, 0xc5, 0x0a, 0xfb, 0xb3, 0xae, 0x5d}};
constexpr HRESULT NETWORK_ERROR = HRESULT(0x89245301), NO_PACKAGE = HRESULT(0x89245208);

// The StoreId in the package's MicrosoftGame.Config (empty if none).
const std::string& store_id() {
    static std::string id = [] {
        std::ifstream f(runtime().title_dir / "MicrosoftGame.Config");
        std::stringstream ss;
        ss << f.rdbuf();
        std::string s = ss.str();
        size_t a = s.find("<StoreId>"), b = s.find("</StoreId>");
        return a != std::string::npos && b != std::string::npos && b > a ? s.substr(a + 9, b - a - 9) : std::string();
    }();
    return id;
}
std::string identifier() {
    char b[16];
    snprintf(b, sizeof b, "%08X", runtime().title_id);
    return b;
}

using This = Object*;
char CONTEXT, QUERY, VALID, INVALID, FAILED_CALL, QUERY_CALL, LICENSE_CALL, GAME_LICENSE;

HRESULT offline(AsyncBlock* b, const char* what) {
    KL_INFO("xgr", "store call (%s): offline", what);
    return finished(b, &FAILED_CALL, NETWORK_ERROR);
}
HRESULT WINAPI status(This, AsyncBlock* b) { return async_status(b, false); }
HRESULT WINAPI create_context(This, const void*, void** out) { return put(out, static_cast<void*>(&CONTEXT)); }
void WINAPI close_(This, void*) {}
HRESULT WINAPI associated_async(This, void*, uint32_t, uint32_t, AsyncBlock* b) { return offline(b, "associated products"); }
HRESULT WINAPI entitled_async(This, void*, uint32_t, uint32_t, AsyncBlock* b) {
    void* q = &QUERY;
    return finished(b, &QUERY_CALL, S_OK, bytes(q));
}
HRESULT WINAPI query_result(This, AsyncBlock* b, void** out) { return result_as(b, &QUERY_CALL, out); }
HRESULT WINAPI enumerate_products(This, void*, void*, const void*) { return S_OK; }
bool WINAPI has_more_pages(This, void*) { return false; }
HRESULT WINAPI next_page_async(This, void*, AsyncBlock* b) { return finished(b, &QUERY_CALL, E_ILLEGAL_METHOD_CALL_); }
HRESULT license_for(bool owned, AsyncBlock* b) {
    void* l = owned ? &VALID : &INVALID;
    return finished(b, &LICENSE_CALL, S_OK, bytes(l));
}
HRESULT WINAPI license_package_async(This, void*, const char* package, AsyncBlock* b) {
    return license_for(package && identifier() == package, b);
}
HRESULT WINAPI license_durables_async(This, void*, const char*, AsyncBlock* b) { return license_for(false, b); }
HRESULT WINAPI license_result(This, AsyncBlock* b, void** out) { return result_as(b, &LICENSE_CALL, out); }
bool WINAPI is_license_valid(This, void* l) { return l == &VALID; }
#pragma pack(push, 1)
struct GameLicense {
    char sku_store_id[18];
    bool is_active, is_trial_owned_by_this_user, is_disc_license, is_trial;
    uint32_t trial_seconds_remaining;
    char trial_unique_id[64];
    int64_t expiration;
};
#pragma pack(pop)
// Natural alignment as the C struct is laid out (no packing on the console ABI).
struct GameLicenseAligned {
    char sku_store_id[18];
    bool is_active, is_trial_owned_by_this_user, is_disc_license, is_trial;
    uint32_t trial_seconds_remaining;
    char trial_unique_id[64];
    int64_t expiration;
};
HRESULT WINAPI game_license_async(This, void*, AsyncBlock* b) {
    GameLicenseAligned l{};
    std::string sku = store_id() + "/0010";
    strncpy_s(l.sku_store_id, sku.c_str(), sizeof l.sku_store_id - 1);
    l.is_active = true;
    KL_INFO("xgr", "game licence: full, active (%s)", sku.c_str());
    return finished(b, &GAME_LICENSE, S_OK, bytes(l));
}
HRESULT WINAPI game_license_result(This, AsyncBlock* b, GameLicenseAligned* out) { return result_as(b, &GAME_LICENSE, out); }
HRESULT WINAPI consumable_async(This, void*, const char*, AsyncBlock* b) { return offline(b, "consumable balance"); }
HRESULT WINAPI fulfillment_async(This, void*, const char*, uint32_t, const void*, AsyncBlock* b) { return offline(b, "fulfillment"); }
HRESULT WINAPI user_id_async(This, void*, const char*, const char*, AsyncBlock* b) { return offline(b, "user store id"); }
HRESULT WINAPI size_status(This, AsyncBlock* b, size_t*) { return async_status(b, false); }
HRESULT WINAPI string_status(This, AsyncBlock* b, size_t, char*) { return async_status(b, false); }
HRESULT WINAPI count_status(This, AsyncBlock* b, uint32_t*) { return async_status(b, false); }
HRESULT WINAPI array_status(This, AsyncBlock* b, uint32_t, void*) { return async_status(b, false); }
HRESULT WINAPI purchase_ui_async(This, void*, const char*, const char*, const char*, AsyncBlock* b) { return offline(b, "purchase page"); }
HRESULT WINAPI product_page_async(This, void*, const char*, AsyncBlock* b) { return offline(b, "product page"); }
HRESULT WINAPI redeem_async(This, void*, const char*, const char* const*, size_t, bool, AsyncBlock* b) { return offline(b, "redeem page"); }
HRESULT WINAPI updates_async(This, void*, AsyncBlock* b) { return offline(b, "update check"); }
HRESULT WINAPI install_async(This, void*, const char* const*, size_t, AsyncBlock* b) { return offline(b, "package download"); }
HRESULT WINAPI package_identifier(This, const char* id, size_t n, char* buf) {
    if (!id || store_id().empty() || store_id() != id) return NO_PACKAGE;
    return put_string(identifier().c_str(), n, buf, nullptr);
}
HRESULT WINAPI register_(This, void*, void*, void*, void*, uint64_t* out) { return put(out, token()); }
bool WINAPI unregister(This, void*, uint64_t, bool) { return true; }

struct Init {
    Init() {
        static_assert(sizeof(GameLicenseAligned) == 104);
        void** t = make_table("store", 76);
        auto s = [&](size_t off, auto fn) { set(t, off, reinterpret_cast<void*>(fn)); };
        s(0x18, &create_context), s(0x20, &close_), s(0x28, &associated_async), s(0x30, &query_result);
        s(0x48, &entitled_async), s(0x50, &query_result), s(0x78, &enumerate_products), s(0x80, &has_more_pages);
        s(0x88, &next_page_async), s(0x90, &query_result), s(0x98, &close_), s(0xa0, &license_package_async);
        s(0xa8, &license_result), s(0xb0, &is_license_valid), s(0xb8, &close_), s(0xe0, &game_license_async);
        s(0xe8, &game_license_result), s(0x108, &consumable_async), s(0x110, &status), s(0x118, &fulfillment_async);
        s(0x128, &user_id_async), s(0x130, &size_status), s(0x138, &string_status), s(0x140, &user_id_async);
        s(0x148, &size_status), s(0x150, &string_status), s(0x188, &purchase_ui_async), s(0x1a8, &redeem_async);
        s(0x1b8, &updates_async), s(0x1c0, &count_status), s(0x1c8, &array_status), s(0x1e0, &install_async);
        s(0x1e8, &status), s(0x1f0, &install_async), s(0x1f8, &count_status), s(0x200, &array_status);
        s(0x208, &package_identifier), s(0x210, &register_), s(0x218, &unregister), s(0x220, &register_);
        s(0x228, &unregister), s(0x238, &license_durables_async), s(0x240, &license_result), s(0x258, &product_page_async);
        static Object o{t, "store"};
        add_interfaces({{PROVIDER, IID_2026, &o}, {PROVIDER, DURABLES_IID, &o}});
    }
} g_init;

}  // namespace
}  // namespace kl::xgr
