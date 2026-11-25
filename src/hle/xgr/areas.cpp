// keylime: the smaller runtime areas, as an offline retail console answers
// them (port of VIctory's xgame.rs, xsystem.rs, storage.rs, xnetworking.rs,
// display.rs, speech.rs and xpackage.rs).
#include <filesystem>
#include <mutex>

#include "hle/files.h"
#include "hle/xgr/xgr.h"
#include "log.h"
#include "runtime.h"

namespace kl::xgr {
namespace {

using This = Object*;
template <class F>
void s(void** t, size_t off, F fn) {
    set(t, off, reinterpret_cast<void*>(fn));
}
HRESULT WINAPI register_cb(This, void*, void*, void*, uint64_t* out) { return put(out, token()); }
bool WINAPI unregister_cb(This, uint64_t, bool) { return true; }

// ---------------------------------------------------------------- game ----
const GUID TITLE_PROVIDER = {0x973a344e, 0x24bf, 0x4d0f, {0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29}};
const GUID TITLE_IID = {0x2549f142, 0x6419, 0x4a06, {0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34}};
const GUID FEATURES = {0x8836fe87, 0xedb9, 0x4fe3, {0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40}};
const GUID INVITES = {0x0651aae2, 0x4012, 0x4077, {0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c}};
const GUID EVENTS = {0xbbfbdcc7, 0xbfe7, 0x409b, {0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d}};

HRESULT WINAPI title_id(This, uint32_t* out) { return put(out, runtime().title_id); }
void WINAPI launch_new_game(This, const char* exe, const char* args, const void*) {
    KL_WARN("xgr", "XLaunchNewGame(%s %s): not supported, the title keeps running", exe ? exe : "", args ? args : "");
}
bool WINAPI feature_available(This, uint32_t f) { return f <= 20 && f != 17; }
HRESULT WINAPI write_event(This, const void*, const char*, const char*, const char*, const char*, const char*) { return S_OK; }

// -------------------------------------------------------------- system ----
const GUID SYSTEM = {0xe349bd1a, 0xfc20, 0x4e40, {0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f}};
const GUID SYSTEM_IID = {0x6fd71f09, 0x7513, 0x49f0, {0x89, 0xbc, 0xbf, 0xaf, 0x5d, 0xf6, 0xf8, 0x52}};
const GUID SYSTEM_IID_2026 = {0x1861cf2e, 0xe18b, 0x4834, {0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf}};
const GUID ANALYTICS = {0xb884675d, 0xb738, 0x4a9c, {0x81, 0x5d, 0x9a, 0x9a, 0x1e, 0x0c, 0x6c, 0x9b}};
HRESULT WINAPI console_id(This, size_t n, char* b, size_t* u) { return put_string("00000000000000000000000000000000000001", n, b, u); }
HRESULT WINAPI sandbox_id(This, size_t n, char* b, size_t* u) { return put_string("RETAIL", n, b, u); }
HRESULT WINAPI device_id(This, size_t n, char* b, size_t* u) { return put_string("a2V5bGltZSBvZmZsaW5lIGRldmljZQ==", n, b, u); }
struct Version {
    uint16_t major, minor, build, revision;
};
struct AnalyticsInfo {
    Version os, hosting_os;
    char family[64], form[64];
};
AnalyticsInfo* WINAPI analytics_info(This, AnalyticsInfo* out) {
    *out = {};
    out->os = out->hosting_os = {10, 0, 22000, 0};
    strcpy_s(out->family, "Xbox");
    strcpy_s(out->form, runtime().console == Console::SeriesS ? "Xbox Series S" : "Xbox Series X");
    return out;
}

// --------------------------------------------------- persistent storage ----
const GUID STORAGE = {0xf4faf4d4, 0x2d04, 0x4fce, {0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84}};
const GUID STORAGE_IID = {0xd29411df, 0x0794, 0x4553, {0x8b, 0x27, 0x95, 0xfc, 0x02, 0xd0, 0xf7, 0x5d}};
const GUID STORAGE_IID_2026 = {0x41a4e10c, 0x5a7e, 0x41d9, {0x8c, 0x37, 0x37, 0xbd, 0xe6, 0x2a, 0x07, 0xd6}};
constexpr uint64_t STORAGE_QUOTA = 4ull << 30;
std::filesystem::path storage_dir() {
    static std::filesystem::path p = [] {
        auto d = runtime().data_dir / "storage";
        std::error_code e;
        std::filesystem::create_directories(d, e);
        files::mount('S', d);
        return d;
    }();
    return p;
}
const char* STORAGE_PATH = "S:\\";
HRESULT WINAPI path_size(This, size_t* out) {
    storage_dir();
    return put(out, strlen(STORAGE_PATH) + 1);
}
HRESULT WINAPI path_(This, size_t n, char* b, size_t* u) {
    storage_dir();
    return put_string(STORAGE_PATH, n, b, u);
}
struct SpaceInfo {
    uint64_t available_free, total_free, used, total;
};
uint64_t used_bytes() {
    uint64_t n = 0;
    std::error_code e;
    for (auto& f : std::filesystem::recursive_directory_iterator(storage_dir(), e))
        if (f.is_regular_file(e)) n += f.file_size(e);
    return n;
}
HRESULT WINAPI space_info(This, SpaceInfo* out) {
    uint64_t used = used_bytes(), free = used < STORAGE_QUOTA ? STORAGE_QUOTA - used : 0;
    return put(out, SpaceInfo{free, free, used, STORAGE_QUOTA});
}
char PROMPT;
HRESULT WINAPI prompt_space(This, uint64_t bytes, AsyncBlock* b) {
    uint64_t used = used_bytes();
    return finished(b, &PROMPT, bytes + used <= STORAGE_QUOTA ? S_OK : E_ABORT_);
}
HRESULT WINAPI prompt_result(This, AsyncBlock* b) { return async_result(b, &PROMPT, 0, nullptr, nullptr); }

// ---------------------------------------------------------- networking ----
const GUID NET = {0x37e56907, 0x2f10, 0x41e8, {0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a}};
const GUID NET_IID_2026 = {0xbf2346b2, 0x39af, 0x4658, {0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3}};
constexpr HRESULT UNREACHABLE = HRESULT(0x800704CF);
struct ConnectivityHint {
    uint32_t level, cost, interface_type;
    bool initialized, approaching, over, roaming;
};
HRESULT WINAPI udp_port(This, uint16_t* out) { return put(out, uint16_t(3074)); }
char PORT, SECURITY;
HRESULT WINAPI udp_port_async(This, AsyncBlock* b) { return finished(b, &PORT, S_OK, bytes(uint16_t(3074))); }
HRESULT WINAPI udp_port_result(This, AsyncBlock* b, uint16_t* out) { return result_as(b, &PORT, out); }
HRESULT WINAPI security_async(This, const char*, AsyncBlock* b) { return finished(b, &SECURITY, UNREACHABLE); }
HRESULT WINAPI security_size(This, AsyncBlock* b, size_t* out) { return async_result_size(b, out); }
HRESULT WINAPI security_result(This, AsyncBlock* b, size_t n, size_t* used, void* buf, void**) { return async_result(b, &SECURITY, n, buf, used); }
HRESULT WINAPI verify_cert(This, void*, const void*) { return UNREACHABLE; }
HRESULT WINAPI connectivity(This, ConnectivityHint* out) { return put(out, ConnectivityHint{1, 0, 0, true, false, false, false}); }

// ------------------------------------------------------------- display ----
const GUID HDR = {0x03f0fe74, 0xfdd9, 0x4e5c, {0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5}};
const GUID HDR_IID = {0x35f07670, 0x706e, 0x4bfb, {0x94, 0x76, 0x09, 0x07, 0x98, 0xc5, 0xeb, 0xf3}};
const GUID SHELL = {0x1b339674, 0x328d, 0x4283, {0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39}};
const GUID CAPTURE = {0xa4f1aee2, 0x4bf1, 0x4485, {0xb0, 0x08, 0xa7, 0xc2, 0x6d, 0x52, 0xac, 0x27}};
const GUID CAPTURE_IID_2026 = {0x22e672d7, 0xb4e3, 0x406c, {0xbd, 0x50, 0x8f, 0x0d, 0x25, 0x23, 0x6f, 0x9e}};
uint32_t WINAPI try_enable_hdr(This, uint32_t pref, void*) {
    KL_INFO("xgr", "HDR mode request %u: SDR presentation", pref);
    return 2;
}
HRESULT WINAPI launch_uri(This, const void*, const char* uri) {
    KL_INFO("xgr", "URI not launched: %s", uri ? uri : "");
    return NOT_SUPPORTED;
}
HRESULT WINAPI acquire_deferral(This, void** out) {
    if (!out) return E_INVALIDARG_;
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED | ES_SYSTEM_REQUIRED);
    *out = reinterpret_cast<void*>(uintptr_t(1));
    return S_OK;
}
void WINAPI close_deferral(This, void*) {}
HRESULT WINAPI screenshot(This, const void*, void*) { return NOT_SUPPORTED; }

// -------------------------------------------------------------- speech ----
const GUID SPEECH = {0x3e241977, 0x6237, 0x41e9, {0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1}};
const GUID SPEECH_IID = {0xd722b373, 0x8c4d, 0x4692, {0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d}};
const uint8_t SILENCE[44] = {'R', 'I', 'F', 'F', 0x24, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ', 0x10, 0, 0, 0, 1, 0, 1, 0,
                             0x80, 0x3e, 0, 0, 0, 0x7d, 0, 0, 2, 0, 0x10, 0, 'd', 'a', 't', 'a', 0, 0, 0, 0};
char SYNTH, STREAM;
HRESULT WINAPI enumerate_voices(This, void*, void*) { return S_OK; }
HRESULT WINAPI synth_create(This, void** out) { return put(out, static_cast<void*>(&SYNTH)); }
HRESULT WINAPI synth_ok(This, void*) { return S_OK; }
HRESULT WINAPI synth_voice(This, void*, const char*) { return S_OK; }
HRESULT WINAPI stream_create(This, void*, const char*, void** out) { return put(out, static_cast<void*>(&STREAM)); }
HRESULT WINAPI stream_size(This, void*, size_t* out) { return put(out, sizeof SILENCE); }
HRESULT WINAPI stream_data(This, void*, size_t n, uint8_t* buf, size_t* used) {
    if (!buf || n < sizeof SILENCE) return E_NOT_SUFFICIENT_BUFFER_;
    memcpy(buf, SILENCE, sizeof SILENCE);
    if (used) *used = sizeof SILENCE;
    return S_OK;
}

// ------------------------------------------------------------- package ----
const GUID PACKAGE = {0xaf406016, 0xe850, 0x4aa8, {0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e}};
const GUID PACKAGE_IID_2026 = {0x0b9d556e, 0xab70, 0x43f4, {0xba, 0xa0, 0x99, 0x99, 0xac, 0x4c, 0xb0, 0x40}};
constexpr HRESULT NO_PACKAGE = HRESULT(0x89245208);
std::string identifier() {
    char b[16];
    snprintf(b, sizeof b, "%08X", runtime().title_id);
    return b;
}
bool ours(const char* id) { return id && identifier() == id; }
uint64_t package_bytes() {
    static uint64_t n = [] {
        uint64_t t = 0;
        std::error_code e;
        for (auto& f : std::filesystem::recursive_directory_iterator(runtime().title_dir, e))
            if (f.is_regular_file(e)) t += f.file_size(e);
        return t;
    }();
    return n;
}
char MONITOR, INSTALL, MOUNT;
HRESULT WINAPI current_identifier(This, size_t n, char* b) { return put_string(identifier().c_str(), n, b, nullptr); }
bool WINAPI is_packaged(This) { return true; }
HRESULT WINAPI create_monitor(This, const char* id, uint32_t, const void*, uint32_t, void*, void** out) {
    return ours(id) ? put(out, static_cast<void*>(&MONITOR)) : NO_PACKAGE;
}
void WINAPI close_monitor(This, void*) {}
struct Progress {
    uint64_t total, installed, launch;
    bool launchable, completed;
};
void WINAPI progress(This, void*, Progress* out) {
    uint64_t n = package_bytes();
    put(out, Progress{n, n, n, true, true});
}
HRESULT WINAPI register_progress(This, void*, void*, void*, uint64_t* out) { return put(out, token()); }
HRESULT WINAPI find_availability(This, const char* id, uint32_t, const void*, uint32_t* out) {
    return ours(id) ? put(out, 0u) : NO_PACKAGE;
}
HRESULT WINAPI change_order(This, const char* id, uint32_t, const void*) { return ours(id) ? S_OK : NO_PACKAGE; }
HRESULT WINAPI install_async(This, const char* id, uint32_t, const void*, uint32_t, bool, AsyncBlock* b) {
    void* m = &MONITOR;
    return ours(id) ? finished(b, &INSTALL, S_OK, bytes(m)) : finished(b, &INSTALL, NO_PACKAGE);
}
HRESULT WINAPI install_result(This, AsyncBlock* b, void** out) { return result_as(b, &INSTALL, out); }
HRESULT WINAPI estimate(This, const char* id, uint32_t, const void*, uint64_t* n, bool* confirm) {
    if (!ours(id)) return NO_PACKAGE;
    if (confirm) *confirm = false;
    return put(n, uint64_t(0));
}
HRESULT WINAPI mount_path(This, const void*, size_t, char*) { return E_INVALIDARG_; }
void WINAPI close_mount(This, void*) {}
bool WINAPI uninstall_package(This, const char*) { return false; }
HRESULT WINAPI mount_async(This, const char*, AsyncBlock* b) { return finished(b, &MOUNT, NO_PACKAGE); }
HRESULT WINAPI mount_result(This, AsyncBlock* b, void** out) { return result_as(b, &MOUNT, out); }
HRESULT WINAPI enumerate(This, uint32_t, uint32_t, void*, void*) { return S_OK; }

struct Init {
    Init() {
        void** title = make_table("title", 5);
        s(title, 0x18, &title_id), s(title, 0x20, &launch_new_game);
        void** feat = make_table("features", 4);
        s(feat, 0x18, &feature_available);
        void** inv = make_table("invites", 5);
        s(inv, 0x18, &register_cb), s(inv, 0x20, &unregister_cb);
        void** ev = make_table("events", 4);
        s(ev, 0x18, &write_event);
        void** sys = make_table("system", 6);
        s(sys, 0x18, &console_id), s(sys, 0x20, &sandbox_id), s(sys, 0x28, &device_id);
        void** an = make_table("analytics", 4);
        s(an, 0x18, &analytics_info);
        void** st = make_table("persistent local storage", 8);
        s(st, 0x18, &path_size), s(st, 0x20, &path_), s(st, 0x28, &space_info), s(st, 0x30, &prompt_space), s(st, 0x38, &prompt_result);
        void** net = make_table("networking", 18);
        s(net, 0x18, &udp_port), s(net, 0x20, &udp_port_async), s(net, 0x28, &udp_port_result), s(net, 0x40, &security_async);
        s(net, 0x48, &security_size), s(net, 0x50, &security_result), s(net, 0x70, &verify_cert), s(net, 0x78, &connectivity);
        s(net, 0x80, &register_cb), s(net, 0x88, &unregister_cb);
        void** hdr = make_table("hdr", 5);
        s(hdr, 0x20, &try_enable_hdr);
        void** sh = make_table("shell", 6);
        s(sh, 0x18, &launch_uri), s(sh, 0x20, &acquire_deferral), s(sh, 0x28, &close_deferral);
        void** cap = make_table("capture", 6);
        s(cap, 0x28, &screenshot);
        void** sp = make_table("accessibility", 22);
        s(sp, 0x40, &enumerate_voices), s(sp, 0x48, &synth_create), s(sp, 0x50, &synth_ok), s(sp, 0x58, &synth_ok);
        s(sp, 0x60, &synth_voice), s(sp, 0x68, &stream_create), s(sp, 0x70, &synth_ok), s(sp, 0x78, &stream_size);
        s(sp, 0x80, &stream_data), s(sp, 0xa8, &stream_create);
        void** pk = make_table("package", 40);
        s(pk, 0x18, &current_identifier), s(pk, 0x20, &is_packaged), s(pk, 0x28, &create_monitor), s(pk, 0x30, &close_monitor);
        s(pk, 0x38, &progress), s(pk, 0x48, &register_progress), s(pk, 0x60, &find_availability), s(pk, 0x70, &change_order);
        s(pk, 0x80, &install_async), s(pk, 0x88, &install_result), s(pk, 0x90, &estimate), s(pk, 0x98, &change_order);
        s(pk, 0xc8, &mount_path), s(pk, 0xd0, &close_mount), s(pk, 0x110, &uninstall_package), s(pk, 0x128, &mount_async);
        s(pk, 0x130, &mount_result), s(pk, 0x138, &enumerate);
        static Object o_title{title, "title"}, o_feat{feat, "features"}, o_inv{inv, "invites"}, o_ev{ev, "events"},
            o_sys{sys, "system"}, o_an{an, "analytics"}, o_st{st, "persistent local storage"}, o_net{net, "networking"},
            o_hdr{hdr, "hdr"}, o_sh{sh, "shell"}, o_cap{cap, "capture"}, o_sp{sp, "accessibility"}, o_pk{pk, "package"};
        add_interfaces({{TITLE_PROVIDER, TITLE_IID, &o_title}, {FEATURES, FEATURES, &o_feat}, {INVITES, INVITES, &o_inv},
                        {EVENTS, EVENTS, &o_ev}, {SYSTEM, SYSTEM_IID, &o_sys}, {SYSTEM, SYSTEM_IID_2026, &o_sys},
                        {ANALYTICS, ANALYTICS, &o_an}, {STORAGE, STORAGE_IID, &o_st}, {STORAGE, STORAGE_IID_2026, &o_st},
                        {NET, NET, &o_net}, {NET, NET_IID_2026, &o_net}, {HDR, HDR_IID, &o_hdr}, {SHELL, SHELL, &o_sh},
                        {CAPTURE, CAPTURE_IID_2026, &o_cap}, {SPEECH, SPEECH_IID, &o_sp}, {PACKAGE, PACKAGE_IID_2026, &o_pk}});
    }
} g_init;

}  // namespace
}  // namespace kl::xgr
