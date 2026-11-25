// keylime: gameinput.dll (console GameInput v0). Gamepads come from XInput
// slots 0-3; while none is connected, the keyboard drives a stand-in pad.
// The keyboard is a device of its own too. A poll thread turns host state
// into readings (one per change) and runs device callbacks on (dis)connect.
// Interface layout as reverse engineered by VIctory (gameinput.rs).
#include <windows.h>
#include <xinput.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "hle/hle.h"
#include "log.h"
#include "stats.h"

#pragma comment(lib, "xinput.lib")

namespace kl::gameinput {
namespace {

constexpr HRESULT DEVICE_DISCONNECTED = HRESULT(0x838A0001), DEVICE_NOT_FOUND = HRESULT(0x838A0002),
                  READING_NOT_FOUND = HRESULT(0x838A0003), REFERENCE_TOO_OLD = HRESULT(0x838A0004);
constexpr uint32_t KEYBOARD_KIND = 0x10, GAMEPAD_KIND = 0x40000, STATUS_CONNECTED = 0x3, ASYNC_ENUMERATION = 1;
enum : uint32_t { MENU = 1, VIEW = 2, A = 4, B = 8, X = 0x10, Y = 0x20, UP = 0x40, DOWN = 0x80, LEFT = 0x100,
                  RIGHT = 0x200, LSHOULDER = 0x400, RSHOULDER = 0x800, LSTICK = 0x1000, RSTICK = 0x2000 };
constexpr size_t PADS = 4, FALLBACK = PADS, KEYBOARD = PADS + 1, DEVICES = PADS + 2, HISTORY = 32;

struct PadState {
    uint32_t buttons = 0;
    float lt = 0, rt = 0, lx = 0, ly = 0, rx = 0, ry = 0;
    bool operator==(const PadState&) const = default;
};
struct KeyState {
    uint32_t scan_code, code_point;
    uint8_t virtual_key;
    bool dead_key;
    uint8_t pad[2];
    bool operator==(const KeyState& o) const { return scan_code == o.scan_code && virtual_key == o.virtual_key; }
};
static_assert(sizeof(KeyState) == 12 && sizeof(PadState) == 28);

uint64_t now_us() {
    static auto start = std::chrono::steady_clock::now();
    return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
}

struct DeviceInfo {
    uint32_t size;
    uint16_t vendor, product, revision;
    uint8_t interface_number, collection;
    uint16_t usage[2], versions[8];
    uint8_t id[32], root_id[32];
    int32_t family;
    uint32_t capabilities, supported_input, rumble_motors;
    uint64_t rest[26];
};
static_assert(offsetof(DeviceInfo, id) == 0x20);

struct Hub;
struct Object {
    const void* const* vtable;
    Hub* hub;
};
struct Device {
    const void* const* vtable;
    Hub* hub;
    size_t index;
    DeviceInfo info;
};
struct Reading {
    const void* const* vtable;
    std::atomic<uint32_t> refs;
    const Device* device;
    uint64_t sequence, timestamp;
    bool is_pad;
    PadState pad;
    std::vector<KeyState> keys;
    uint32_t kind() const { return is_pad ? GAMEPAD_KIND : KEYBOARD_KIND; }
    bool same(const Reading& o) const { return is_pad == o.is_pad && (is_pad ? pad == o.pad : keys == o.keys); }
};

extern const void* const kReading[25];

uint32_t WINAPI reading_add_ref(Reading* r) { return ++r->refs; }
uint32_t WINAPI reading_release(Reading* r) {
    uint32_t left = --r->refs;
    if (!left) delete r;
    return left;
}

using DeviceCallback = void(WINAPI*)(uint64_t, void*, const Device*, uint64_t, uint32_t, uint32_t);
struct Registration {
    uint64_t token;
    const Device* device;
    uint32_t kind, status;
    void* context;
    DeviceCallback callback;
};
struct Event {
    Registration r;
    size_t device;
    uint64_t timestamp;
    uint32_t current, previous;
};
struct Slot {
    bool connected = false;
    uint64_t sequence = 0;
    std::deque<Reading*> history;  // each holds one reference
};

uint32_t kind_of(size_t i) { return i == KEYBOARD ? KEYBOARD_KIND : GAMEPAD_KIND; }

struct Hub {
    Object object;
    Device devices[DEVICES];
    std::mutex lock;
    std::condition_variable returned;
    Slot slots[DEVICES];
    std::vector<Registration> callbacks;
    std::vector<Event> pending;
    std::vector<std::pair<uint64_t, DWORD>> running;
    std::atomic<uint64_t> next_token{1};

    bool matches(const Registration& r, size_t i) const {
        return (kind_of(i) & r.kind) && (!r.device || r.device == &devices[i]);
    }
    int index(const Device* d) const {
        for (size_t i = 0; i < DEVICES; ++i)
            if (&devices[i] == d) return int(i);
        return -1;
    }
    void update(size_t i, const Reading* value, uint64_t now, std::vector<Event>& events) {
        Slot& s = slots[i];
        bool was = s.connected;
        s.connected = value != nullptr;
        if (value && (!was || s.history.empty() || !s.history.back()->same(*value))) {
            auto* r = new Reading();
            r->vtable = kReading;
            r->is_pad = value->is_pad;
            r->pad = value->pad;
            r->keys = value->keys;
            r->refs = 1;
            r->device = &devices[i];
            r->sequence = ++s.sequence;
            r->timestamp = now;
            if (s.history.size() == HISTORY) {
                reading_release(s.history.front());
                s.history.pop_front();
            }
            s.history.push_back(r);
        }
        if (was != s.connected) {
            KL_INFO("input", "device %zu %s", i, s.connected ? "connected" : "disconnected");
            for (auto& r : callbacks)
                if (matches(r, i) && (r.status & STATUS_CONNECTED))
                    events.push_back({r, i, now, s.connected ? STATUS_CONNECTED : 0, s.connected ? 0 : STATUS_CONNECTED});
        }
    }
    void deliver(std::vector<Event>& events) {
        DWORD me = GetCurrentThreadId();
        for (auto& e : events) {
            {
                std::lock_guard g(lock);
                bool live = false;
                for (auto& c : callbacks) live |= c.token == e.r.token;
                if (!live) continue;
                running.push_back({e.r.token, me});
            }
            e.r.callback(e.r.token, e.r.context, &devices[e.device], e.timestamp, e.current, e.previous);
            std::lock_guard g(lock);
            std::erase(running, std::pair<uint64_t, DWORD>{e.r.token, me});
            returned.notify_all();
        }
    }
    void poll();
};

// ---------------------------------------------------------------- host ----

bool focused() {
    DWORD pid = 0;
    GetWindowThreadProcessId(GetForegroundWindow(), &pid);
    return pid == GetCurrentProcessId();
}

std::vector<KeyState> host_keys() {
    std::vector<KeyState> keys;
    if (!focused()) return keys;
    for (int vk = 8; vk < 256; ++vk) {
        if (vk == VK_SHIFT || vk == VK_CONTROL || vk == VK_MENU) continue;  // sided ones are reported
        if (!(GetAsyncKeyState(vk) & 0x8000)) continue;
        KeyState k{};
        k.virtual_key = uint8_t(vk);
        k.scan_code = MapVirtualKeyW(UINT(vk), MAPVK_VK_TO_VSC_EX);
        k.code_point = MapVirtualKeyW(UINT(vk), MAPVK_VK_TO_CHAR) & 0x7FFFFFFF;
        keys.push_back(k);
    }
    return keys;
}

PadState from_xinput(const XINPUT_GAMEPAD& p) {
    static const std::pair<WORD, uint32_t> map[] = {
        {XINPUT_GAMEPAD_START, MENU}, {XINPUT_GAMEPAD_BACK, VIEW}, {XINPUT_GAMEPAD_A, A}, {XINPUT_GAMEPAD_B, B},
        {XINPUT_GAMEPAD_X, X}, {XINPUT_GAMEPAD_Y, Y}, {XINPUT_GAMEPAD_DPAD_UP, UP}, {XINPUT_GAMEPAD_DPAD_DOWN, DOWN},
        {XINPUT_GAMEPAD_DPAD_LEFT, LEFT}, {XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT}, {XINPUT_GAMEPAD_LEFT_SHOULDER, LSHOULDER},
        {XINPUT_GAMEPAD_RIGHT_SHOULDER, RSHOULDER}, {XINPUT_GAMEPAD_LEFT_THUMB, LSTICK}, {XINPUT_GAMEPAD_RIGHT_THUMB, RSTICK}};
    PadState s;
    for (auto& [h, b] : map)
        if (p.wButtons & h) s.buttons |= b;
    auto stick = [](SHORT v) { float f = float(v) / 32767.0f; return f < -1 ? -1.0f : f; };
    s.lt = float(p.bLeftTrigger) / 255.0f;
    s.rt = float(p.bRightTrigger) / 255.0f;
    s.lx = stick(p.sThumbLX);
    s.ly = stick(p.sThumbLY);
    s.rx = stick(p.sThumbRX);
    s.ry = stick(p.sThumbRY);
    return s;
}

// Keyboard stand-in pad: Enter=A, Backspace=B, Esc=Menu, Tab=View, X/Y, arrows=dpad,
// Q/E shoulders, F/H stick clicks, WASD left stick, IJKL right stick, Z/C triggers.
PadState fallback(const std::vector<KeyState>& keys) {
    auto on = [&](int vk) {
        for (auto& k : keys)
            if (k.virtual_key == vk) return 1.0f;
        return 0.0f;
    };
    static const std::pair<int, uint32_t> map[] = {
        {VK_ESCAPE, MENU}, {VK_TAB, VIEW}, {VK_RETURN, A}, {VK_BACK, B}, {'X', X}, {'Y', Y}, {VK_UP, UP},
        {VK_DOWN, DOWN}, {VK_LEFT, LEFT}, {VK_RIGHT, RIGHT}, {'Q', LSHOULDER}, {'E', RSHOULDER}, {'F', LSTICK}, {'H', RSTICK}};
    PadState s;
    for (auto& [vk, b] : map)
        if (on(vk) != 0) s.buttons |= b;
    s.lt = on('Z');
    s.rt = on('C');
    s.lx = on('D') - on('A');
    s.ly = on('W') - on('S');
    s.rx = on('L') - on('J');
    s.ry = on('I') - on('K');
    return s;
}

void Hub::poll() {
    // XInput on an empty slot is slow: look again once a second.
    static uint64_t last_empty_check[PADS] = {};
    static bool present[PADS] = {};
    std::optional<PadState> pads[PADS];
    uint64_t now = now_us();
    for (DWORD i = 0; i < PADS; ++i) {
        if (!present[i] && now - last_empty_check[i] < 1000000 && last_empty_check[i]) continue;
        XINPUT_STATE st{};
        present[i] = XInputGetState(i, &st) == ERROR_SUCCESS;
        if (!present[i]) last_empty_check[i] = now;
        else pads[i] = from_xinput(st.Gamepad);
    }
    auto keys = host_keys();
    std::vector<Event> events;
    {
        std::lock_guard g(lock);
        events = std::move(pending);
        pending.clear();
        bool any = false;
        for (size_t i = 0; i < PADS; ++i) {
            Reading r{};
            r.is_pad = true;
            if (pads[i]) r.pad = *pads[i], any = true;
            update(i, pads[i] ? &r : nullptr, now, events);
        }
        Reading fb{};
        fb.is_pad = true;
        fb.pad = fallback(keys);
        update(FALLBACK, any ? nullptr : &fb, now, events);
        Reading kb{};
        kb.is_pad = false;
        kb.keys = keys;
        update(KEYBOARD, &kb, now, events);
    }
    deliver(events);
}

// ----------------------------------------------------------- the object ----

using Out = void**;
HRESULT fail(Out out, HRESULT hr) {
    if (out) *out = nullptr;
    return hr;
}
HRESULT hand(Out out, Reading* r) {
    if (!out) return E_INVALIDARG;
    if (!r) return fail(out, READING_NOT_FOUND);
    reading_add_ref(r);
    *out = r;
    return S_OK;
}
Hub* hub_of(Object* o) { return o->hub; }

HRESULT WINAPI no_interface(void*, const GUID*, Out out) { return fail(out, E_NOINTERFACE); }
uint32_t WINAPI add_ref(void*) { return 1; }
uint64_t WINAPI timestamp(Object*) { return now_us(); }

HRESULT WINAPI current_reading(Object* o, uint32_t kind, const Device* d, Out out) {
    stats::add(stats::InputReadings);
    Hub* h = hub_of(o);
    std::lock_guard g(h->lock);
    auto newest = [&](size_t i) -> Reading* {
        auto& hs = h->slots[i].history;
        return (!hs.empty() && (hs.back()->kind() & kind)) ? hs.back() : nullptr;
    };
    Reading* found = nullptr;
    if (!d) {
        for (size_t i = 0; i < DEVICES; ++i)
            if (h->slots[i].connected)
                if (auto* r = newest(i); r && (!found || r->timestamp > found->timestamp)) found = r;
    } else {
        int i = h->index(d);
        if (i < 0) return fail(out, E_INVALIDARG);
        if (!h->slots[i].connected) return fail(out, DEVICE_DISCONNECTED);
        found = newest(size_t(i));
    }
    return hand(out, found);
}

HRESULT relative(Object* o, const Reading* ref, uint32_t kind, const Device* d, Out out, bool forward) {
    if (!ref) return fail(out, E_INVALIDARG);
    Hub* h = hub_of(o);
    int i = h->index(ref->device);
    if (i < 0 || (d && d != ref->device)) return fail(out, E_INVALIDARG);
    std::lock_guard g(h->lock);
    auto& hs = h->slots[i].history;
    if (hs.empty() || ref->sequence < hs.front()->sequence) return fail(out, REFERENCE_TOO_OLD);
    Reading* found = nullptr;
    if (forward) {
        for (auto* r : hs)
            if (r->sequence > ref->sequence && (r->kind() & kind)) { found = r; break; }
    } else {
        for (auto it = hs.rbegin(); it != hs.rend(); ++it)
            if ((*it)->sequence < ref->sequence && ((*it)->kind() & kind)) { found = *it; break; }
    }
    return hand(out, found);
}
HRESULT WINAPI next_reading(Object* o, const Reading* r, uint32_t k, const Device* d, Out out) { return relative(o, r, k, d, out, true); }
HRESULT WINAPI previous_reading(Object* o, const Reading* r, uint32_t k, const Device* d, Out out) { return relative(o, r, k, d, out, false); }
HRESULT WINAPI temporal_reading(Object* o, uint64_t time, const Device* d, Out out) {
    Hub* h = hub_of(o);
    int i = h->index(d);
    if (i < 0) return fail(out, E_INVALIDARG);
    std::lock_guard g(h->lock);
    auto& hs = h->slots[i].history;
    for (auto it = hs.rbegin(); it != hs.rend(); ++it)
        if ((*it)->timestamp <= time) return hand(out, *it);
    return fail(out, READING_NOT_FOUND);
}

HRESULT token(Hub* h, uint64_t* out) {
    if (out) *out = h->next_token++;
    return S_OK;
}
HRESULT WINAPI register_reading(Object* o, const void*, uint32_t, float, void*, const void*, uint64_t* out) {
    KL_WARN("input", "reading callbacks are never called");
    return token(hub_of(o), out);
}
HRESULT WINAPI register_device(Object* o, const Device* d, uint32_t kind, uint32_t status, uint32_t enumeration,
                               void* context, DeviceCallback cb, uint64_t* out) {
    if (!cb) return E_INVALIDARG;
    Hub* h = hub_of(o);
    Registration r{h->next_token++, d, kind, status, context, cb};
    std::vector<Event> found;
    {
        std::lock_guard g(h->lock);
        if (enumeration && (status & STATUS_CONNECTED)) {
            uint64_t now = now_us();
            for (size_t i = 0; i < DEVICES; ++i)
                if (h->slots[i].connected && h->matches(r, i)) found.push_back({r, i, now, STATUS_CONNECTED, 0});
        }
        KL_INFO("input", "device callback kind %x, enumeration %u, %zu devices", kind, enumeration, found.size());
        h->callbacks.push_back(r);
        if (enumeration == ASYNC_ENUMERATION) {
            h->pending.insert(h->pending.end(), found.begin(), found.end());
            found.clear();
        }
    }
    if (out) *out = r.token;
    h->deliver(found);  // blocking enumeration: on the caller's thread, outside our lock
    return S_OK;
}
HRESULT WINAPI register_other(Object* o, const void*, void*, const void*, uint64_t* out) { return token(hub_of(o), out); }
void forget(Hub* h, uint64_t t) {
    std::lock_guard g(h->lock);
    std::erase_if(h->callbacks, [&](auto& r) { return r.token == t; });
    std::erase_if(h->pending, [&](auto& e) { return e.r.token == t; });
}
void WINAPI stop_callback(Object* o, uint64_t t) { forget(hub_of(o), t); }
bool WINAPI unregister_callback(Object* o, uint64_t t, uint64_t timeout_us) {
    Hub* h = hub_of(o);
    forget(h, t);
    DWORD me = GetCurrentThreadId();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
    std::unique_lock g(h->lock);
    for (;;) {
        bool any = false;
        for (auto& [tok, th] : h->running) {
            if (tok != t) continue;
            if (th == me) return false;
            any = true;
        }
        if (!any) return true;
        if (h->returned.wait_until(g, deadline) == std::cv_status::timeout) return false;
    }
}
HRESULT WINAPI create_dispatcher(Object*, Out out) { return fail(out, E_NOTIMPL); }
HRESULT WINAPI create_aggregate(Object*, uint32_t, Out out) { return fail(out, E_NOTIMPL); }
HRESULT WINAPI find_by_id(Object* o, const uint8_t* id, Out out) {
    if (!id || !out) return fail(out, E_INVALIDARG);
    for (auto& d : hub_of(o)->devices)
        if (!memcmp(d.info.id, id, 32)) return *out = const_cast<Device*>(&d), S_OK;
    return fail(out, DEVICE_NOT_FOUND);
}
HRESULT WINAPI find_device(Object*, const void*, Out out) { return fail(out, DEVICE_NOT_FOUND); }
HRESULT WINAPI enable_oem(Object*, uint16_t, uint16_t, uint8_t, uint8_t) { return S_OK; }
void WINAPI set_focus_policy(Object*, uint32_t) {}

const void* const kInput[22] = {
    (void*)no_interface, (void*)add_ref, (void*)add_ref, (void*)timestamp, (void*)current_reading,
    (void*)next_reading, (void*)previous_reading, (void*)temporal_reading, (void*)register_reading,
    (void*)register_device, (void*)register_other, (void*)register_other, (void*)stop_callback,
    (void*)unregister_callback, (void*)create_dispatcher, (void*)create_aggregate, (void*)find_by_id,
    (void*)find_device, (void*)find_device, (void*)find_device, (void*)enable_oem, (void*)set_focus_policy};

// -------------------------------------------------------------- devices ----

const DeviceInfo* WINAPI device_info(const Device* d) { return &d->info; }
uint32_t WINAPI device_status(const Device* d) {
    std::lock_guard g(d->hub->lock);
    return d->hub->slots[d->index].connected ? STATUS_CONNECTED : 0;
}
void WINAPI set_rumble(const Device* d, const float* p) {
    if (d->index >= PADS) return;
    auto speed = [](float v) { return WORD((v < 0 ? 0 : v > 1 ? 1 : v) * 65535.0f); };
    XINPUT_VIBRATION v{p ? speed(p[0]) : WORD(0), p ? speed(p[1]) : WORD(0)};
    XInputSetState(DWORD(d->index), &v);
}
HRESULT WINAPI unsupported(const void*) { return E_NOTIMPL; }

const void* const kDevice[21] = {
    (void*)no_interface, (void*)add_ref, (void*)add_ref, (void*)device_info, (void*)device_status,
    (void*)unsupported, (void*)unsupported, (void*)unsupported, (void*)unsupported, (void*)unsupported,
    (void*)set_rumble, (void*)unsupported, (void*)unsupported, (void*)unsupported, (void*)unsupported,
    (void*)unsupported, (void*)unsupported, (void*)unsupported, (void*)unsupported, (void*)unsupported,
    (void*)unsupported};

// ------------------------------------------------------------- readings ----

uint32_t WINAPI input_kind(const Reading* r) { return r->kind(); }
uint64_t WINAPI sequence_number(const Reading* r, uint32_t kind) { return (r->kind() & kind) ? r->sequence : 0; }
uint64_t WINAPI reading_timestamp(const Reading* r) { return r->timestamp; }
void WINAPI reading_device(const Reading* r, const Device** out) { if (out) *out = r->device; }
uint32_t WINAPI key_count(const Reading* r) { return r->is_pad ? 0 : uint32_t(r->keys.size()); }
uint32_t WINAPI key_state(const Reading* r, uint32_t count, KeyState* out) {
    if (r->is_pad || !out) return 0;
    size_t n = r->keys.size() < count ? r->keys.size() : count;
    memcpy(out, r->keys.data(), n * sizeof(KeyState));
    return uint32_t(n);
}
bool WINAPI gamepad_state(const Reading* r, PadState* out) {
    if (!r->is_pad || !out) return false;
    *out = r->pad;
    return true;
}
uintptr_t WINAPI nothing(const void*) { return 0; }

extern const void* const kReading[25] = {
    (void*)no_interface, (void*)reading_add_ref, (void*)reading_release, (void*)input_kind, (void*)sequence_number,
    (void*)reading_timestamp, (void*)reading_device, (void*)nothing, (void*)nothing, (void*)nothing,
    (void*)nothing, (void*)nothing, (void*)nothing, (void*)nothing, (void*)key_count, (void*)key_state,
    (void*)nothing, (void*)nothing, (void*)nothing, (void*)nothing, (void*)nothing, (void*)nothing,
    (void*)gamepad_state, (void*)nothing, (void*)nothing};

Hub* g_hub = nullptr;

Hub* start_hub() {
    auto* h = new Hub();
    h->object = {kInput, h};
    for (size_t i = 0; i < DEVICES; ++i) {
        Device& d = h->devices[i];
        d.vtable = kDevice;
        d.hub = h;
        d.index = i;
        d.info = {};
        d.info.size = sizeof(DeviceInfo);
        memcpy(d.info.id, "keylime", 7);
        d.info.id[31] = uint8_t(i + 1);
        memcpy(d.info.root_id, d.info.id, 32);
        d.info.supported_input = kind_of(i);
        d.info.rumble_motors = i < PADS ? 3 : 0;
    }
    h->poll();
    std::thread([h] {
        SetThreadDescription(GetCurrentThread(), L"keylime-input");
        for (;;) {
            Sleep(8);
            h->poll();
        }
    }).detach();
    return h;
}

HRESULT WINAPI GameInputCreate(Object** out) {
    static std::once_flag once;
    std::call_once(once, [] { g_hub = start_hub(); });
    KL_INFO("input", "GameInputCreate");
    if (!out) return E_INVALIDARG;
    *out = &g_hub->object;
    return S_OK;
}

}  // namespace

KL_MODULE("gameinput", (std::vector<std::string>{"gameinput.dll"}),
          (std::vector<hle::Export>{KL_EXPORT_AS("#1", GameInputCreate)}));

}  // namespace kl::gameinput
