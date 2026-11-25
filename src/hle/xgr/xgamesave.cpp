// keylime: XGameSave, the user's game saves on the host under the title's
// data directory (port of VIctory's xgamesave.rs). A provider is a folder
// per user and service configuration, a container a folder in it. Updates
// are all or nothing: each writes a complete new version (v<N>.new/,
// unchanged blobs hard-linked), flushes it and commits with one rename
// (v<N>). A version's `manifest` holds the exact container name, display
// name and blob names (blob i is the file i), so no title name ever
// becomes a host file name.
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>

#include "hle/xgr/xgr.h"
#include "log.h"
#include "runtime.h"

namespace kl::xgr {
namespace {

namespace fs = std::filesystem;
const GUID PROVIDER = {0x704c3f58, 0xe629, 0x4cc2, {0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2}};
const GUID IID_2026 = {0x1bfff3af, 0xf14a, 0x40a3, {0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9}};

constexpr HRESULT INVALID_CONTAINER_NAME = HRESULT(0x80830001), NO_ACCESS = HRESULT(0x80830002),
                  OUT_OF_LOCAL_STORAGE = HRESULT(0x80830003), UPDATE_TOO_BIG = HRESULT(0x80830005),
                  QUOTA_EXCEEDED = HRESULT(0x80830006), BUFFER_TOO_SMALL = HRESULT(0x80830007), BLOB_NOT_FOUND = HRESULT(0x80830008);
constexpr uint64_t QUOTA = 256ull << 20;
constexpr size_t MAX_UPDATE = 16u << 20, BLOB = 24;
constexpr uint64_t XUID = 0x0009000000000001ull;

std::recursive_mutex g_saves;

std::string file_name(const std::string& name) {
    std::string out;
    for (unsigned char b : name) {
        if (isalnum(b) || b == '-' || b == '_') out.push_back(char(b));
        else {
            char h[4];
            snprintf(h, sizeof h, "%%%02X", b);
            out += h;
        }
    }
    return out;
}
fs::path container_dir(const fs::path& root, const std::string& name) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char b : name) h = (h ^ b) * 0x100000001b3ull;
    std::string readable = file_name(name).substr(0, 100);
    char hex[24];
    snprintf(hex, sizeof hex, "~%016llx", (unsigned long long)h);
    return root / (readable + hex);
}
bool valid_container(const std::string& name) {
    if (name.empty() || name.size() > 256) return false;
    size_t slash = name.rfind('/');
    std::string path = slash == std::string::npos ? "" : name.substr(0, slash), file = slash == std::string::npos ? name : name.substr(slash + 1);
    for (unsigned char b : path)
        if (!isalnum(b) && b != '_' && b != '/') return false;
    if (file.empty() || file.back() == '.' || file.find("..") != std::string::npos) return false;
    for (unsigned char b : file)
        if (!isalnum(b) && b != '_' && b != '.' && b != '-') return false;
    return true;
}

struct Version {
    fs::path dir;
    uint64_t number = 0;
    std::string name, display;
    std::vector<std::string> blobs;
    fs::path blob(size_t i) const { return dir / std::to_string(i); }
    uint64_t size(size_t i) const {
        std::error_code e;
        auto n = fs::file_size(blob(i), e);
        return e ? 0 : n;
    }
};

std::vector<uint8_t> read_file(const fs::path& p, bool& ok) {
    std::ifstream f(p, std::ios::binary);
    ok = bool(f);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
bool write_file(const fs::path& p, const void* data, size_t n) {
    HANDLE h = CreateFileW(p.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    DWORD w = 0;
    bool ok = !n || (WriteFile(h, data, DWORD(n), &w, nullptr) && w == n);
    ok = ok && FlushFileBuffers(h);
    CloseHandle(h);
    return ok;
}

std::vector<uint8_t> manifest(const std::string& name, const std::string& display, const std::vector<std::string>& blobs) {
    std::vector<uint8_t> out;
    auto add = [&](const std::string& s) {
        uint32_t n = uint32_t(s.size());
        out.insert(out.end(), reinterpret_cast<uint8_t*>(&n), reinterpret_cast<uint8_t*>(&n) + 4);
        out.insert(out.end(), s.begin(), s.end());
    };
    add(name), add(display);
    for (auto& b : blobs) add(b);
    return out;
}

// The committed version of a container folder (leftovers cleaned on sight).
std::optional<Version> current(const fs::path& dir) {
    std::error_code e;
    if (!fs::exists(dir, e)) return std::nullopt;
    std::vector<uint64_t> committed;
    for (auto& entry : fs::directory_iterator(dir, e)) {
        std::string f = entry.path().filename().string();
        if (f.size() > 4 && f.substr(f.size() - 4) == ".new") fs::remove_all(entry.path(), e);
        else if (f.size() > 1 && f[0] == 'v' && f.find_first_not_of("0123456789", 1) == std::string::npos)
            committed.push_back(std::stoull(f.substr(1)));
    }
    if (committed.empty()) return std::nullopt;
    std::sort(committed.begin(), committed.end());
    for (size_t i = 0; i + 1 < committed.size(); ++i) fs::remove_all(dir / ("v" + std::to_string(committed[i])), e);
    Version v;
    v.number = committed.back();
    v.dir = dir / ("v" + std::to_string(v.number));
    bool ok;
    auto m = read_file(v.dir / "manifest", ok);
    if (!ok) return std::nullopt;
    std::vector<std::string> names;
    for (size_t at = 0; at + 4 <= m.size();) {
        uint32_t n;
        memcpy(&n, m.data() + at, 4);
        at += 4;
        if (at + n > m.size()) return std::nullopt;
        names.emplace_back(reinterpret_cast<const char*>(m.data() + at), n);
        at += n;
    }
    if (names.size() < 2) return std::nullopt;
    v.name = names[0], v.display = names[1];
    v.blobs.assign(names.begin() + 2, names.end());
    return v;
}
std::optional<Version> current_named(const fs::path& dir, const std::string& name) {
    auto v = current(dir);
    if (v && v->name != name) return std::nullopt;
    return v;
}

struct Source {
    std::string name;
    const std::vector<uint8_t>* data;  // or kept:
    fs::path kept;
};
bool commit(const fs::path& dir, uint64_t number, const std::string& name, const std::string& display, const std::vector<Source>& blobs) {
    std::error_code e;
    fs::path staged = dir / ("v" + std::to_string(number) + ".new");
    fs::remove_all(staged, e);
    fs::create_directories(staged, e);
    std::vector<std::string> names;
    for (size_t i = 0; i < blobs.size(); ++i) {
        fs::path p = staged / std::to_string(i);
        if (blobs[i].data) {
            if (!write_file(p, blobs[i].data->data(), blobs[i].data->size())) return false;
        } else {
            fs::create_hard_link(blobs[i].kept, p, e);
            if (e) {
                e.clear();
                fs::copy_file(blobs[i].kept, p, e);
                if (e) return false;
            }
        }
        names.push_back(blobs[i].name);
    }
    auto m = manifest(name, display, names);
    if (!write_file(staged / "manifest", m.data(), m.size())) return false;
    fs::rename(staged, dir / ("v" + std::to_string(number)), e);
    return !e;
}

std::vector<std::pair<std::string, fs::path>> containers(const fs::path& root) {
    std::vector<std::pair<std::string, fs::path>> out;
    std::error_code e;
    for (auto& entry : fs::directory_iterator(root, e)) {
        if (!entry.is_directory(e)) continue;
        std::string f = entry.path().filename().string();
        if (f.size() > 5 && f.substr(f.size() - 5) == ".gone") fs::remove_all(entry.path(), e);
        else if (f.find('~') != std::string::npos)
            if (auto v = current(entry.path())) out.push_back({v->name, entry.path()});
    }
    std::sort(out.begin(), out.end());
    return out;
}
uint64_t used_bytes(const fs::path& root) {
    uint64_t n = 0;
    for (auto& [name, dir] : containers(root))
        if (auto v = current_named(dir, name))
            for (size_t i = 0; i < v->blobs.size(); ++i) n += v->size(i);
    return n;
}

struct Provider {
    fs::path root;
};
struct Container {
    std::string name;
    fs::path dir;
};
struct Update {
    std::string name;
    fs::path dir;
    std::optional<std::string> display;
    std::vector<std::pair<std::string, std::optional<std::vector<uint8_t>>>> changes;

    HRESULT submit() const {
        std::lock_guard g(g_saves);
        auto cur = current(dir);
        if (cur && cur->name != name) return NO_ACCESS;
        std::vector<Source> blobs;
        if (cur)
            for (size_t i = 0; i < cur->blobs.size(); ++i) blobs.push_back({cur->blobs[i], nullptr, cur->blob(i)});
        uint64_t replaced = 0;
        for (auto& [n, data] : changes) {
            for (size_t i = 0; i < blobs.size();)
                if (blobs[i].name == n) {
                    if (cur && !blobs[i].data)
                        for (size_t k = 0; k < cur->blobs.size(); ++k)
                            if (cur->blobs[k] == n) replaced += cur->size(k);
                    blobs.erase(blobs.begin() + i);
                } else {
                    ++i;
                }
            if (data) blobs.push_back({n, &*data, {}});
        }
        std::sort(blobs.begin(), blobs.end(), [](const Source& a, const Source& b) { return a.name < b.name; });
        size_t written = 0;
        for (auto& b : blobs)
            if (b.data) written += b.data->size();
        if (written > MAX_UPDATE) return UPDATE_TOO_BIG;
        uint64_t used = used_bytes(dir.parent_path());
        if (written > (used < QUOTA ? QUOTA - used : 0) + replaced) return QUOTA_EXCEEDED;
        std::string disp = display ? *display : cur ? cur->display : "";
        std::error_code e;
        fs::create_directories(dir, e);
        if (!commit(dir, cur ? cur->number + 1 : 1, name, disp, blobs)) return OUT_OF_LOCAL_STORAGE;
        if (cur) fs::remove_all(cur->dir, e);
        return S_OK;
    }
};

HRESULT delete_container(const fs::path& root, const std::string& name) {
    std::lock_guard g(g_saves);
    fs::path dir = container_dir(root, name);
    if (!current_named(dir, name)) return S_OK;
    fs::path gone = dir;
    gone += ".gone";
    std::error_code e;
    fs::remove_all(gone, e);
    fs::rename(dir, gone, e);
    if (e) return NO_ACCESS;
    fs::remove_all(gone, e);
    return S_OK;
}

HRESULT read_blobs_of(const Container& c, const std::vector<std::string>* wanted, std::vector<std::pair<std::string, std::vector<uint8_t>>>& out) {
    std::lock_guard g(g_saves);
    auto v = current_named(c.dir, c.name);
    if (!v) return BLOB_NOT_FOUND;
    const std::vector<std::string>& names = wanted ? *wanted : v->blobs;
    for (auto& n : names) {
        auto it = std::find(v->blobs.begin(), v->blobs.end(), n);
        if (it == v->blobs.end()) return BLOB_NOT_FOUND;
        bool ok;
        out.push_back({n, read_file(v->blob(size_t(it - v->blobs.begin())), ok)});
        if (!ok) return NO_ACCESS;
    }
    return S_OK;
}
// Blob entries {name ptr, u32 size, pad, data ptr} then names and data, addresses from `base`.
std::vector<uint8_t> layout(const std::vector<std::pair<std::string, std::vector<uint8_t>>>& blobs, uintptr_t base) {
    std::vector<uint8_t> out(blobs.size() * BLOB, 0);
    for (size_t i = 0; i < blobs.size(); ++i) {
        uint64_t name_at = base + out.size();
        out.insert(out.end(), blobs[i].first.begin(), blobs[i].first.end());
        out.push_back(0);
        uint64_t data_at = base + out.size();
        out.insert(out.end(), blobs[i].second.begin(), blobs[i].second.end());
        uint32_t n = uint32_t(blobs[i].second.size());
        memcpy(&out[i * BLOB], &name_at, 8);
        memcpy(&out[i * BLOB + 8], &n, 4);
        memcpy(&out[i * BLOB + 16], &data_at, 8);
    }
    return out;
}
std::vector<std::string> names_of(const char* const* names, uint32_t count) {
    std::vector<std::string> v;
    for (uint32_t i = 0; i < count; ++i) v.push_back(names[i] ? names[i] : "");
    return v;
}

struct ContainerInfo {
    const char* name;
    const char* display_name;
    uint32_t blob_count;
    uint64_t total_size;
    int64_t last_modified;
    bool needs_sync;
};
struct BlobInfo {
    const char* name;
    uint32_t size;
};
using ContainerCallback = bool(WINAPI*)(const ContainerInfo*, void*);
using BlobCallback = bool(WINAPI*)(const BlobInfo*, void*);

void report(const fs::path& root, const std::string* only, void* ctx, ContainerCallback cb) {
    struct Row {
        std::string name, display;
        uint32_t count;
        uint64_t size;
        int64_t modified;
    };
    std::vector<Row> rows;
    {
        std::lock_guard g(g_saves);
        for (auto& [name, dir] : containers(root)) {
            if (only && name != *only) continue;
            auto v = current_named(dir, name);
            if (!v) continue;
            uint64_t size = 0;
            for (size_t i = 0; i < v->blobs.size(); ++i) size += v->size(i);
            std::error_code e;
            auto t = fs::last_write_time(v->dir / "manifest", e);
            int64_t secs = e ? 0 : std::chrono::duration_cast<std::chrono::seconds>(std::chrono::clock_cast<std::chrono::system_clock>(t).time_since_epoch()).count();
            rows.push_back({v->name, v->display, uint32_t(v->blobs.size()), size, secs});
        }
    }
    for (auto& r : rows) {
        ContainerInfo info{r.name.c_str(), r.display.c_str(), r.count, r.size, r.modified, false};
        if (!cb(&info, ctx)) break;
    }
}

using This = Object*;
char INIT, QUOTA_CALL, DELETE_CALL, READ, SUBMIT;

HRESULT WINAPI initialize_async(This, const void*, const char* config, bool, AsyncBlock* b) {
    std::string c = config && *config ? config : "default";
    char user[24];
    snprintf(user, sizeof user, "%016llX", (unsigned long long)XUID);
    fs::path root = runtime().data_dir / "saves" / user / file_name(c);
    std::error_code e;
    fs::create_directories(root, e);
    if (e) return finished(b, &INIT, NO_ACCESS);
    KL_INFO("xgr", "game saves for configuration %s in %ls", c.c_str(), root.c_str());
    auto* p = new Provider{root};
    return finished(b, &INIT, S_OK, bytes(p));
}
HRESULT WINAPI initialize_result(This, AsyncBlock* b, Provider** out) { return result_as(b, &INIT, out); }
void WINAPI close_provider(This, Provider* p) { delete p; }
HRESULT WINAPI quota_async(This, Provider* p, AsyncBlock* b) {
    if (!p) return finished(b, &QUOTA_CALL, E_INVALIDARG_);
    std::lock_guard g(g_saves);
    uint64_t used = used_bytes(p->root);
    int64_t left = int64_t(used < QUOTA ? QUOTA - used : 0);
    return finished(b, &QUOTA_CALL, S_OK, bytes(left));
}
HRESULT WINAPI quota_result(This, AsyncBlock* b, int64_t* out) { return result_as(b, &QUOTA_CALL, out); }
HRESULT WINAPI delete_async(This, Provider* p, const char* name, AsyncBlock* b) {
    HRESULT hr = !p || !name ? E_INVALIDARG_ : !valid_container(name) ? INVALID_CONTAINER_NAME : delete_container(p->root, name);
    return finished(b, &DELETE_CALL, hr);
}
HRESULT WINAPI delete_result(This, AsyncBlock* b) { return async_result(b, &DELETE_CALL, 0, nullptr, nullptr); }
HRESULT WINAPI container_info(This, Provider* p, const char* name, void* ctx, ContainerCallback cb) {
    if (!p || !name || !cb) return E_INVALIDARG_;
    if (!valid_container(name)) return INVALID_CONTAINER_NAME;
    std::string n = name;
    report(p->root, &n, ctx, cb);
    return S_OK;
}
HRESULT WINAPI enumerate_containers(This, Provider* p, void* ctx, ContainerCallback cb) {
    if (!p || !cb) return E_INVALIDARG_;
    report(p->root, nullptr, ctx, cb);
    return S_OK;
}
HRESULT WINAPI create_container(This, Provider* p, const char* name, Container** out) {
    if (!p || !name) return E_INVALIDARG_;
    if (!valid_container(name)) return INVALID_CONTAINER_NAME;
    return put(out, new Container{name, container_dir(p->root, name)});
}
void WINAPI close_container(This, Container* c) { delete c; }
HRESULT WINAPI enumerate_blobs(This, Container* c, const char* prefix, void* ctx, BlobCallback cb) {
    if (!c || !cb) return E_INVALIDARG_;
    std::string pre = prefix ? prefix : "";
    std::vector<std::pair<std::string, uint64_t>> blobs;
    {
        std::lock_guard g(g_saves);
        if (auto v = current_named(c->dir, c->name))
            for (size_t i = 0; i < v->blobs.size(); ++i) blobs.push_back({v->blobs[i], v->size(i)});
    }
    for (auto& [n, size] : blobs) {
        if (n.rfind(pre, 0) != 0) continue;
        BlobInfo info{n.c_str(), uint32_t(size)};
        if (!cb(&info, ctx)) break;
    }
    return S_OK;
}
HRESULT WINAPI read_blobs(This, Container* c, const char* const* names, uint32_t* count, size_t size, uint8_t* out) {
    if (!c) return E_INVALIDARG_;
    std::vector<std::string> wanted;
    if (names) wanted = names_of(names, count ? *count : 0);
    std::vector<std::pair<std::string, std::vector<uint8_t>>> blobs;
    HRESULT hr = read_blobs_of(*c, names ? &wanted : nullptr, blobs);
    if (FAILED(hr)) return hr;
    auto b = layout(blobs, uintptr_t(out));
    if (!out || size < b.size()) return BUFFER_TOO_SMALL;
    memcpy(out, b.data(), b.size());
    return put(count, uint32_t(blobs.size()));
}
HRESULT WINAPI read_async(This, Container* c, const char* const* names, uint32_t count, AsyncBlock* b) {
    if (!c) return finished(b, &READ, E_INVALIDARG_);
    std::vector<std::string> wanted;
    if (names) wanted = names_of(names, count);
    std::vector<std::pair<std::string, std::vector<uint8_t>>> blobs;
    HRESULT hr = read_blobs_of(*c, names ? &wanted : nullptr, blobs);
    if (FAILED(hr)) return finished(b, &READ, hr);
    return finished(b, &READ, S_OK, layout(blobs, 0));
}
HRESULT WINAPI read_result(This, AsyncBlock* b, size_t size, uint8_t* out, uint32_t* count) {
    size_t used = 0;
    HRESULT hr = async_result(b, &READ, size, out, &used);
    if (hr == E_NOT_SUFFICIENT_BUFFER_) return BUFFER_TOO_SMALL;
    if (FAILED(hr)) return hr;
    size_t n = 0;
    if (used) {
        uint64_t first;
        memcpy(&first, out, 8);
        n = size_t(first / BLOB);  // the first name follows the entries
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t field : {size_t(0), size_t(16)}) {
            uint64_t v;
            memcpy(&v, out + i * BLOB + field, 8);
            v += uint64_t(uintptr_t(out));
            memcpy(out + i * BLOB + field, &v, 8);
        }
    return put(count, uint32_t(n));
}
HRESULT WINAPI create_update(This, Container* c, const char* display, Update** out) {
    if (!c) return E_INVALIDARG_;
    auto* u = new Update{c->name, c->dir, display ? std::optional<std::string>(display) : std::nullopt, {}};
    return put(out, u);
}
void WINAPI close_update(This, Update* u) { delete u; }
HRESULT WINAPI write_blob(This, Update* u, const char* name, const uint8_t* data, size_t size) {
    if (!u || !name || !*name || (!data && size)) return E_INVALIDARG_;
    if (size > MAX_UPDATE) return UPDATE_TOO_BIG;
    u->changes.push_back({name, std::vector<uint8_t>(data, data + size)});
    return S_OK;
}
HRESULT WINAPI delete_blob(This, Update* u, const char* name) {
    if (!u || !name || !*name) return E_INVALIDARG_;
    u->changes.push_back({name, std::nullopt});
    return S_OK;
}
HRESULT WINAPI submit_async(This, Update* u, AsyncBlock* b) { return finished(b, &SUBMIT, u ? u->submit() : E_INVALIDARG_); }
HRESULT WINAPI submit_result(This, AsyncBlock* b) { return async_result(b, &SUBMIT, 0, nullptr, nullptr); }

struct Init {
    Init() {
        void** t = make_table("game save", 30);
        auto s = [&](size_t off, auto fn) { set(t, off, reinterpret_cast<void*>(fn)); };
        s(0x20, &initialize_async), s(0x28, &initialize_result), s(0x30, &close_provider), s(0x40, &quota_async);
        s(0x48, &quota_result), s(0x58, &delete_async), s(0x60, &delete_result), s(0x68, &container_info);
        s(0x70, &enumerate_containers), s(0x80, &create_container), s(0x88, &close_container), s(0x98, &enumerate_blobs);
        s(0xa0, &read_blobs), s(0xa8, &read_async), s(0xb0, &read_result), s(0xb8, &create_update), s(0xc0, &close_update);
        s(0xc8, &write_blob), s(0xd0, &delete_blob), s(0xe0, &submit_async), s(0xe8, &submit_result);
        static Object o{t, "game save"};
        add_interfaces({{PROVIDER, PROVIDER, &o}, {PROVIDER, IID_2026, &o}});
    }
} g_init;

}  // namespace
}  // namespace kl::xgr
