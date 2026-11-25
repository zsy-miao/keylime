// keylime: the console's DirectStorage (XCRDAPI.dll). A factory opens files
// and makes request queues and status arrays the title polls. Requests of
// every queue run on one worker pool (in parallel, started in queue order);
// completion stays in order: a status entry is written once every request
// enqueued before it on its queue has finished, with the first failure since
// the previous entry. Slot layout from VIctory (dstorage.rs).
#include <windows.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "hle/files.h"
#include "hle/hle.h"
#include "hle/xmem.h"
#include "log.h"
#include "stats.h"
#include "miniz.h"

namespace kl::dstorage {
namespace {

constexpr GUID FACTORY = {0x62959d11, 0xd45f, 0x4884, {0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae}};
constexpr HRESULT END_OF_FILE = HRESULT(0x80070026), INVALID_DATA = HRESULT(0x8007000D);
constexpr uint64_t ZLIB = 1, BCPACK = 0xf << 1, PAGE_DESTINATION = 1ull << 21, MEMORY_SOURCE = 1ull << 22;

std::string guid_text(const GUID* g) {
    if (!g) return "null";
    char b[64];
    snprintf(b, sizeof b, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", g->Data1, g->Data2, g->Data3,
             g->Data4[0], g->Data4[1], g->Data4[2], g->Data4[3], g->Data4[4], g->Data4[5], g->Data4[6], g->Data4[7]);
    return b;
}

// NOTE: title code reads word 0 as the method table, so our objects are plain
// structs (no C++ virtuals) with an explicit header.
struct Header {
    const void* const* vtable;
    const char* area;
    std::atomic<uint32_t> refs;
};

template <class T>
struct Instance {
    Header h;
    T state;
};

template <class T>
uint32_t WINAPI add_ref(Instance<T>* o) { return ++o->h.refs; }
template <class T>
uint32_t WINAPI release(Instance<T>* o) {
    uint32_t left = --o->h.refs;
    if (!left) delete o;
    return left;
}
uint32_t WINAPI static_ref(void*) { return 1; }

// A slot nobody implemented yet: says which.
template <int SLOT>
HRESULT WINAPI missing(Header* self) {
    KL_WARN("dstorage", "%s: method slot 0x%x not implemented", self ? self->area : "?", SLOT * 8);
    return E_NOTIMPL;
}

HRESULT put(void** out, void* v) {
    if (!out) return E_POINTER;
    *out = v;
    return S_OK;
}

// ---------------------------------------------------------------- types ----

struct File {
    HANDLE handle;
    ~File() { CloseHandle(handle); }
};
struct StatusArray {
    std::vector<std::atomic<HRESULT>> entries;
    explicit StatusArray(uint32_t n) : entries(n) {
        for (auto& e : entries) e = S_OK;
    }
};

#pragma pack(push, 8)
struct Request {
    uint64_t options;
    void* destination;     // an address, or with PAGE_DESTINATION an array of 64 KiB page numbers
    uint64_t page_offset;  // low 16 bits: where in the first page
    uint32_t destination_size;
    const void* source;    // our File instance, or with MEMORY_SOURCE an address
    uint64_t offset;
    uint32_t source_size;
    uint32_t intermediate_size;
    uint64_t tag;
    const char* name;
};
#pragma pack(pop)
static_assert(sizeof(Request) == 0x48);

struct QueueDesc {
    uint64_t source_type;
    uint16_t capacity;
    int8_t priority;
    const char* name;
    uint64_t unknown;
};

struct Cmd {
    bool is_status;
    Request req;
    Instance<StatusArray>* array;
    uint32_t index;
    uint64_t seq;
};

struct Lane {
    uint64_t id;
    uint16_t capacity;
    std::vector<Cmd> unsubmitted;
    std::deque<Cmd> reads;        // submitted, not started
    std::deque<Cmd> statuses;     // submitted, waiting for earlier reads
    std::map<uint64_t, HRESULT> running;  // started reads by seq (result S_FALSE while running)
    HRESULT first_failure = S_OK;
    bool closed = false;
    size_t held() const { return unsubmitted.size() + reads.size() + statuses.size() + running.size(); }
};

struct Scheduler {
    std::mutex lock;
    std::condition_variable work, finished;
    std::map<uint64_t, Lane> lanes;
    uint64_t next_id = 1, next_seq = 1, rr = 0;
    std::atomic<uint64_t> requests{0}, bytes{0}, failures{0};

    void submit(Lane& l) {
        for (auto& c : l.unsubmitted) (c.is_status ? l.statuses : l.reads).push_back(c);
        l.unsubmitted.clear();
    }
    // Status entries whose earlier reads have all finished.
    void complete(Lane& l) {
        while (!l.statuses.empty()) {
            uint64_t s = l.statuses.front().seq;
            bool earlier_open = (!l.reads.empty() && l.reads.front().seq < s);
            for (auto& [seq, hr] : l.running)
                if (seq < s && hr == S_FALSE) earlier_open = true;
            if (earlier_open) break;
            // failures of finished reads before this entry
            for (auto it = l.running.begin(); it != l.running.end() && it->first < s;) {
                if (FAILED(it->second) && l.first_failure == S_OK) l.first_failure = it->second;
                it = l.running.erase(it);
            }
            Cmd st = l.statuses.front();
            l.statuses.pop_front();
            st.array->state.entries[st.index].store(l.first_failure, std::memory_order_release);
            l.first_failure = S_OK;
            release(st.array);
        }
        // No status waiting: finished reads fold into the next entry's result.
        if (l.statuses.empty())
            for (auto it = l.running.begin(); it != l.running.end();) {
                if (it->second == S_FALSE) { ++it; continue; }
                if (FAILED(it->second) && l.first_failure == S_OK) l.first_failure = it->second;
                it = l.running.erase(it);
            }
    }
};

Scheduler* g_sched = nullptr;

HRESULT from_win32(DWORD e) { return HRESULT(0x80070000 | (e & 0xffff)); }

bool read_at(HANDLE h, uint64_t offset, void* out, uint32_t len, HRESULT& hr) {
    uint32_t done = 0;
    while (done < len) {
        OVERLAPPED ov{};
        uint64_t at = offset + done;
        ov.Offset = DWORD(at);
        ov.OffsetHigh = DWORD(at >> 32);
        DWORD got = 0;
        if (!ReadFile(h, static_cast<uint8_t*>(out) + done, len - done, &got, &ov)) {
            hr = from_win32(GetLastError());
            return false;
        }
        if (!got) {
            hr = END_OF_FILE;
            return false;
        }
        done += got;
    }
    return true;
}

HRESULT execute(const Request& r) {
    if (r.options & ~(ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE)) return E_NOTIMPL;
    if (!r.source || !r.destination) return E_INVALIDARG;
    const unsigned bc = unsigned((r.options & BCPACK) >> 1);
    if (bc) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) KL_WARN("dstorage", "BCPack requests fail for now (BC format %u)", bc);
        return E_NOTIMPL;
    }
    // Source bytes.
    std::vector<uint8_t> src_buf;
    const uint8_t* src = nullptr;
    if (r.options & MEMORY_SOURCE) {
        src = static_cast<const uint8_t*>(r.source) + 0;
    } else {
        auto* f = static_cast<const Instance<File>*>(r.source);
        src_buf.resize(r.source_size);
        HRESULT hr = S_OK;
        if (!read_at(f->state.handle, r.offset, src_buf.data(), r.source_size, hr)) return hr;
        src = src_buf.data();
    }
    // Output bytes and size.
    const bool zlib = r.options & ZLIB;
    size_t out_len = zlib ? r.destination_size : r.source_size;
    if (!zlib && r.source_size > r.destination_size) return E_INVALIDARG;
    auto fill = [&](uint8_t* out) -> HRESULT {
        if (!zlib) {
            memcpy(out, src, r.source_size);
            return S_OK;
        }
        size_t n = tinfl_decompress_mem_to_mem(out, r.destination_size, src, r.source_size, TINFL_FLAG_PARSE_ZLIB_HEADER);
        return n == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED ? INVALID_DATA : S_OK;
    };
    if (!(r.options & PAGE_DESTINATION)) return fill(static_cast<uint8_t*>(r.destination));
    size_t offset = size_t(r.page_offset & 0xffff);
    std::vector<uint8_t> tmp(out_len);
    HRESULT hr = fill(tmp.data());
    if (FAILED(hr)) return hr;
    size_t pages = (offset + out_len + xmem::kPage - 1) / xmem::kPage;
    return xmem::write_physical(static_cast<const uint64_t*>(r.destination), pages, offset, tmp.data(), out_len) ? S_OK : E_INVALIDARG;
}

void worker() {
    SetThreadDescription(GetCurrentThread(), L"keylime-dstorage");
    Scheduler& s = *g_sched;
    std::unique_lock g(s.lock);
    for (;;) {
        // Round robin over lanes with a read to start.
        Lane* pick = nullptr;
        if (!s.lanes.empty()) {
            auto it = s.lanes.upper_bound(s.rr);
            for (size_t n = 0; n < s.lanes.size(); ++n, ++it) {
                if (it == s.lanes.end()) it = s.lanes.begin();
                if (!it->second.closed && !it->second.reads.empty()) {
                    pick = &it->second;
                    break;
                }
            }
        }
        if (!pick) {
            s.work.wait(g);
            continue;
        }
        s.rr = pick->id;
        Cmd c = pick->reads.front();
        pick->reads.pop_front();
        pick->running[c.seq] = S_FALSE;
        uint64_t lane_id = pick->id;
        g.unlock();
        HRESULT hr = execute(c.req);
        stats::add(stats::StorageRequests);
        stats::add(stats::StorageBytes, c.req.destination_size);
        if (FAILED(hr)) stats::add(stats::StorageFailures);
        ++s.requests;
        s.bytes += c.req.destination_size;
        if (FAILED(hr) && s.failures++ < 16)
            KL_WARN("dstorage", "request failed %08lx: options %llx, %u -> %u bytes, name %s", hr,
                    (unsigned long long)c.req.options, c.req.source_size, c.req.destination_size,
                    c.req.name ? c.req.name : "");
        g.lock();
        if (auto it = s.lanes.find(lane_id); it != s.lanes.end()) {
            it->second.running[c.seq] = hr;
            s.complete(it->second);
        }
        s.finished.notify_all();
    }
}

Scheduler& sched() {
    static std::once_flag once;
    std::call_once(once, [] {
        g_sched = new Scheduler();
        unsigned n = std::thread::hardware_concurrency() / 2;
        if (n < 2) n = 2;
        if (n > 8) n = 8;
        for (unsigned i = 0; i < n; ++i) std::thread(worker).detach();
        KL_INFO("dstorage", "%u workers", n);
    });
    return *g_sched;
}

// --------------------------------------------------------------- queues ----

struct Queue {
    uint64_t id;
};

void close_lane(uint64_t id) {
    Scheduler& s = sched();
    std::unique_lock g(s.lock);
    auto it = s.lanes.find(id);
    if (it == s.lanes.end()) return;
    Lane& l = it->second;
    l.closed = true;
    for (auto& c : l.unsubmitted) if (c.is_status) release(c.array);
    for (auto& c : l.statuses) release(c.array);
    l.unsubmitted.clear();
    l.reads.clear();
    l.statuses.clear();
    s.finished.wait(g, [&] {
        for (auto& [seq, hr] : l.running) if (hr == S_FALSE) return false;
        return true;
    });
    s.lanes.erase(id);
    s.finished.notify_all();
}

void enqueue(Instance<Queue>* q, Cmd c) {
    Scheduler& s = sched();
    std::unique_lock g(s.lock);
    for (;;) {
        auto it = s.lanes.find(q->state.id);
        if (it == s.lanes.end() || it->second.closed) {
            if (c.is_status) release(c.array);
            return;
        }
        if (it->second.held() + 1 < std::max<size_t>(it->second.capacity, 2)) break;
        s.finished.wait(g);
    }
    Lane& l = s.lanes[q->state.id];
    c.seq = s.next_seq++;
    l.unsubmitted.push_back(c);
    // The console submits on its own once half the queue is enqueued.
    if (l.unsubmitted.size() >= std::max<size_t>(l.capacity / 2, 1)) {
        s.submit(l);
        s.complete(l);
        s.work.notify_all();
    }
}

void WINAPI enqueue_request(Instance<Queue>* q, const Request* r) {
    if (!r) return;
    Cmd c{};
    c.req = *r;
    enqueue(q, c);
}
void WINAPI enqueue_status(Instance<Queue>* q, Instance<StatusArray>* a, uint32_t index) {
    if (!a || index >= a->state.entries.size()) return;
    a->state.entries[index].store(E_PENDING, std::memory_order_release);
    add_ref(a);
    Cmd c{};
    c.is_status = true;
    c.array = a;
    c.index = index;
    enqueue(q, c);
}
void WINAPI submit(Instance<Queue>* q) {
    Scheduler& s = sched();
    std::lock_guard g(s.lock);
    auto it = s.lanes.find(q->state.id);
    if (it == s.lanes.end()) return;
    s.submit(it->second);
    s.complete(it->second);
    s.work.notify_all();
}
void WINAPI close(Instance<Queue>* q) { close_lane(q->state.id); }
uint32_t WINAPI queue_release(Instance<Queue>* q) {
    uint32_t left = --q->h.refs;
    if (!left) {
        close_lane(q->state.id);
        delete q;
    }
    return left;
}

const void* const kQueue[12] = {
    (void*)missing<0>, (void*)add_ref<Queue>, (void*)queue_release, (void*)enqueue_request, (void*)enqueue_status,
    (void*)missing<5>, (void*)submit, (void*)missing<7>, (void*)close, (void*)missing<9>, (void*)missing<10>,
    (void*)missing<11>};

// -------------------------------------------------------- status arrays ----

HRESULT WINAPI status_result(Instance<StatusArray>* a, uint32_t i) {
    return i < a->state.entries.size() ? a->state.entries[i].load(std::memory_order_acquire) : E_INVALIDARG;
}
bool WINAPI is_complete(Instance<StatusArray>* a, uint32_t i) { return status_result(a, i) != E_PENDING; }
const void* const kStatus[5] = {(void*)missing<0>, (void*)add_ref<StatusArray>, (void*)release<StatusArray>,
                                (void*)is_complete, (void*)status_result};

const void* const kFile[8] = {(void*)missing<0>, (void*)add_ref<File>, (void*)release<File>, (void*)missing<3>,
                              (void*)missing<4>, (void*)missing<5>, (void*)missing<6>, (void*)missing<7>};

// -------------------------------------------------------------- factory ----

HRESULT WINAPI open_file(void*, const wchar_t* path, const GUID* iid, void** out) {
    if (!path) return E_INVALIDARG;
    std::wstring host;
    const wchar_t* p = files::translate(path, host) ? host.c_str() : path;
    HANDLE h = CreateFileW(p, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        KL_WARN("dstorage", "opening %ls failed: %lu", path, e);
        return from_win32(e);
    }
    KL_TRACE("dstorage", "file opened %ls", path);
    stats::add(stats::FilesOpened);
    auto* f = new Instance<File>{{kFile, "dstorage file", {1}}, {h}};
    (void)iid;
    return put(out, f);
}

HRESULT WINAPI create_status_array(void*, uint32_t capacity, const char* name, const GUID*, void** out) {
    KL_INFO("dstorage", "status array %u (%s)", capacity, name ? name : "");
    auto* a = new Instance<StatusArray>{{kStatus, "dstorage status array", {1}}, StatusArray(capacity)};
    return put(out, a);
}

HRESULT WINAPI create_queue(void*, const QueueDesc* d, const GUID* iid, void** out) {
    if (!d) return E_INVALIDARG;
    KL_INFO("dstorage", "queue %s: source %llu, capacity %u, priority %d", d->name ? d->name : "",
            (unsigned long long)d->source_type, d->capacity, d->priority);
    Scheduler& s = sched();
    uint64_t id;
    {
        std::lock_guard g(s.lock);
        id = s.next_id++;
        Lane& l = s.lanes[id];
        l.id = id;
        l.capacity = d->capacity;
    }
    (void)iid;
    return put(out, new Instance<Queue>{{kQueue, "dstorage queue", {1}}, {id}});
}

HRESULT WINAPI set_staging_buffer_size(void*, uint32_t size) {
    KL_INFO("dstorage", "staging buffer %u bytes", size);
    return S_OK;
}

const void* const kFactory[16] = {
    (void*)missing<0>, (void*)static_ref, (void*)static_ref, (void*)missing<3>, (void*)open_file,
    (void*)create_status_array, (void*)missing<6>, (void*)missing<7>, (void*)set_staging_buffer_size,
    (void*)missing<9>, (void*)create_queue, (void*)missing<11>, (void*)missing<12>, (void*)missing<13>,
    (void*)missing<14>, (void*)missing<15>};
Header g_factory{kFactory, "dstorage factory", {1}};

HRESULT WINAPI DStorageGetFactory(const GUID* iid, void** out) {
    if (iid && *iid == FACTORY) return put(out, &g_factory);
    KL_WARN("dstorage", "DStorageGetFactory: unknown interface %s", guid_text(iid).c_str());
    return E_NOINTERFACE;
}

}  // namespace

KL_MODULE("dstorage", (std::vector<std::string>{"xcrdapi.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(DStorageGetFactory)}));

}  // namespace kl::dstorage
