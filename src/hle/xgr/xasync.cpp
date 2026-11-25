// keylime: XTaskQueue and XAsync, the runtime's callback queues and async
// call blocks (port of VIctory's xasync.rs). A queue has a work port and a
// completion port, each manual (the title dispatches), thread pool,
// serialised on one thread, or immediate. A null queue is the process queue.
// An async block's system area holds its status and result size, so both
// stay readable after our state for the call is gone.
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "hle/xgr/xgr.h"
#include "log.h"

namespace kl::xgr {
namespace {

const GUID PROVIDER = {0x073b7dcb, 0x1fcf, 0x4030, {0x94, 0xbe, 0xe3, 0xc9, 0xeb, 0x62, 0x34, 0x28}};
constexpr uint32_t WORK = 0, COMPLETION = 1, INF = 0xffffffffu;
enum class Mode { Manual, ThreadPool, Serialized, Immediate };

using Callback = void(WINAPI*)(void*, bool);
using MonitorCallback = void(WINAPI*)(void*, void*, uint32_t);
using TerminatedCallback = void(WINAPI*)(void*);

struct Task {
    void* context;
    Callback callback;
    void run(bool canceled) const { callback(context, canceled); }
};

// The pool behind thread-pool ports: a new worker whenever more tasks wait than workers idle,
// so a callback blocking on another never starves it.
struct Pool {
    std::mutex m;
    std::condition_variable ready;
    std::deque<Task> tasks;
    size_t idle = 0;
    void submit(Task t) {
        std::unique_lock g(m);
        tasks.push_back(t);
        if (tasks.size() > idle) {
            std::thread([this] { work(); }).detach();
        } else {
            ready.notify_one();
        }
    }
    void work() {
        SetThreadDescription(GetCurrentThread(), L"keylime-xtaskqueue");
        std::unique_lock g(m);
        for (;;) {
            while (tasks.empty()) {
                ++idle;
                ready.wait(g);
                --idle;
            }
            Task t = tasks.front();
            tasks.pop_front();
            g.unlock();
            t.run(false);
            g.lock();
        }
    }
};
Pool g_pool;

struct Port {
    Mode mode;
    std::mutex m;
    std::condition_variable ready;
    std::deque<Task> tasks;
    explicit Port(Mode md) : mode(md) {
        if (mode == Mode::Serialized)
            std::thread([this] {
                SetThreadDescription(GetCurrentThread(), L"keylime-xtaskqueue-serial");
                for (;;) {
                    Task t;
                    if (take(INF, [] { return false; }, t)) t.run(false);
                }
            }).detach();
    }
    void submit(Task t) {
        switch (mode) {
            case Mode::Immediate: t.run(false); break;
            case Mode::ThreadPool: g_pool.submit(t); break;
            default: {
                std::lock_guard g(m);
                tasks.push_back(t);
                ready.notify_one();
            }
        }
    }
    template <class F>
    bool take(uint32_t timeout_ms, F ended, Task& out) {
        std::unique_lock g(m);
        auto waiting = [&] { return !tasks.empty() || ended(); };
        if (timeout_ms == INF) ready.wait(g, waiting);
        else ready.wait_for(g, std::chrono::milliseconds(timeout_ms), waiting);
        if (tasks.empty()) return false;
        out = tasks.front();
        tasks.pop_front();
        return true;
    }
};

// Queues live for the process (handles are their addresses; closing is free).
struct Queue {
    Port* ports[2];
    std::mutex m;
    std::vector<std::tuple<uint64_t, void*, MonitorCallback>> monitors;
    std::atomic<bool> terminated{false};

    void deliver(uint32_t port, Task t) {
        ports[port]->submit(t);
        decltype(monitors) copy;
        {
            std::lock_guard g(m);
            copy = monitors;
        }
        for (auto& [id, ctx, cb] : copy) cb(ctx, this, port);
    }
    HRESULT submit(uint32_t port, uint32_t delay_ms, Task t) {
        if (port > COMPLETION) return E_INVALIDARG_;
        if (terminated) return E_ABORT_;
        if (!delay_ms) {
            deliver(port, t);
        } else {
            std::thread([this, port, delay_ms, t] {
                Sleep(delay_ms);
                deliver(port, t);
            }).detach();
        }
        return S_OK;
    }
};

std::mutex g_process_lock;
bool g_process_set = false;
Queue* g_process = nullptr;

Mode mode_of(uint32_t m, bool& ok) {
    ok = m <= 3;
    return Mode(m);
}

Queue* process_queue() {
    std::lock_guard g(g_process_lock);
    if (!g_process_set) {
        g_process_set = true;
        g_process = new Queue{{new Port(Mode::ThreadPool), new Port(Mode::ThreadPool)}};
    }
    return g_process;
}
Queue* queue_of(void* h) { return h ? static_cast<Queue*>(h) : process_queue(); }

// ------------------------------------------------------------- async ----
struct Call {
    Provider provider;
    void* context;
    const void* identity;
    std::mutex m;
    std::condition_variable finished;
    bool done = false;
    HRESULT op(uint32_t o, AsyncBlock* b, size_t size, void* buf) {
        ProviderData d{b, size, buf, context};
        return provider(o, &d);
    }
};
std::mutex g_calls_lock;
std::unordered_map<AsyncBlock*, std::shared_ptr<Call>> g_calls;

std::shared_ptr<Call> call_of(AsyncBlock* b) {
    std::lock_guard g(g_calls_lock);
    auto it = g_calls.find(b);
    return it == g_calls.end() ? nullptr : it->second;
}
std::atomic<uintptr_t>& status_word(AsyncBlock* b) { return *reinterpret_cast<std::atomic<uintptr_t>*>(&b->internal[1]); }
std::atomic<uintptr_t>& size_word(AsyncBlock* b) { return *reinterpret_cast<std::atomic<uintptr_t>*>(&b->internal[2]); }
HRESULT status_of(AsyncBlock* b) { return HRESULT(uint32_t(status_word(b).load())); }

void release(AsyncBlock* b) {
    std::shared_ptr<Call> c;
    {
        std::lock_guard g(g_calls_lock);
        auto it = g_calls.find(b);
        if (it == g_calls.end()) return;
        c = it->second;
        g_calls.erase(it);
    }
    c->op(OP_CLEANUP, b, 0, nullptr);
}

void WINAPI run_completion(void* block, bool) {
    auto* b = static_cast<AsyncBlock*>(block);
    if (b->callback) b->callback(b);
}

}  // namespace

HRESULT async_begin(AsyncBlock* b, void* context, const void* identity, Provider p) {
    if (!p || !b || call_of(b)) return E_INVALIDARG_;
    status_word(b) = uintptr_t(uint32_t(E_PENDING_));
    size_word(b) = 0;
    auto c = std::make_shared<Call>();
    c->provider = p, c->context = context, c->identity = identity;
    {
        std::lock_guard g(g_calls_lock);
        g_calls[b] = c;
    }
    HRESULT hr = c->op(OP_BEGIN, b, 0, nullptr);
    if (FAILED(hr)) async_complete(b, hr, 0);
    return S_OK;
}

HRESULT async_schedule(AsyncBlock* b, uint32_t delay_ms) {
    if (!b || !call_of(b)) return E_INVALIDARG_;
    auto work = [](void* block, bool canceled) {
        auto* bb = static_cast<AsyncBlock*>(block);
        auto c = call_of(bb);
        if (!c) return;
        HRESULT hr = canceled ? E_ABORT_ : c->op(OP_DO_WORK, bb, 0, nullptr);
        if (hr != E_PENDING_) async_complete(bb, hr, 0);
    };
    return queue_of(b->queue)->submit(WORK, delay_ms, Task{b, static_cast<Callback>(work)});
}

void async_complete(AsyncBlock* b, HRESULT result, size_t required) {
    auto c = call_of(b);
    if (!c || result == E_PENDING_) return;
    std::unique_lock g(c->m);
    if (c->done) return;
    void* q = b->queue;
    auto routine = b->callback;
    if (FAILED(result) || required == 0) release(b);
    size_word(b) = required;
    status_word(b) = uintptr_t(uint32_t(result));
    c->done = true;
    c->finished.notify_all();
    g.unlock();
    if (routine) {
        Task t{b, &run_completion};
        if (queue_of(q)->submit(COMPLETION, 0, t) != S_OK) t.run(true);
    }
}

HRESULT async_status(AsyncBlock* b, bool wait) {
    if (!b) return E_INVALIDARG_;
    if (wait)
        if (auto c = call_of(b)) {
            std::unique_lock g(c->m);
            c->finished.wait(g, [&] { return c->done; });
        }
    return status_of(b);
}

HRESULT async_result_size(AsyncBlock* b, size_t* out) {
    if (!b || !out) return E_INVALIDARG_;
    HRESULT hr = status_of(b);
    if (SUCCEEDED(hr)) *out = size_word(b);
    return hr;
}

HRESULT async_result(AsyncBlock* b, const void* identity, size_t size, void* buffer, size_t* used) {
    if (!b) return E_INVALIDARG_;
    HRESULT hr = status_of(b);
    size_t need = size_word(b);
    if (FAILED(hr) || need == 0) {
        if (SUCCEEDED(hr) && used) *used = 0;
        return hr;
    }
    auto c = call_of(b);
    if (!c) return E_ILLEGAL_METHOD_CALL_;
    if (c->identity != identity) return E_INVALIDARG_;
    if (size < need || !buffer) return E_NOT_SUFFICIENT_BUFFER_;
    hr = c->op(OP_GET_RESULT, b, size, buffer);
    if (used) *used = need;
    release(b);
    return hr;
}

namespace {
struct Finished {
    HRESULT hr;
    std::vector<uint8_t> bytes;
};
HRESULT WINAPI finished_provider(uint32_t op, const ProviderData* d) {
    auto* f = static_cast<Finished*>(d->context);
    switch (op) {
        case OP_CLEANUP: delete f; break;
        case OP_BEGIN: {
            HRESULT hr = f->hr;
            size_t n = SUCCEEDED(hr) ? f->bytes.size() : 0;
            async_complete(d->block, hr, n);
            break;
        }
        case OP_GET_RESULT:
            if (!f->bytes.empty()) memcpy(d->buffer, f->bytes.data(), f->bytes.size());
            break;
    }
    return S_OK;
}
struct Pending {
    std::function<HRESULT(std::vector<uint8_t>&)> work;
    std::vector<uint8_t> bytes;
};
HRESULT WINAPI pending_provider(uint32_t op, const ProviderData* d) {
    auto* p = static_cast<Pending*>(d->context);
    switch (op) {
        case OP_CLEANUP: delete p; break;
        case OP_BEGIN: {
            AsyncBlock* b = d->block;
            std::thread([p, b] {
                HRESULT hr = p->work ? p->work(p->bytes) : E_ABORT_;
                async_complete(b, hr, SUCCEEDED(hr) ? p->bytes.size() : 0);
            }).detach();
            break;
        }
        case OP_GET_RESULT:
            if (!p->bytes.empty()) memcpy(d->buffer, p->bytes.data(), p->bytes.size());
            break;
    }
    return S_OK;
}
}  // namespace

HRESULT finished(AsyncBlock* b, const void* identity, HRESULT hr, std::vector<uint8_t> bytes) {
    auto* f = new Finished{hr, std::move(bytes)};
    HRESULT r = async_begin(b, f, identity, &finished_provider);
    if (FAILED(r)) delete f;
    return r;
}

HRESULT pending(AsyncBlock* b, const void* identity, std::function<HRESULT(std::vector<uint8_t>&)> work) {
    auto* p = new Pending{std::move(work), {}};
    HRESULT r = async_begin(b, p, identity, &pending_provider);
    if (FAILED(r)) delete p;
    return r;
}

HRESULT queue_submit(void* q, uint32_t port, uint32_t delay_ms, void* context, void(WINAPI* callback)(void*, bool)) {
    if (!callback) return E_INVALIDARG_;
    return queue_of(q)->submit(port, delay_ms, Task{context, callback});
}

namespace {
// ------------------------------------------------------------- methods ----
using This = Object*;
using Work = HRESULT(WINAPI*)(AsyncBlock*);

HRESULT WINAPI m_get_status(This, AsyncBlock* b, uint8_t wait) { return async_status(b, wait != 0); }
HRESULT WINAPI m_get_result_size(This, AsyncBlock* b, size_t* out) { return async_result_size(b, out); }
void WINAPI m_cancel(This, AsyncBlock* b) {
    if (auto c = call_of(b)) {
        c->op(OP_CANCEL, b, 0, nullptr);
        async_complete(b, E_ABORT_, 0);
    }
}
HRESULT WINAPI run_provider(uint32_t op, const ProviderData* d) {
    if (op == OP_BEGIN) return async_schedule(d->block, 0);
    if (op == OP_DO_WORK) return reinterpret_cast<Work>(d->context)(d->block);
    return S_OK;
}
HRESULT WINAPI m_run(This, AsyncBlock* b, Work w) {
    if (!w) return E_INVALIDARG_;
    return async_begin(b, reinterpret_cast<void*>(w), reinterpret_cast<const void*>(&run_provider), &run_provider);
}
HRESULT WINAPI m_begin(This, AsyncBlock* b, void* ctx, const void* identity, const char*, Provider p) {
    return async_begin(b, ctx, identity, p);
}
HRESULT WINAPI m_schedule(This, AsyncBlock* b, uint32_t delay) { return async_schedule(b, delay); }
void WINAPI m_complete(This, AsyncBlock* b, HRESULT hr, size_t size) { async_complete(b, hr, size); }
HRESULT WINAPI m_get_result(This, AsyncBlock* b, const void* identity, size_t size, void* buf, size_t* used) {
    return async_result(b, identity, size, buf, used);
}
HRESULT WINAPI m_create(This, uint32_t work, uint32_t completion, void** out) {
    bool a, c;
    Mode w = mode_of(work, a), cm = mode_of(completion, c);
    if (!a || !c || !out) return E_INVALIDARG_;
    *out = new Queue{{new Port(w), new Port(cm)}};
    return S_OK;
}
HRESULT WINAPI m_create_composite(This, void* work, void* completion, void** out) {
    if (!work || !completion || !out) return E_INVALIDARG_;
    *out = new Queue{{static_cast<Port*>(work), static_cast<Port*>(completion)}};
    return S_OK;
}
HRESULT WINAPI m_get_port(This, void* q, uint32_t port, void** out) {
    if (port > COMPLETION || !out) return E_INVALIDARG_;
    *out = queue_of(q)->ports[port];
    return S_OK;
}
HRESULT WINAPI m_duplicate(This, void* q, void** out) {
    if (!out) return E_INVALIDARG_;
    *out = queue_of(q);
    return S_OK;
}
bool WINAPI m_dispatch(This, void* q, uint32_t port, uint32_t timeout) {
    Queue* qu = queue_of(q);
    if (port > COMPLETION || qu->ports[port]->mode != Mode::Manual) return false;
    Task t;
    if (!qu->ports[port]->take(timeout, [&] { return qu->terminated.load(); }, t)) return false;
    t.run(false);
    return true;
}
void WINAPI m_close(This, void*) {}
HRESULT WINAPI m_submit(This, void* q, uint32_t port, void* ctx, Callback cb) { return queue_submit(q, port, 0, ctx, cb); }
HRESULT WINAPI m_submit_delayed(This, void* q, uint32_t port, uint32_t delay, void* ctx, Callback cb) {
    return queue_submit(q, port, delay, ctx, cb);
}
HRESULT WINAPI m_terminate(This, void* q, uint8_t, void* ctx, TerminatedCallback cb) {
    Queue* qu = queue_of(q);
    qu->terminated = true;
    for (Port* p : qu->ports) {
        std::deque<Task> pending;
        {
            std::lock_guard g(p->m);
            pending.swap(p->tasks);
        }
        p->ready.notify_all();
        for (auto& t : pending) t.run(true);
    }
    if (cb) cb(ctx);
    return S_OK;
}
std::atomic<uint64_t> g_next_monitor{1};
HRESULT WINAPI m_register_monitor(This, void* q, void* ctx, MonitorCallback cb, uint64_t* token_out) {
    if (!cb || !token_out) return E_INVALIDARG_;
    Queue* qu = queue_of(q);
    uint64_t id = g_next_monitor++;
    std::lock_guard g(qu->m);
    qu->monitors.emplace_back(id, ctx, cb);
    *token_out = id;
    return S_OK;
}
void WINAPI m_unregister_monitor(This, void* q, uint64_t id) {
    Queue* qu = queue_of(q);
    std::lock_guard g(qu->m);
    std::erase_if(qu->monitors, [&](auto& m) { return std::get<0>(m) == id; });
}
bool WINAPI m_get_process_queue(This, void** out) {
    Queue* q = process_queue();
    if (out) *out = q;
    return q != nullptr;
}
void WINAPI m_set_process_queue(This, void* q) {
    std::lock_guard g(g_process_lock);
    g_process_set = true;
    g_process = static_cast<Queue*>(q);
}
void WINAPI m_check_thread(This) {}

struct Init {
    Init() {
        void** t = make_table("task queue", 31);
        auto s = [&](size_t off, auto fn) { set(t, off, reinterpret_cast<void*>(fn)); };
        s(0x18, &m_get_status), s(0x20, &m_get_result_size), s(0x28, &m_cancel), s(0x30, &m_run), s(0x38, &m_begin);
        s(0x48, &m_schedule), s(0x50, &m_complete), s(0x58, &m_get_result), s(0x60, &m_create), s(0x68, &m_create_composite);
        s(0x70, &m_get_port), s(0x78, &m_duplicate), s(0x80, &m_dispatch), s(0x88, &m_close), s(0x90, &m_submit);
        s(0x98, &m_submit_delayed), s(0xb0, &m_terminate), s(0xb8, &m_register_monitor), s(0xc0, &m_unregister_monitor);
        s(0xc8, &m_get_process_queue), s(0xd0, &m_set_process_queue), s(0xe8, &m_check_thread);
        static Object o{t, "task queue"};
        add_interfaces({{PROVIDER, PROVIDER, &o}});
    }
} g_init;

}  // namespace
}  // namespace kl::xgr
