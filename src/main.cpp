// keylime: runs an Xbox Series title on Windows.
//   keylime run <title.exe> [--execute] [--console series-x|series-s]
//   keylime imports <title.exe>        list what the title imports and how we would bind it
#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>

#include <algorithm>
#include <atomic>

#include <cstdio>
#include <fstream>
#include <thread>
#include <sstream>
#include <string>
#include <vector>

#include "gpu/exec.h"
#include "gpu/fence.h"
#include "hle/files.h"
#include "hle/hle.h"
#include "loader.h"
#include "log.h"
#include "stats.h"
#include "runtime.h"

namespace fs = std::filesystem;

namespace kl {

namespace cpu { void install(); }
namespace shader { int tool(const std::filesystem::path&, bool); int validate(const std::filesystem::path&, bool driver); }
static Runtime g_runtime;
Runtime& runtime() { return g_runtime; }

static fs::path local_dir(const char* sub) {
    wchar_t buf[MAX_PATH];
    DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", buf, MAX_PATH);
    fs::path base = n ? fs::path(buf) : fs::temp_directory_path();
    auto p = base / L"keylime" / sub;
    fs::create_directories(p);
    return p;
}

// ---------------------------------------------------------------- crashes ----

static void report(EXCEPTION_POINTERS* ep, const char* how) {
    auto* er = ep->ExceptionRecord;
    DWORD code = er->ExceptionCode;
    auto* c = ep->ContextRecord;
    std::string what;
    if (code == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2) {
        const char* kind = er->ExceptionInformation[0] == 0 ? "read" : er->ExceptionInformation[0] == 1 ? "write" : "execute";
        char b[96];
        snprintf(b, sizeof b, " (%s at 0x%llx)", kind, (unsigned long long)er->ExceptionInformation[1]);
        what = b;
    }
    KL_ERROR("crash", "%s exception 0x%08lx%s at %s", how, code, what.c_str(), describe(c->Rip).c_str());
    KL_ERROR("crash", "rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx rsp=%llx", c->Rax, c->Rbx, c->Rcx,
             c->Rdx, c->Rsi, c->Rdi, c->Rsp);
    KL_ERROR("crash", "r8=%llx r9=%llx r10=%llx r11=%llx r12=%llx r13=%llx r14=%llx r15=%llx", c->R8, c->R9,
             c->R10, c->R11, c->R12, c->R13, c->R14, c->R15);
    // A call through a bad pointer: name the call site (the return address is at [rsp]).
    auto* sp = reinterpret_cast<const uintptr_t*>(c->Rsp);
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(sp, &mbi, sizeof mbi) && mbi.State == MEM_COMMIT) {
        if (code == EXCEPTION_ACCESS_VIOLATION && er->ExceptionInformation[0] == 8)
            KL_ERROR("crash", "  called from %s", describe(sp[0]).c_str());
        auto* end = reinterpret_cast<const uintptr_t*>(static_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize);
        int shown = 0;
        uintptr_t exe = uintptr_t(GetModuleHandleW(nullptr));
        for (int i = 0; i < 2048 && sp + i < end && shown < 12; ++i)
            if (loader().find(sp[i]) || (sp[i] >= exe && sp[i] < exe + (32u << 20))) {
                KL_ERROR("crash", "  stack[%d] %s", i, describe(sp[i]).c_str());
                ++shown;
            }
    }
}

// Runs the report on a fresh thread when the faulting stack may be unusable.
static void report_safely(EXCEPTION_POINTERS* ep, const char* how) {
    if (ep->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW) {
        std::thread t([&] { report(ep, how); });
        t.join();
    } else {
        report(ep, how);
    }
}

static void minidump(EXCEPTION_POINTERS* ep) {
    std::thread t([ep, tid = GetCurrentThreadId()] {
        HANDLE f = CreateFileW(L"logs/crash.dmp", GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, 0, nullptr);
        if (f == INVALID_HANDLE_VALUE) return;
        MINIDUMP_EXCEPTION_INFORMATION mei{tid, ep, FALSE};
        MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(), f,
                          MINIDUMP_TYPE(MiniDumpWithIndirectlyReferencedMemory | MiniDumpWithThreadInfo), &mei,
                          nullptr, nullptr);
        CloseHandle(f);
        KL_ERROR("crash", "minidump written to logs/crash.dmp");
    });
    t.join();
}

// First-chance faults: many are handled (our probes, the title's own SEH), so each distinct
// faulting address is reported once and the search goes on. The real crash is reported again,
// with a minidump, by the unhandled filter.
static LONG CALLBACK on_exception(EXCEPTION_POINTERS* ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_STACK_OVERFLOW && code != EXCEPTION_PRIV_INSTRUCTION &&
        code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != STATUS_HEAP_CORRUPTION && code != 0xC0000409)
        return EXCEPTION_CONTINUE_SEARCH;
    static std::atomic<uintptr_t> seen[64];
    static std::atomic<int> count{0};
    uintptr_t rip = ep->ContextRecord->Rip;
    for (int i = 0, n = std::min(count.load(), 64); i < n; ++i)
        if (seen[i] == rip) return EXCEPTION_CONTINUE_SEARCH;
    int at = count++;
    if (at >= 64) return EXCEPTION_CONTINUE_SEARCH;
    seen[at] = rip;
    report_safely(ep, "first-chance");
    return EXCEPTION_CONTINUE_SEARCH;
}

static LPTOP_LEVEL_EXCEPTION_FILTER g_title_filter = nullptr;
static LONG WINAPI on_unhandled(EXCEPTION_POINTERS* ep) {
    static std::atomic<bool> once{false};
    if (!once.exchange(true)) {
        report_safely(ep, "UNHANDLED");
        minidump(ep);
    }
    return g_title_filter ? g_title_filter(ep) : EXCEPTION_CONTINUE_SEARCH;
}
// The title's SetUnhandledExceptionFilter: kept behind ours.
LPTOP_LEVEL_EXCEPTION_FILTER title_unhandled_filter(LPTOP_LEVEL_EXCEPTION_FILTER f) {
    auto old = g_title_filter;
    g_title_filter = f;
    return old;
}

// KEYLIME_STACKS_AT=<seconds>: logs where every thread is at that moment (title return addresses
// on each stack), for finding what a stalled title waits on. Addresses are gathered while the
// thread is suspended (no allocation), described after it resumes.
static void dump_stacks(const char* why) {
    {
        std::vector<std::string> lines;
        lines.push_back(std::string("thread stacks: ") + why);
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        THREADENTRY32 te{sizeof te};
        DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
        uintptr_t exe = uintptr_t(GetModuleHandleW(nullptr));
        for (BOOL ok = Thread32First(snap, &te); ok; ok = Thread32Next(snap, &te)) {
            if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
            HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
            if (!h) continue;
            uintptr_t found[12] = {};
            int n = 0;
            CONTEXT c{};
            c.ContextFlags = CONTEXT_FULL;
            if (SuspendThread(h) != DWORD(-1)) {
                if (GetThreadContext(h, &c)) {
                    found[n++] = c.Rip;
                    auto* sp = reinterpret_cast<const uintptr_t*>(c.Rsp);
                    MEMORY_BASIC_INFORMATION mbi{};
                    if (VirtualQuery(sp, &mbi, sizeof mbi)) {
                        auto* end = reinterpret_cast<const uintptr_t*>(static_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize);
                        for (int i = 0; i < 4096 && sp + i < end && n < 12; ++i) {
                            DWORD64 base = 0;
                            // Code only: return addresses inside functions with unwind data, in title images or ours.
                            if ((loader().find(sp[i]) || (sp[i] >= exe && sp[i] < exe + (32u << 20))) &&
                                RtlLookupFunctionEntry(sp[i], &base, nullptr))
                                found[n++] = sp[i];
                        }
                    }
                }
                ResumeThread(h);
            }
            PWSTR desc = nullptr;
            std::string name;
            if (SUCCEEDED(GetThreadDescription(h, &desc)) && desc) {
                for (PWSTR q = desc; *q; ++q) name.push_back(char(*q));
                LocalFree(desc);
            }
            CloseHandle(h);
            if (!n) continue;
            lines.push_back("thread " + std::to_string(te.th32ThreadID) + " " + name + ": at " + describe(found[0]));
            for (int i = 1; i < n; ++i) lines.push_back("    " + describe(found[i]));
        }
        CloseHandle(snap);
        // The file first: a deadlock may hold the log's lock.
        if (FILE* f = fopen("logs/stacks.txt", "a")) {
            for (auto& l : lines) fprintf(f, "%s\n", l.c_str());
            fclose(f);
        }
        for (auto& l : lines) KL_INFO("stacks", "%s", l.c_str());
    }
}

static void stacks_later() {
    char v[16]{};
    if (GetEnvironmentVariableA("KEYLIME_STACKS_AT", v, 16)) {
        double at = atof(v);
        std::thread([at] {
            Sleep(DWORD(at * 1000));
            dump_stacks("KEYLIME_STACKS_AT");
        }).detach();
    }
    // Watchdog: frames were being shown and then stop -> where is everyone, and what does the GPU wait on?
    std::thread([] {
        uint64_t last = 0;
        int still = 0, dumps = 0;
        for (;;) {
            Sleep(1000);
            uint64_t now = stats::g_counters[stats::FramesShown].load();
            if (now != last || now == 0) {
                last = now, still = 0;
                continue;
            }
            ++still;
            if ((still == 5 || still == 20) && dumps++ < 4) {
                dump_stacks(still == 5 ? "no frame shown for 5 s" : "no frame shown for 20 s");
                KL_WARN("hang", "no frame shown for %d s: what every queue is waiting on follows", still);
                kl::gpu::exec::dump_state();
                kl::gpu::fence::dump_state();
            }
        }
    }).detach();
}

// ------------------------------------------------------------------- run ----

static uint32_t title_id_from_manifest(const fs::path& dir) {
    std::ifstream f(dir / "MicrosoftGame.config");
    std::stringstream ss;
    ss << f.rdbuf();
    auto s = ss.str();
    auto at = s.find("<TitleId>");
    if (at == std::string::npos) return 0;
    return uint32_t(strtoul(s.c_str() + at + 9, nullptr, 16));
}

static int run(const fs::path& exe, bool execute, bool list_only) {
    auto& rt = runtime();
    rt.title_dir = fs::absolute(exe).parent_path();
    rt.exe_name = exe.filename();
    rt.cache_dir = local_dir("cache");
    rt.title_id = title_id_from_manifest(rt.title_dir);
    char tid[16];
    snprintf(tid, sizeof tid, "%08X", rt.title_id);
    rt.data_dir = local_dir("data") / tid;
    fs::create_directories(rt.data_dir / "temp");
    // The console's fixed drives: the package as G:, temporary files as T:.
    files::mount('G', rt.title_dir);
    files::mount('T', rt.data_dir / "temp");
    KL_INFO("main", "title %s (%ls), %zu emulated modules", tid, exe.c_str(), hle::modules().size());

    static Loader l(rt.title_dir, rt.cache_dir / "images");
    set_loader(&l);
    std::string error;
    if (!l.load_main(exe, error)) {
        KL_ERROR("main", "%s", error.c_str());
        return 1;
    }
    l.bind_all();
    if (list_only || !execute) {
        KL_INFO("main", "not executing (pass --execute to start the title)");
        return 0;
    }
    AddVectoredExceptionHandler(1, on_exception);
    SetUnhandledExceptionFilter(on_unhandled);
    cpu::install();
    stacks_later();  // first: AMD-only instructions are emulated before any report
    stats::start();
    uint32_t code = l.run_main();
    KL_INFO("main", "title exited with %u", code);
    return int(code);
}

}  // namespace kl

int wmain(int argc, wchar_t** argv) {
    using namespace kl;
    SetConsoleOutputCP(CP_UTF8);
    if (argc >= 4 && !wcscmp(argv[1], L"shader") && (!wcscmp(argv[3], L"--spirv") || !wcscmp(argv[3], L"--spirv-only"))) {
        log_open(nullptr);
        return shader::validate(argv[2], !wcscmp(argv[3], L"--spirv"));
    }
    if (argc >= 3 && !wcscmp(argv[1], L"shader")) return shader::tool(argv[2], argc > 3 && !wcscmp(argv[3], L"--dis"));
    if (argc < 3 || (wcscmp(argv[1], L"run") != 0 && wcscmp(argv[1], L"imports") != 0)) {
        fprintf(stderr, "keylime run <title.exe> [--execute] [--console series-x|series-s]\n"
                        "keylime imports <title.exe>\n");
        return 2;
    }
    bool execute = false;
    for (int i = 3; i < argc; ++i) {
        if (!wcscmp(argv[i], L"--execute")) execute = true;
        else if (!wcscmp(argv[i], L"--console") && i + 1 < argc)
            runtime().console = !wcscmp(argv[++i], L"series-s") ? Console::SeriesS : Console::SeriesX;
    }
    fs::create_directories(L"logs");
    log_open(L"logs\\keylime.log");
    KL_INFO("main", "keylime 0.1 (C++)");
    return run(argv[2], execute, !wcscmp(argv[1], L"imports"));
}
