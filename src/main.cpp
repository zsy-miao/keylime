// keylime: runs an Xbox Series title on Windows.
//   keylime run <title.exe> [--execute] [--console series-x|series-s]
//   keylime imports <title.exe>        list what the title imports and how we would bind it
#include <windows.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "hle/files.h"
#include "hle/hle.h"
#include "loader.h"
#include "log.h"
#include "stats.h"
#include "runtime.h"

namespace fs = std::filesystem;

namespace kl {

namespace cpu { void install(); }
namespace shader { int tool(const std::filesystem::path&, bool); }
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

static LONG CALLBACK on_exception(EXCEPTION_POINTERS* ep) {
    auto* er = ep->ExceptionRecord;
    DWORD code = er->ExceptionCode;
    // Only faults worth a report; C++ throws, debug prints, etc. pass silently.
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_STACK_OVERFLOW && code != EXCEPTION_PRIV_INSTRUCTION &&
        code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != STATUS_HEAP_CORRUPTION && code != 0xC0000409)
        return EXCEPTION_CONTINUE_SEARCH;
    static LONG reports = 0;
    if (InterlockedIncrement(&reports) > 8) return EXCEPTION_CONTINUE_SEARCH;
    auto* c = ep->ContextRecord;
    std::string what;
    if (code == EXCEPTION_ACCESS_VIOLATION && er->NumberParameters >= 2) {
        const char* kind = er->ExceptionInformation[0] == 0 ? "read" : er->ExceptionInformation[0] == 1 ? "write" : "execute";
        char b[96];
        snprintf(b, sizeof b, " (%s at 0x%llx)", kind, (unsigned long long)er->ExceptionInformation[1]);
        what = b;
    }
    KL_ERROR("crash", "exception 0x%08lx%s at %s", code, what.c_str(), describe(c->Rip).c_str());
    KL_ERROR("crash", "rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx rsp=%llx", c->Rax, c->Rbx, c->Rcx,
             c->Rdx, c->Rsi, c->Rdi, c->Rsp);
    KL_ERROR("crash", "r8=%llx r9=%llx r10=%llx r11=%llx r12=%llx r13=%llx r14=%llx r15=%llx", c->R8, c->R9,
             c->R10, c->R11, c->R12, c->R13, c->R14, c->R15);
    // Return addresses on the stack that fall inside mapped images.
    auto* sp = reinterpret_cast<const uintptr_t*>(c->Rsp);
    MEMORY_BASIC_INFORMATION mbi{};
    int shown = 0;
    if (VirtualQuery(sp, &mbi, sizeof mbi)) {
        auto* end = reinterpret_cast<const uintptr_t*>(static_cast<const char*>(mbi.BaseAddress) + mbi.RegionSize);
        for (int i = 0; i < 2048 && sp + i < end && shown < 12; ++i)
            if (loader().find(sp[i]) || (sp[i] >= uintptr_t(GetModuleHandleW(nullptr)) &&
                                         sp[i] < uintptr_t(GetModuleHandleW(nullptr)) + (32u << 20))) {
                KL_ERROR("crash", "  stack[%d] %s", i, describe(sp[i]).c_str());
                ++shown;
            }
    }
    return EXCEPTION_CONTINUE_SEARCH;
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
    cpu::install();  // first: AMD-only instructions are emulated before any report
    stats::start();
    uint32_t code = l.run_main();
    KL_INFO("main", "title exited with %u", code);
    return int(code);
}

}  // namespace kl

int wmain(int argc, wchar_t** argv) {
    using namespace kl;
    SetConsoleOutputCP(CP_UTF8);
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
