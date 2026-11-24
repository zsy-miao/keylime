#include "log.h"

#include <windows.h>

#include <cstdio>
#include <mutex>

namespace kl {
namespace {
std::mutex g_lock;
FILE* g_file = nullptr;
LARGE_INTEGER g_start{}, g_freq{};
bool g_trace = false;
}  // namespace

void log_open(const wchar_t* path) {
    QueryPerformanceFrequency(&g_freq);
    QueryPerformanceCounter(&g_start);
    char v[8]{};
    g_trace = GetEnvironmentVariableA("KEYLIME_TRACE", v, sizeof v) > 0 && v[0] != '0';
    if (path) _wfopen_s(&g_file, path, L"w");
}

bool trace_enabled() { return g_trace; }

void logf(Level level, const char* area, const char* fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    double t = double(now.QuadPart - g_start.QuadPart) / double(g_freq.QuadPart ? g_freq.QuadPart : 1);
    static const char* names[] = {"TRACE", " INFO", " WARN", "ERROR"};
    std::lock_guard lock(g_lock);
    char line[2300];
    int n = snprintf(line, sizeof line, "%9.3f %s [%5lu] %-8s %s\n", t, names[int(level)],
                     GetCurrentThreadId(), area, msg);
    if (n > 0) {
        fwrite(line, 1, size_t(n), stdout);
        if (g_file) {
            fwrite(line, 1, size_t(n), g_file);
            if (level >= Level::Warn) fflush(g_file);
        }
    }
}

}  // namespace kl
