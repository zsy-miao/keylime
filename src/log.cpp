#include "log.h"

#include <windows.h>

#include <cstdio>
#include <cstring>
#include <mutex>

namespace kl {
namespace {
std::recursive_mutex g_lock;  // recursive: a fault inside logf must still be reportable
FILE* g_file = nullptr;
LARGE_INTEGER g_start{}, g_freq{};
bool g_trace = false;
bool g_color = false;
}  // namespace

void log_open(const wchar_t* path) {
    QueryPerformanceFrequency(&g_freq);
    QueryPerformanceCounter(&g_start);
    char v[8]{};
    g_trace = GetEnvironmentVariableA("KEYLIME_TRACE", v, sizeof v) > 0 && v[0] != '0';
    if (path) _wfopen_s(&g_file, path, L"w");
    // Colors in a real console (ANSI escape codes).
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (GetConsoleMode(out, &mode) && SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) g_color = true;
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
        if (g_color) {
            static const char* colors[] = {"[90m", "[0m", "[33m", "[91m"};
            const char* c = strcmp(area, "status") == 0 ? "[96m" : strcmp(area, "title") == 0 ? "[92m" : colors[int(level)];
            fputs(c, stdout);
            fwrite(line, 1, size_t(n), stdout);
            fputs("[0m", stdout);
        } else {
            fwrite(line, 1, size_t(n), stdout);
        }
        fflush(stdout);
        if (g_file) {
            fwrite(line, 1, size_t(n), g_file);
            fflush(g_file);
        }
    }
}

}  // namespace kl
