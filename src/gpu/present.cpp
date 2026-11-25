// keylime GPU: PresentX. Until the shader translator draws the title's
// frames, each present paints keylime's live status into the title's window
// (GDI), so the window shows the game is running instead of staying white.
#include "gpu/present.h"

#include <windows.h>

#include <chrono>
#include <string>

#include "log.h"
#include "stats.h"

namespace kl::gpu::present {
namespace {

HWND g_window = nullptr;

BOOL CALLBACK find_window(HWND h, LPARAM out) {
    DWORD pid = 0;
    GetWindowThreadProcessId(h, &pid);
    if (pid == GetCurrentProcessId() && IsWindowVisible(h) && h != GetConsoleWindow()) {
        *reinterpret_cast<HWND*>(out) = h;
        return FALSE;
    }
    return TRUE;
}

void paint() {
    if (!g_window || !IsWindow(g_window)) {
        g_window = nullptr;
        EnumWindows(find_window, LPARAM(&g_window));
        if (!g_window) return;
        // (no SetWindowText: it waits on the window's thread, which may be waiting on us)
    }
    RECT rc;
    GetClientRect(g_window, &rc);
    HDC dc = GetDC(g_window);
    if (!dc) return;
    HDC mem = CreateCompatibleDC(dc);
    HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom);
    HGDIOBJ old = SelectObject(mem, bmp);
    HBRUSH bg = CreateSolidBrush(RGB(12, 14, 18));
    FillRect(mem, &rc, bg);
    DeleteObject(bg);
    SetBkMode(mem, TRANSPARENT);
    int h = rc.bottom / 22 > 16 ? rc.bottom / 22 : 16;
    HFONT big = CreateFontW(h * 2, 0, 0, 0, FW_BOLD, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Segoe UI");
    HFONT small = CreateFontW(h, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0, CLEARTYPE_QUALITY, 0, L"Consolas");
    auto line = [&](HFONT f, COLORREF c, int y, const std::wstring& t) {
        SelectObject(mem, f);
        SetTextColor(mem, c);
        TextOutW(mem, h, y, t.c_str(), int(t.size()));
    };
    using namespace stats;
    auto v = [](Counter c) { return g_counters[c].load(); };
    static uint64_t last_frames = 0;
    static auto last_t = std::chrono::steady_clock::now();
    static double fps = 0;
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_t).count();
    if (dt >= 0.5) {
        fps = double(v(Presents) - last_frames) / dt;
        last_frames = v(Presents), last_t = now;
    }
    wchar_t b[256];
    int y = h;
    line(big, RGB(140, 220, 90), y, L"keylime");
    y += h * 3;
    line(small, RGB(220, 220, 220), y, L"GTA VI is running. Its frames are not drawn yet: the shader translator comes next.");
    y += h * 2;
    swprintf_s(b, L"frames presented   %llu   (%.0f fps)", (unsigned long long)v(Presents), fps);
    line(small, RGB(120, 200, 255), y, b), y += h + h / 2;
    swprintf_s(b, L"GPU work recorded  %llu submits, %llu draws, %llu dispatches", (unsigned long long)v(Submits),
               (unsigned long long)v(Draws), (unsigned long long)v(Dispatches));
    line(small, RGB(120, 200, 255), y, b), y += h + h / 2;
    swprintf_s(b, L"pipelines          %llu", (unsigned long long)v(Pipelines));
    line(small, RGB(120, 200, 255), y, b), y += h + h / 2;
    swprintf_s(b, L"streamed           %.0f MB in %llu requests (%llu failed)", double(v(StorageBytes)) / (1 << 20),
               (unsigned long long)v(StorageRequests), (unsigned long long)v(StorageFailures));
    line(small, RGB(120, 200, 255), y, b), y += h + h / 2;
    swprintf_s(b, L"audio              %llu XMA voices playing, %llu frames decoded", (unsigned long long)v(XmaVoices),
               (unsigned long long)v(XmaFrames));
    line(small, RGB(120, 200, 255), y, b), y += h + h / 2;
    swprintf_s(b, L"input              %llu readings", (unsigned long long)v(InputReadings));
    line(small, RGB(120, 200, 255), y, b), y += h * 2;
    line(small, RGB(140, 140, 140), y, L"Keyboard: Enter=A  Backspace=B  Esc=Menu  Tab=View  arrows=D-pad  WASD/IJKL=sticks");
    BitBlt(dc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
    SelectObject(mem, old);
    DeleteObject(bmp);
    DeleteObject(big);
    DeleteObject(small);
    DeleteDC(mem);
    ReleaseDC(g_window, dc);
}

}  // namespace

HRESULT present(VkQueue, std::mutex*, uint32_t planes, const void*, const void*) {
    stats::add(stats::Presents);
    static int n = 0;
    if (n++ < 1) KL_INFO("gpu", "PresentX: %u planes (status screen until frames are translated)", planes);
    // Painting every frame is cheap at 30 fps; skip if behind.
    static auto last = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    auto now = std::chrono::steady_clock::now();
    if (now - last >= std::chrono::milliseconds(100)) {
        last = now;
        paint();
    }
    return S_OK;
}

}  // namespace kl::gpu::present
