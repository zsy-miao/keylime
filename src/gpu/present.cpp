// keylime GPU: PresentX. The presented plane's first resource (the title's
// back buffer, a Vulkan image here) is scaled into a swap chain on the
// title's window (aspect kept, MAILBOX/IMMEDIATE so presenting never stalls
// the title's own pacing). Until a frame can be shown, each present paints
// keylime's live status into the window (GDI) instead.
#include "gpu/present.h"

#include <windows.h>

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/exec.h"
#include "gpu/host.h"
#include "gpu/images.h"
#include "gpu/memory.h"
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

namespace {
// One display plane of PresentX (0x48 bytes; VIctory's layout).
struct Plane {
    uint64_t token;
    uint32_t count;
    uint32_t pad0;
    void* const* resources;
    const int32_t* src_rects;
    const int32_t* dst_rects;
    uint32_t color_space, scale_filter, extended_count, pad1;
    const void* extended;
    uint32_t flags, pad2;
};
static_assert(sizeof(Plane) == 0x48);

struct Swap {
    HWND window = nullptr;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR chain = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{};
    std::vector<VkImage> images;
    std::vector<VkImageView> views;  // for the debug panel
    std::vector<VkSemaphore> done;
    VkSemaphore acquire[8]{};
    uint32_t next = 0;
    bool broken = false;
} g_swap;
bool g_shown = false;

VkSemaphore binary() {
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore s = VK_NULL_HANDLE;
    vk::vkCreateSemaphore(host::get().device, &si, nullptr, &s);
    return s;
}

bool make_chain() {
    auto& h = host::get();
    Swap& w = g_swap;
    if (!w.window) {
        EnumWindows(find_window, LPARAM(&w.window));
        if (!w.window) return false;
    }
    if (!w.surface) {
        VkWin32SurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        si.hinstance = GetModuleHandleW(nullptr);
        si.hwnd = w.window;
        if (!KL_VK_CHECK(vk::vkCreateWin32SurfaceKHR(h.instance, &si, nullptr, &w.surface))) return w.broken = true, false;
        VkBool32 ok = VK_FALSE;
        vk::vkGetPhysicalDeviceSurfaceSupportKHR(h.physical, h.queues.graphics_family, w.surface, &ok);
        if (!ok) {
            KL_ERROR("gpu", "the graphics queue cannot present to the title's window");
            return w.broken = true, false;
        }
        for (auto& s : w.acquire) s = binary();
    }
    VkSurfaceCapabilitiesKHR caps;
    vk::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(h.physical, w.surface, &caps);
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) return false;  // minimised
    uint32_t n = 0;
    vk::vkGetPhysicalDeviceSurfaceFormatsKHR(h.physical, w.surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vk::vkGetPhysicalDeviceSurfaceFormatsKHR(h.physical, w.surface, &n, formats.data());
    VkSurfaceFormatKHR pick = formats.empty() ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR} : formats[0];
    for (auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) pick = f;
    vk::vkGetPhysicalDeviceSurfacePresentModesKHR(h.physical, w.surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vk::vkGetPhysicalDeviceSurfacePresentModesKHR(h.physical, w.surface, &n, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : modes)
        if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
    for (auto m : modes)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = w.surface;
    ci.minImageCount = std::max(caps.minImageCount, 3u);
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = pick.format;
    ci.imageColorSpace = pick.colorSpace;
    ci.imageExtent = caps.currentExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = w.chain;
    VkSwapchainKHR chain;
    if (!KL_VK_CHECK(vk::vkCreateSwapchainKHR(h.device, &ci, nullptr, &chain))) return false;
    if (w.chain) {
        vk::vkDeviceWaitIdle(h.device);
        vk::vkDestroySwapchainKHR(h.device, w.chain, nullptr);
    }
    w.chain = chain;
    w.format = pick.format;
    w.extent = caps.currentExtent;
    vk::vkGetSwapchainImagesKHR(h.device, chain, &n, nullptr);
    w.images.resize(n);
    vk::vkGetSwapchainImagesKHR(h.device, chain, &n, w.images.data());
    for (auto v : w.views) vk::vkDestroyImageView(h.device, v, nullptr);
    w.views.assign(n, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = w.images[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = pick.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk::vkCreateImageView(h.device, &vi, nullptr, &w.views[i]);
    }
    while (w.done.size() < n) w.done.push_back(binary());
    KL_INFO("gpu", "swap chain %ux%u, %u images, %s", w.extent.width, w.extent.height, n,
            mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo");
    return true;
}

void layout(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = from, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &b;
    vk::vkCmdPipelineBarrier2(cb, &di);
}

#include "gpu/overlay.inc"

// KEYLIME_SNAPSHOTS=<dir>: writes what is shown at some frames as BMP files (for checking frames without a person).
const char* snapshot_dir() {
    static char dir[MAX_PATH] = {};
    static bool on = GetEnvironmentVariableA("KEYLIME_SNAPSHOTS", dir, MAX_PATH) > 0;
    return on ? dir : nullptr;
}
bool snapshot_due(uint64_t frame) {
    static const uint64_t at[] = {10, 60, 150, 300, 450, 600, 900, 1200};
    for (uint64_t a : at)
        if (a == frame) return true;
    return frame > 1200 && frame % 60 == 0;  // the opening cutscene starts near frame 1400
}
void write_bmp(const char* path, const uint8_t* bgra, uint32_t w, uint32_t h) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    uint32_t row = w * 4, size = row * h;
    uint8_t hdr[54] = {'B', 'M'};
    auto put = [&](int at, uint32_t v) { memcpy(hdr + at, &v, 4); };
    put(2, 54 + size), put(10, 54), put(14, 40), put(18, w), put(22, uint32_t(-int32_t(h)));
    hdr[26] = 1, hdr[28] = 32;
    put(34, size);
    fwrite(hdr, 1, 54, f);
    fwrite(bgra, 1, size, f);
    fclose(f);
}

// Shows `src` in the window; false if it could not.
// The part of the source shown: the title renders below its target's size at times (dynamic
// resolution) and says which rectangle holds the picture.
int32_t g_src_rect[4] = {};
bool show(int q, images::Image* src) {
    auto& h = host::get();
    Swap& w = g_swap;
    if (w.broken) return false;
    if (!w.chain && !make_chain()) return false;
    VkSemaphore acq = w.acquire[w.next++ % 8];
    uint32_t idx = 0;
    VkResult r = vk::vkAcquireNextImageKHR(h.device, w.chain, 100'000'000ull, acq, VK_NULL_HANDLE, &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        make_chain();
        return false;
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) return false;
    auto f = images::facts(src);
    VkImage dst = w.images[idx];
    int32_t sx0 = 0, sy0 = 0, sx1 = int32_t(f.width), sy1 = int32_t(f.height);
    if (g_src_rect[2] > g_src_rect[0] && g_src_rect[3] > g_src_rect[1] && g_src_rect[0] >= 0 && g_src_rect[1] >= 0 &&
        g_src_rect[2] <= int32_t(f.width) && g_src_rect[3] <= int32_t(f.height))
        sx0 = g_src_rect[0], sy0 = g_src_rect[1], sx1 = g_src_rect[2], sy1 = g_src_rect[3];
    // Fit the frame to the window, keeping its aspect.
    double sx = double(w.extent.width) / (sx1 - sx0), sy = double(w.extent.height) / (sy1 - sy0), s = std::min(sx, sy);
    int32_t dw = int32_t((sx1 - sx0) * s), dh = int32_t((sy1 - sy0) * s);
    int32_t ox = (int32_t(w.extent.width) - dw) / 2, oy = (int32_t(w.extent.height) - dh) / 2;
    static uint64_t frame = 0;
    ++frame;
    memory::Block grab;
    if (snapshot_dir() && snapshot_due(frame)) memory::make_block(uint64_t(w.extent.width) * w.extent.height * 4, false, grab);
    overlay::frame();
    bool ok = exec::run(q, [&](VkCommandBuffer cb) {
        layout(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue black{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk::vkCmdClearColorImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        layout(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageBlit b{};
        b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[0] = {sx0, sy0, 0};
        b.srcOffsets[1] = {sx1, sy1, 1};
        b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.dstOffsets[0] = {ox, oy, 0};
        b.dstOffsets[1] = {ox + dw, oy + dh, 1};
        vk::vkCmdBlitImage(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
        overlay::draw(cb, dst, w.views[idx], w.extent);
        if (grab.buffer) {
            layout(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy c{};
            c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent = {w.extent.width, w.extent.height, 1};
            vk::vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, grab.buffer, 1, &c);
            layout(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        } else {
            layout(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }
    }, acq, w.done[idx]);
    if (!ok) return false;
    if (grab.buffer) {
        exec::with_queue(q, [](VkQueue vq) { vk::vkQueueWaitIdle(vq); });
        char path[MAX_PATH + 64];
        snprintf(path, sizeof path, "%s/frame-%05llu.bmp", snapshot_dir(), (unsigned long long)frame);
        if (w.format == VK_FORMAT_B8G8R8A8_UNORM || w.format == VK_FORMAT_B8G8R8A8_SRGB) write_bmp(path, grab.map, w.extent.width, w.extent.height);
        KL_INFO("gpu", "snapshot of frame %llu: %s (source %ux%u format %d)", (unsigned long long)frame, path, f.width, f.height, int(f.format));
        memory::free_block(grab);
    }
    VkResult pr = VK_SUCCESS;
    exec::with_queue(q, [&](VkQueue vq) {
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &w.done[idx];
        pi.swapchainCount = 1;
        pi.pSwapchains = &w.chain;
        pi.pImageIndices = &idx;
        pr = vk::vkQueuePresentKHR(vq, &pi);
    });
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) make_chain();
    return true;
}
}  // namespace

HRESULT present(int q, uint32_t planes, const void* plane_list, const void*) {
    stats::add(stats::Presents);
    images::frame_done();
    static uint32_t seen_planes = ~0u;
    if (planes != seen_planes && plane_list) {  // how many planes, and what each shows (the world may not be the first)
        seen_planes = planes;
        auto* pl = static_cast<const Plane*>(plane_list);
        for (uint32_t i = 0; i < planes && i < 4; ++i) {
            images::Image* im = pl[i].count && pl[i].resources ? images::of_resource(pl[i].resources[0]) : nullptr;
            auto fa = im ? images::facts(im) : images::Facts{};
            int32_t r[4] = {};
            if (pl[i].src_rects) memcpy(r, pl[i].src_rects, sizeof r);
            KL_INFO("gpu", "PresentX: plane %u of %u: %ux%u format %d, flags %x, color space %u, src rect %d,%d-%d,%d", i, planes, fa.width,
                    fa.height, int(fa.format), pl[i].flags, pl[i].color_space, r[0], r[1], r[2], r[3]);
        }
    }
    images::Image* src = nullptr;
    if (planes && plane_list) {
        auto* p = static_cast<const Plane*>(plane_list);
        if (p->count && p->resources && p->resources[0]) src = images::of_resource(p->resources[0]);
        static int32_t last[4] = {};
        if (p->src_rects) memcpy(g_src_rect, p->src_rects, sizeof g_src_rect);
        else memset(g_src_rect, 0, sizeof g_src_rect);
        if (memcmp(last, g_src_rect, sizeof last)) {
            static int told = 0;
            if (told++ < 40) KL_INFO("gpu", "PresentX: source rectangle %d,%d-%d,%d", g_src_rect[0], g_src_rect[1], g_src_rect[2], g_src_rect[3]);
            memcpy(last, g_src_rect, sizeof last);
        }
    }
    if (src && show(q, src)) {
        if (!g_shown) g_shown = true, KL_INFO("gpu", "first frame shown: the title's own image is on screen now");
        stats::add(stats::FramesShown);
        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        stats::raise(stats::FrameWorstUs, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now - last).count()));
        last = now;
        return S_OK;
    }
    if (g_shown) return S_OK;
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
