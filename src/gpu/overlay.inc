// keylime debug panel: small text and a frame-time graph in the top-left corner, drawn with
// clear rectangles straight into the swap chain image (no pipelines, no background; a dark
// shadow keeps it readable). F3 hides/shows it; KEYLIME_OVERLAY=0 starts it hidden.
namespace overlay {

// 5x7 glyphs, rows top to bottom, 5 bits each (bit 4 = left column).
struct Glyph { char c; uint8_t rows[7]; };
const Glyph kFont[] = {
    {'0', {14, 17, 19, 21, 25, 17, 14}}, {'1', {4, 12, 4, 4, 4, 4, 14}},  {'2', {14, 17, 1, 2, 4, 8, 31}},
    {'3', {30, 1, 1, 14, 1, 1, 30}},     {'4', {2, 6, 10, 18, 31, 2, 2}}, {'5', {31, 16, 30, 1, 1, 17, 14}},
    {'6', {6, 8, 16, 30, 17, 17, 14}},   {'7', {31, 1, 2, 4, 8, 8, 8}},   {'8', {14, 17, 17, 14, 17, 17, 14}},
    {'9', {14, 17, 17, 15, 1, 2, 12}},   {'A', {14, 17, 17, 31, 17, 17, 17}}, {'B', {30, 17, 17, 30, 17, 17, 30}},
    {'C', {14, 17, 16, 16, 16, 17, 14}}, {'D', {28, 18, 17, 17, 17, 18, 28}}, {'E', {31, 16, 16, 30, 16, 16, 31}},
    {'F', {31, 16, 16, 30, 16, 16, 16}}, {'G', {14, 17, 16, 23, 17, 17, 15}}, {'H', {17, 17, 17, 31, 17, 17, 17}},
    {'I', {14, 4, 4, 4, 4, 4, 14}},      {'J', {7, 2, 2, 2, 2, 18, 12}},  {'K', {17, 18, 20, 24, 20, 18, 17}},
    {'L', {16, 16, 16, 16, 16, 16, 31}}, {'M', {17, 27, 21, 21, 17, 17, 17}}, {'N', {17, 17, 25, 21, 19, 17, 17}},
    {'O', {14, 17, 17, 17, 17, 17, 14}}, {'P', {30, 17, 17, 30, 16, 16, 16}}, {'Q', {14, 17, 17, 17, 21, 18, 13}},
    {'R', {30, 17, 17, 30, 20, 18, 17}}, {'S', {15, 16, 16, 14, 1, 1, 30}}, {'T', {31, 4, 4, 4, 4, 4, 4}},
    {'U', {17, 17, 17, 17, 17, 17, 14}}, {'V', {17, 17, 17, 17, 17, 10, 4}}, {'W', {17, 17, 17, 21, 21, 21, 10}},
    {'X', {17, 17, 10, 4, 10, 17, 17}},  {'Y', {17, 17, 10, 4, 4, 4, 4}},  {'Z', {31, 1, 2, 4, 8, 16, 31}},
    {'.', {0, 0, 0, 0, 0, 12, 12}},      {':', {0, 12, 12, 0, 12, 12, 0}}, {'/', {1, 1, 2, 4, 8, 16, 16}},
    {'%', {24, 25, 2, 4, 8, 19, 3}},     {'-', {0, 0, 0, 31, 0, 0, 0}},   {'(', {2, 4, 8, 8, 8, 4, 2}},
    {')', {8, 4, 2, 2, 2, 4, 8}},        {'|', {4, 4, 4, 4, 4, 4, 4}},    {'+', {0, 4, 4, 31, 4, 4, 0}},
    {',', {0, 0, 0, 0, 12, 4, 8}},       {'=', {0, 0, 31, 0, 31, 0, 0}},
};
const uint8_t* glyph(char c) {
    if (c >= 'a' && c <= 'z') c = char(c - 32);
    for (auto& g : kFont)
        if (g.c == c) return g.rows;
    return nullptr;
}

constexpr int S = 2;  // screen pixels per font pixel
std::vector<VkClearRect> g_text, g_shadow, g_bars[3];

void rect(std::vector<VkClearRect>& out, int x, int y, int w, int h) {
    if (w > 0 && h > 0) out.push_back({{{x, y}, {uint32_t(w), uint32_t(h)}}, 0, 1});
}
void text(int x, int y, const char* s) {
    for (; *s; ++s, x += 6 * S) {
        const uint8_t* g = glyph(*s);
        if (!g) continue;
        for (int r = 0; r < 7; ++r)
            for (int c = 0; c < 5;) {
                if (!(g[r] & (16 >> c))) { ++c; continue; }
                int run = c;
                while (run < 5 && (g[r] & (16 >> run))) ++run;
                rect(g_text, x + c * S, y + r * S, (run - c) * S, S);
                rect(g_shadow, x + c * S + S / 2 + 1, y + r * S + S / 2 + 1, (run - c) * S, S);
                c = run;
            }
    }
}

double g_ms[120] = {};
int g_at = 0;
bool g_visible = true;
std::chrono::steady_clock::time_point g_last{};

void frame() {
    auto now = std::chrono::steady_clock::now();
    if (g_last.time_since_epoch().count()) g_ms[g_at++ % 120] = std::chrono::duration<double, std::milli>(now - g_last).count();
    g_last = now;
    static bool started = [] {
        char v[4]{};
        if (GetEnvironmentVariableA("KEYLIME_OVERLAY", v, 4) && v[0] == '0') g_visible = false;
        return true;
    }();
    (void)started;
    static bool was = false;
    bool down = (GetAsyncKeyState(VK_F3) & 0x8000) != 0;
    if (down && !was) g_visible = !g_visible;
    was = down;
}

// Rebuilds the rectangles for this frame from the counters.
void build() {
    g_text.clear(), g_shadow.clear();
    for (auto& b : g_bars) b.clear();
    using namespace stats;
    auto v = [](Counter c) { return (unsigned long long)g_counters[c].load(std::memory_order_relaxed); };
    // Per-second rates from the last second of counters.
    static uint64_t prev[Count] = {}, rate[Count] = {};
    static auto stamp = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    if (now - stamp >= std::chrono::seconds(1)) {
        double sec = std::chrono::duration<double>(now - stamp).count();
        for (int i = 0; i < Count; ++i) {
            uint64_t x = g_counters[i].load(std::memory_order_relaxed);
            rate[i] = uint64_t((x - prev[i]) / sec + 0.5), prev[i] = x;
        }
        stamp = now;
    }
    double sum = 0, worst = 0;
    int n = std::min(g_at, 120);
    for (int i = 0; i < n; ++i) sum += g_ms[i], worst = std::max(worst, g_ms[i]);
    double avg = n ? sum / n : 0;
    char b[128];
    int x = 10, y = 10, line = 9 * S;
    snprintf(b, sizeof b, "KEYLIME  %.0f FPS  %.1f MS  (WORST %.0f)", avg > 0 ? 1000.0 / avg : 0.0, avg, worst);
    text(x, y, b), y += line;
    // Graph: 120 frames, 0..66 ms tall (33 ms line = 30 fps).
    const int gh = 40, gw = 2;
    for (int i = 0; i < 120; ++i) {
        double ms = g_ms[(g_at + i) % 120];
        int hgt = std::min(gh, int(ms * gh / 66.0 + 0.5));
        int band = ms <= 20 ? 0 : ms <= 40 ? 1 : 2;
        rect(g_bars[band], x + i * gw, y + gh - hgt, gw - 1 > 0 ? gw - 1 : 1, hgt);
    }
    rect(g_shadow, x, y + gh / 2, 120 * gw, 1);  // the 30 fps line
    y += gh + 6;
    snprintf(b, sizeof b, "DRAWS %llu/S  DISPATCHES %llu/S  SUBMITS %llu/S", (unsigned long long)rate[DrawsRun],
             (unsigned long long)rate[DispatchesRun], (unsigned long long)rate[Submits]);
    text(x, y, b), y += line;
    snprintf(b, sizeof b, "SHADERS  FOR DRAWS %llu  EARLY %llu  OPTIMIZING %llu  BUILT %llu  SKIPPED WAITING %llu/S", v(CompileUrgent),
             v(CompileWarm), v(CompileOptimized), v(PipelinesBuilt), (unsigned long long)rate[WaitingOnCompile]);
    text(x, y, b), y += line;
    snprintf(b, sizeof b, "VRAM %llu / %llu MIB  IMAGES %llu  UPLOADS %llu/S", v(ImageMiB), v(VramBudgetMiB), v(ImagesResident),
             (unsigned long long)rate[TextureUploads]);
    text(x, y, b), y += line;
    snprintf(b, sizeof b, "STREAMING %.1f MB/S  XMA VOICES %llu  OPUS %llu/S", rate[StorageBytes] / 1048576.0, v(XmaVoices),
             (unsigned long long)rate[OpusPackets]);
    text(x, y, b), y += line;
}

void clear(VkCommandBuffer cb, std::vector<VkClearRect>& rects, float r, float g, float bl, VkExtent2D ext) {
    // Keep inside the image (clear rects must be).
    std::erase_if(rects, [&](const VkClearRect& c) {
        return c.rect.offset.x + int(c.rect.extent.width) > int(ext.width) || c.rect.offset.y + int(c.rect.extent.height) > int(ext.height);
    });
    if (rects.empty()) return;
    VkClearAttachment a{VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
    a.clearValue.color = {{r, g, bl, 1}};
    vk::vkCmdClearAttachments(cb, 1, &a, uint32_t(rects.size()), rects.data());
}

// Called with the image in TRANSFER_DST layout; leaves it there.
void draw(VkCommandBuffer cb, VkImage img, VkImageView view, VkExtent2D ext) {
    if (!g_visible || !view) return;
    build();
    layout(cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    ca.imageView = view;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, ext};
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vk::vkCmdBeginRendering(cb, &ri);
    clear(cb, g_shadow, 0, 0, 0, ext);
    clear(cb, g_bars[0], 0.35f, 0.9f, 0.45f, ext);
    clear(cb, g_bars[1], 0.95f, 0.8f, 0.25f, ext);
    clear(cb, g_bars[2], 0.95f, 0.3f, 0.3f, ext);
    clear(cb, g_text, 0.92f, 0.95f, 0.9f, ext);
    vk::vkCmdEndRendering(cb);
    layout(cb, img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
}

}  // namespace overlay
