// keylime formats: BCPack decoder, ported from VIctory's bcpack.rs (GPL-3.0).
//
// A packed stream has no header. It is 12 or 16 interleaved rANS lanes; each
// lane is a run of fragments (header byte, length, payload) coded with one of
// 128 static models, and a fixed schedule decides which lane takes the next
// input byte. The lanes' symbols form one record per block: two back
// references (to one of the last 256 blocks, or to a per-format dictionary
// standing in for blocks -1 to -255) and the residuals of the endpoints and
// indices against them.
#include "formats/bcpack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "bcpack_tables.h"

namespace kl::bcpack {
namespace {

constexpr size_t MODELS = 128, BUCKETS = 9, HISTORY = 256, DICTIONARY = 255, ROUND_BYTES = 8;
constexpr uint32_t LOW = 1u << 24;

struct Plain {
    uint8_t ep[8]{};
    uint8_t ix[32]{};
};
struct Bc6Ep {
    uint16_t e[4][3]{};
};
struct Bc6Ix {
    uint8_t mode = 0;
    int16_t ix[16]{};
};
struct Bc7Ep {
    uint16_t ctrl = 0;
    uint16_t rows[3][8]{};
};
struct Bc7Ix {
    uint8_t bits[2]{};
    int16_t ix[2][16]{};
};

struct Tables {
    uint32_t freq[MODELS][BUCKETS]{};
    uint32_t start[MODELS][BUCKETS + 1]{};
    uint8_t bucket[MODELS][256]{};
    uint64_t recip[MODELS][BUCKETS]{};
    uint8_t rank1[256], rank3a[64], rank3c[256], rank4[64], rank5[64];
    uint16_t ctrl7[512];
    std::vector<Plain> bc1, bc3, bc4, bc5;
    std::vector<std::pair<Bc6Ep, Bc6Ix>> bc6;
    std::vector<std::pair<Bc7Ep, Bc7Ix>> bc7;
};

struct Reader {
    const uint8_t* p;
    uint8_t u8() { return *p++; }
    uint16_t u16() {
        uint16_t v = uint16_t(p[0] | p[1] << 8);
        p += 2;
        return v;
    }
    template <size_t N>
    void bytes(uint8_t (&out)[N]) {
        memcpy(out, p, N);
        p += N;
    }
};

const char* parse(Tables& t) {
    Reader r{TABLE_DATA};
    for (size_t m = 0; m < MODELS; ++m) {
        for (auto& f : t.freq[m]) f = r.u16();
        size_t used = m / 16 + 2;  // model 16 g + shape has g + 2 buckets; bucket b > 0 holds 2^(b-1) symbols
        uint32_t at = 0;
        for (size_t b = 0; b < used; ++b) {
            if (!t.freq[m][b]) return "a model has an empty bucket";
            t.start[m][b] = at;
            t.recip[m][b] = ((1ull << 32) + t.freq[m][b] - 1) / t.freq[m][b];
            at += t.freq[m][b] << (b ? b - 1 : 0);
        }
        if (at != 1u << 16) return "a model does not tile 65536";
        for (size_t b = used; b <= BUCKETS; ++b) t.start[m][b] = 1u << 16;
        for (uint32_t w = 0; w < 256; ++w) {
            uint32_t first = w << 8;
            uint8_t n = 0;
            for (size_t b = 1; b <= BUCKETS; ++b)
                if (t.start[m][b] <= first) ++n;
            t.bucket[m][w] = n;
        }
    }
    r.bytes(t.rank1), r.bytes(t.rank3a), r.bytes(t.rank3c), r.bytes(t.rank4), r.bytes(t.rank5);
    for (auto& c : t.ctrl7) c = r.u16();
    auto plain = [&](std::vector<Plain>& out, size_t ep, size_t ix) {
        out.resize(DICTIONARY);
        for (auto& p : out) {
            for (size_t k = 0; k < ep; ++k) p.ep[k] = r.u8();
            for (size_t k = 0; k < ix; ++k) p.ix[k] = r.u8();
        }
    };
    plain(t.bc1, 6, 16), plain(t.bc3, 8, 32), plain(t.bc4, 2, 16), plain(t.bc5, 4, 32);
    t.bc6.resize(DICTIONARY);
    for (auto& [ep, ix] : t.bc6) {
        r.u8();  // the endpoint entry's mode byte plays no part
        for (auto& row : ep.e)
            for (auto& v : row) v = r.u16();
        ix.mode = r.u8();
        for (auto& v : ix.ix) v = int16_t(r.u16());
    }
    t.bc7.resize(DICTIONARY);
    for (auto& [ep, ix] : t.bc7) {
        ep.ctrl = r.u16();
        for (auto& row : ep.rows)
            for (auto& v : row) v = r.u16();
        ix.bits[0] = r.u8(), ix.bits[1] = r.u8();
        for (auto& set : ix.ix)
            for (auto& v : set) v = int16_t(r.u16());
    }
    if (size_t(r.p - TABLE_DATA) != TABLE_BYTES) return "table file has the wrong size";
    return nullptr;
}

const Tables* tables(const char*& why) {
    static std::unique_ptr<Tables> t;
    static const char* error = [] {
        t = std::make_unique<Tables>();
        return parse(*t);
    }();
    why = error;
    return error ? nullptr : t.get();
}

// ---------------------------------------------------------------- lanes ----
struct Lane {
    uint8_t width = 0, phase = 0, model = 0;
    bool longf = false, reset = false, need = false, stalled = false;
    bool has_latch = false, has_out = false;
    uint8_t latch = 0, out = 0;
    uint32_t x = 0, left = 0;

    size_t bucket_of(const Tables& t, uint32_t slot) const {
        const uint32_t* start = t.start[model];
        size_t b = t.bucket[model][slot >> 8];
        while (start[b + 1] <= slot) ++b;
        return b;
    }

    void step(const Tables& t) {
        if (phase != 3) {
            if (!has_latch) return;
            uint8_t b = latch;
            has_latch = false;
            need = true;
            if (phase == 0) {
                uint8_t group = uint8_t(width - (b >> 6));
                model = uint8_t(group * 16 + (b & 15));
                reset = (b & 0x10) != 0;
                longf = (b & 0x20) != 0;
                left = 0;
                phase = 1;
            } else if (phase == 1 && longf) {
                left = b;
                phase = 2;
            } else if (phase == 1) {
                left = uint32_t(b) + 1;
                phase = 3;
            } else {
                left = ((uint32_t(b) << 8) | left) + 257;
                phase = 3;
            }
            return;
        }
        uint32_t xx = x;
        if (xx < LOW && has_latch) {
            has_latch = false;
            left -= 1;
            xx = (xx << 8) | latch;
        }
        need = left != 0 && !has_latch;
        if (xx > LOW) {
            const uint32_t* freq = t.freq[model];
            const uint32_t* start = t.start[model];
            uint32_t slot = xx & 0xffff;
            size_t b = bucket_of(t, slot);
            uint32_t sym, f, rem;
            if (b == 0) {
                sym = 0, f = freq[0] - 1, rem = slot - 1;
            } else {
                uint32_t offset = slot - start[b];
                uint32_t q = uint32_t((uint64_t(offset) * t.recip[model][b]) >> 32);
                sym = (1u << (b - 1)) + q, f = freq[b], rem = offset - q * freq[b];
            }
            x = (xx >> 16) * f + rem;
            out = uint8_t(sym), has_out = true;
        } else if (left != 0) {
            x = xx;
        } else {
            phase = 0;  // the fragment ends; a byte read in this step does not reach the carried state
            need = !has_latch;
            if (reset) x = 0;
        }
    }
};

const char* lanes(const Tables& t, const uint8_t* widths, size_t nw, size_t per, size_t blocks, const uint8_t* src, size_t src_len,
                  uint8_t* records) {
    Lane ls[16];
    for (size_t j = 0; j < nw; ++j) ls[j].width = widths[j], ls[j].need = true;
    size_t stride = nw * per, read = 0;
    uint32_t full = (1u << nw) - 1, done = 0;
    size_t pos = 0, at = 0;
    int idle = 0;
    while (pos < blocks * per) {
        size_t taken = 0;
        bool moved = false;
        for (size_t j = 0; j < nw; ++j) {
            Lane& lane = ls[j];
            uint32_t bit = 1u << j;
            size_t before = taken;
            if (!lane.stalled) lane.step(t);
            bool waiting = lane.has_out && (done & bit);
            if (lane.need && !waiting && taken < ROUND_BYTES && read < src_len) {
                lane.latch = src[read++], lane.has_latch = true;
                ++taken;
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if (!lane.stalled && lane.has_out) {
                lane.has_out = false;
                records[at + j * per] = lane.out;
                done |= bit;
                moved = true;
            }
        }
        if (done == full) {
            ++pos;
            at += pos % per == 0 ? stride - per + 1 : 1;
            done = 0;
            moved = true;
        }
        idle = (taken > 0 || moved) ? 0 : idle + 1;
        if (idle > 3) return "the stream ends early or is corrupt";
    }
    return nullptr;
}

// --------------------------------------------------------------- blocks ----
uint8_t interleave(uint8_t a, uint8_t b) {
    auto spread = [](uint8_t v) {
        uint8_t acc = 0;
        for (int k = 0; k < 4; ++k) acc |= uint8_t(((v >> k) & 1) << (2 * k));
        return acc;
    };
    return uint8_t(spread(a) | spread(b) << 1);
}
void references(const uint8_t* s, bool paired, size_t& ep_ref, size_t& ix_ref) {
    uint8_t d = paired ? interleave(s[0], s[1]) : s[0];
    uint8_t ix = paired ? interleave(s[2], s[3]) : s[1];
    uint8_t delta = uint8_t((d >> 1) ^ uint8_t(-(d & 1)));
    ep_ref = uint8_t(ix + delta), ix_ref = ix;
}
template <class T>
const T& pick(size_t i, size_t r, const std::vector<T>& history, const std::vector<T>& dictionary, const T& zero) {
    if (r == 0) return zero;
    if (r <= i) return history[(i - r) % HISTORY];
    return dictionary[r - i - 1];
}
uint8_t zz(uint8_t v, uint32_t bits) {
    uint8_t mask = uint8_t((1u << bits) - 1);
    return uint8_t(((v >> 1) ^ ((v & 1) ? mask : 0)) & mask);
}

struct Widths {
    uint8_t ep[8];
    uint8_t ix[32];
    bool paired;
};
constexpr Widths make_widths(std::array<uint8_t, 8> ep, int ix3, int ix2, bool paired) {
    Widths w{};
    for (int k = 0; k < 8; ++k) w.ep[k] = ep[k];
    for (int k = 0; k < ix3 + ix2; ++k) w.ix[k] = k < ix3 ? 3 : 2;
    w.paired = paired;
    return w;
}
constexpr Widths W1 = make_widths({5, 6, 5, 5, 6, 5, 0, 0}, 0, 16, false);
constexpr Widths W3 = make_widths({5, 6, 5, 5, 6, 5, 8, 8}, 16, 16, true);
constexpr Widths W4 = make_widths({8, 8, 0, 0, 0, 0, 0, 0}, 16, 0, false);
constexpr Widths W5 = make_widths({8, 8, 8, 8, 0, 0, 0, 0}, 32, 0, true);

void quads(const uint8_t* rank, const uint8_t* s, uint8_t* ix) {
    static const int P[4] = {0, 1, 4, 5};
    for (int q = 0; q < 4; ++q) {
        uint8_t v = rank[s[q]];
        int base = (q >> 1) * 8 + (q & 1) * 2;
        for (int k = 0; k < 4; ++k) ix[base + P[k]] = (v >> (2 * k)) & 3;
    }
}
void pairs(const uint8_t* rank, const uint8_t* s, size_t n, uint8_t* ix) {
    for (size_t k = 0; k < n; ++k) {
        uint8_t v = rank[s[k] & 63];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = (v >> 4) & 7;
    }
}

Plain bc1_residual(const Tables& t, const uint8_t* s) {
    Plain v;
    static const uint32_t bits[6] = {5, 6, 5, 5, 6, 5};
    for (int k = 0; k < 6; ++k) v.ep[k] = zz(s[2 + k], bits[k]);
    quads(t.rank1, s + 8, v.ix);
    return v;
}
Plain bc3_residual(const Tables& t, const uint8_t* s) {
    Plain v;
    uint8_t ep[8] = {zz(s[14], 5), zz(s[16], 6), zz(s[18], 5), zz(s[15], 5), zz(s[17], 6), zz(s[19], 5), zz(s[4], 8), zz(s[5], 8)};
    memcpy(v.ep, ep, 8);
    pairs(t.rank3a, s + 6, 8, v.ix);
    quads(t.rank3c, s + 20, v.ix + 16);
    return v;
}
Plain bc4_residual(const Tables& t, const uint8_t* s) {
    Plain v;
    v.ep[0] = zz(s[2], 8), v.ep[1] = zz(s[3], 8);
    pairs(t.rank4, s + 4, 8, v.ix);
    return v;
}
Plain bc5_residual(const Tables& t, const uint8_t* s) {
    Plain v;
    for (int k = 0; k < 4; ++k) v.ep[k] = zz(s[4 + k], 8);
    pairs(t.rank5, s + 8, 16, v.ix);
    return v;
}

const uint32_t BC1_CODE[4] = {0, 2, 3, 1};
const uint64_t BC4_CODE[8] = {0, 2, 3, 4, 5, 6, 7, 1};

void bc1_color(const uint8_t* ep, const uint8_t* ix, uint8_t* out) {
    uint16_t c0 = uint16_t(ep[0] << 11 | ep[1] << 5 | ep[2]);
    uint16_t c1 = uint16_t(ep[3] << 11 | ep[4] << 5 | ep[5]);
    uint32_t bits = 0;
    for (int p = 0; p < 16; ++p) bits |= BC1_CODE[ix[p] & 3] << (2 * p);
    memcpy(out, &c0, 2), memcpy(out + 2, &c1, 2), memcpy(out + 4, &bits, 4);
}
void bc4_alpha(uint8_t e0, uint8_t e1, const uint8_t* ix, uint8_t* out) {
    uint64_t bits = 0;
    for (int p = 0; p < 16; ++p) bits |= BC4_CODE[ix[p] & 7] << (16 + 3 * p);
    uint64_t v = uint64_t(uint8_t(e0 ^ 0x80)) | uint64_t(uint8_t(e1 ^ 0x80)) << 8 | bits;
    memcpy(out, &v, 8);
}

template <class R, class P>
void plain(const uint8_t* records, size_t stride, uint8_t* dst, size_t block, size_t blocks, const std::vector<Plain>& dict, const Widths& w,
           R residual, P pack) {
    std::vector<Plain> history(HISTORY);
    Plain zero;
    for (size_t i = 0; i < blocks; ++i) {
        const uint8_t* s = records + i * stride;
        size_t er, ir;
        references(s, w.paired, er, ir);
        const Plain& epr = pick(i, er, history, dict, zero);
        const Plain& ixr = pick(i, ir, history, dict, zero);
        Plain v = residual(s);
        for (int k = 0; k < 8; ++k) v.ep[k] = uint8_t((v.ep[k] + epr.ep[k]) & uint8_t((1u << w.ep[k]) - 1));
        for (int k = 0; k < 32; ++k) v.ix[k] = uint8_t((v.ix[k] + ixr.ix[k]) & uint8_t((1u << w.ix[k]) - 1));
        history[i % HISTORY] = v;
        pack(v, dst + i * block);
    }
}

// A byte's even and odd bits as two zigzag-coded 4-bit residuals.
std::array<int16_t, 2> nibbles(uint8_t b, bool odd_first) {
    auto half = [](uint8_t v) {
        uint8_t acc = 0;
        for (int k = 0; k < 4; ++k) acc |= uint8_t(((v >> (2 * k)) & 1) << k);
        return acc;
    };
    auto z = [](uint8_t u) { return int16_t(int16_t(u >> 1) ^ -int16_t(u & 1)); };
    int16_t even = z(half(b)), odd = z(half(uint8_t(b >> 1)));
    return odd_first ? std::array<int16_t, 2>{odd, even} : std::array<int16_t, 2>{even, odd};
}
int16_t add_index(int16_t cur, uint8_t bits, int16_t refv, uint8_t ref_bits) {
    auto up = [](int16_t v, uint8_t b) -> uint16_t { return (b == 0 || b > 16) ? 0 : uint16_t(uint16_t(v) << (16 - b)); };
    int16_t sum = int16_t(uint16_t(up(cur, bits) + up(refv, ref_bits)));
    return bits == 0 ? 0 : int16_t(sum >> (16 - bits));
}

// Little-endian bit writer for one 128-bit block.
struct Bits {
    uint64_t lo = 0, hi = 0;
    uint32_t at = 0;
    void put(uint64_t value, uint32_t n) {
        if (n > 0 && at < 128) {
            uint64_t v = n >= 64 ? value : value & ((1ull << n) - 1);
            if (at < 64) {
                lo |= v << at;
                if (at && at + n > 64) hi |= v >> (64 - at);
            } else {
                hi |= v << (at - 64);
            }
        }
        at += n;
    }
    void store(uint8_t* out) const { memcpy(out, &lo, 8), memcpy(out + 8, &hi, 8); }
};

// ------------------------------------------------------------------ BC7 ----
const uint8_t ANCHOR2[64] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2,
                             8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15,
                             15, 15, 15, 2, 2, 15};
const uint8_t ANCHOR3A[64] = {3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15,
                              8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3};
const uint8_t ANCHOR3B[64] = {15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8,
                              15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8};
const uint8_t BC7_BITS[8][4] = {{4, 4, 4, 0}, {6, 6, 6, 0}, {5, 5, 5, 0}, {7, 7, 7, 0}, {5, 5, 5, 6}, {7, 7, 7, 8}, {7, 7, 7, 7}, {5, 5, 5, 5}};
const uint8_t BC7_INDEX[8][2] = {{3, 0}, {3, 0}, {2, 0}, {2, 0}, {3, 2}, {2, 2}, {4, 0}, {2, 0}};

bool pbits(uint16_t mode) { return mode == 0 || mode == 1 || mode == 3 || mode == 6 || mode == 7; }
std::array<uint8_t, 4> bc7_precision(uint16_t ctrl) {
    uint16_t mode = ctrl & 7;
    size_t rot = (ctrl >> 3) & 3;
    std::array<uint8_t, 4> p = {BC7_BITS[mode][0], BC7_BITS[mode][1], BC7_BITS[mode][2], BC7_BITS[mode][3]};
    if (rot) std::swap(p[3], p[rot - 1]);
    else if (pbits(mode)) p[0] += 1;
    return p;
}

void bc7_pack(const Bc7Ep& ep, const Bc7Ix& ix, std::array<uint8_t, 4> prec, uint8_t* out) {
    uint16_t ctrl = ep.ctrl, mode = ctrl & 7;
    size_t subsets = (ctrl & 0xff) >> 6, part = (ctrl >> 8) & 63, rot = (ctrl >> 3) & 3;
    Bits w;
    w.put(1ull << mode, uint32_t(mode) + 1);
    switch (mode) {
        case 0: w.put(part, 4); break;
        case 1: case 2: case 3: case 7: w.put(part, 6); break;
        case 4: w.put(rot, 2), w.put((ctrl >> 5) & 1, 1); break;
        case 5: w.put(rot, 2); break;
        default: break;
    }
    size_t ends = 2 * subsets;
    auto field = [&](size_t row, size_t col, uint8_t p) { return uint64_t(ep.rows[row][col] >> (16 - p)); };
    int order[4] = {0, 1, 2, 3};
    if (rot) std::swap(order[3], order[rot - 1]);
    for (int ch : order) {
        uint8_t p = prec[ch];
        if (!p) continue;
        uint32_t drop = (ch == 0 && pbits(mode)) ? 1 : 0;
        for (size_t e = 0; e < ends; ++e) w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, p - drop);
    }
    if (mode == 1) {
        for (size_t row = 0; row < subsets; ++row) w.put(field(row, 0, prec[0]) & 1, 1);
    } else if (pbits(mode)) {
        for (size_t e = 0; e < ends; ++e) w.put(field(e / 2, e % 2, prec[0]) & 1, 1);
    }
    uint8_t anchors[3] = {0, 255, 255};
    if (subsets == 2) anchors[1] = ANCHOR2[part];
    if (subsets == 3) anchors[1] = ANCHOR3A[part], anchors[2] = ANCHOR3B[part];
    auto indices = [&](int set) {
        uint8_t bits = ix.bits[set];
        for (int p = 0; p < 16; ++p) {
            bool anchor = anchors[0] == p || anchors[1] == p || anchors[2] == p;
            w.put(uint16_t(ix.ix[set][p]), uint32_t(bits) - (anchor ? 1 : 0));
        }
    };
    if (mode == 4) indices(1), indices(0);
    else if (mode == 5) indices(0), indices(1);
    else indices(0);
    w.store(out);
}

const char* bc7(const uint8_t* records, size_t stride, uint8_t* dst, size_t blocks, const Tables& t) {
    std::vector<std::pair<Bc7Ep, Bc7Ix>> history(HISTORY);
    std::pair<Bc7Ep, Bc7Ix> zero{};
    for (size_t i = 0; i < blocks; ++i) {
        const uint8_t* s = records + i * stride;
        size_t er, ir;
        references(s, true, er, ir);
        uint16_t ctrl = t.ctrl7[size_t(s[31] & 15) << 5 | (s[30] & 31)];
        if (!ctrl) return "unused BC7 control code";
        size_t mode = ctrl & 7, subsets = (ctrl & 0xff) >> 6;
        auto prec = bc7_precision(ctrl);
        bool three = subsets == 3;
        auto or0 = [&](uint8_t v) { return three ? uint8_t(0) : v; };
        auto if3 = [&](uint8_t v) { return three ? v : uint8_t(0); };
        uint8_t stored[3][8] = {{s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])},
                                {s[12], s[13], s[14], s[15], s[16], s[17], or0(s[18]), or0(s[19])},
                                {s[20], s[21], if3(s[10]), if3(s[11]), if3(s[18]), if3(s[19]), 0, 0}};
        const Bc7Ep& r = pick(i, er, history, t.bc7, zero).first;
        Bc7Ep ep;
        ep.ctrl = ctrl;
        size_t common = std::min<size_t>(subsets, (r.ctrl & 0xff) >> 6);
        size_t columns = prec[3] > 0 ? 8 : 6;
        for (size_t c = 0; c < columns; ++c) {
            uint8_t p = prec[c / 2];
            if (!p) continue;
            uint16_t mask = uint16_t(0xffffu << (16 - p));
            for (size_t row = 0; row < subsets && row < 3; ++row) {
                uint8_t v = stored[row][c];
                uint16_t residual = uint16_t(uint16_t((v >> 1) ^ ((v & 1) ? 0xffff : 0)) << (16 - p));
                uint16_t base = r.rows[row < common ? row : 0][c] & mask;
                ep.rows[row][c] = uint16_t(base + residual);
            }
        }
        const uint8_t* bits = BC7_INDEX[mode];
        const Bc7Ix& rx = pick(i, ir, history, t.bc7, zero).second;
        Bc7Ix ix;
        ix.bits[0] = bits[0], ix.bits[1] = bits[1];
        static const size_t OFFS[2][4] = {{22, 24, 26, 28}, {12, 14, 16, 20}};
        for (int set = 0; set < 2; ++set) {
            if (!bits[set]) continue;
            for (int k = 0; k < 4; ++k)
                for (int b = 0; b < 2; ++b) {
                    auto pair = nibbles(s[OFFS[set][k] + b], k < 3);
                    for (int h = 0; h < 2; ++h) {
                        int p = 4 * k + 2 * b + h;
                        ix.ix[set][p] = add_index(pair[h], bits[set], rx.ix[set][p], rx.bits[set]);
                    }
                }
        }
        history[i % HISTORY] = {ep, ix};
        bc7_pack(ep, ix, prec, dst + i * 16);
    }
    return nullptr;
}

// ----------------------------------------------------------------- BC6H ----
const uint8_t BC6_BITS[15][4] = {{0, 0, 0, 0}, {10, 5, 5, 5}, {7, 6, 6, 6}, {11, 5, 4, 4}, {11, 4, 5, 4}, {11, 4, 4, 5}, {9, 5, 5, 5}, {8, 6, 5, 5},
                                 {8, 5, 6, 5},  {8, 5, 5, 6},  {6, 6, 6, 6}, {10, 10, 10, 10}, {11, 9, 9, 9}, {12, 8, 8, 8}, {16, 4, 4, 4}};
const std::pair<uint64_t, uint32_t> BC6_CODE[15] = {{0, 0}, {0, 2}, {1, 2}, {2, 5}, {6, 5}, {10, 5}, {14, 5}, {18, 5},
                                                    {22, 5}, {26, 5}, {30, 5}, {3, 5}, {7, 5}, {11, 5}, {15, 5}};
struct Run {
    uint8_t ch, ep, first, len;
};
constexpr uint8_t R = 0, G = 1, B = 2;
constexpr Run D{3, 0, 0, 5};
#define r_(c, e, f, l) Run{c, e, f, l}
const std::vector<Run> BC6_LAYOUT[15] = {
    {},
    {r_(G, 2, 4, 1), r_(B, 2, 4, 1), r_(B, 3, 4, 1), r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 5), r_(G, 3, 4, 1),
     r_(G, 2, 0, 4), r_(G, 1, 0, 5), r_(B, 3, 0, 1), r_(G, 3, 0, 4), r_(B, 1, 0, 5), r_(B, 3, 1, 1), r_(B, 2, 0, 4), r_(R, 2, 0, 5),
     r_(B, 3, 2, 1), r_(R, 3, 0, 5), r_(B, 3, 3, 1), D},
    {r_(G, 2, 5, 1), r_(G, 3, 4, 1), r_(G, 3, 5, 1), r_(R, 0, 0, 7), r_(B, 3, 0, 1), r_(B, 3, 1, 1), r_(B, 2, 4, 1), r_(G, 0, 0, 7),
     r_(B, 2, 5, 1), r_(B, 3, 2, 1), r_(G, 2, 4, 1), r_(B, 0, 0, 7), r_(B, 3, 3, 1), r_(B, 3, 5, 1), r_(B, 3, 4, 1), r_(R, 1, 0, 6),
     r_(G, 2, 0, 4), r_(G, 1, 0, 6), r_(G, 3, 0, 4), r_(B, 1, 0, 6), r_(B, 2, 0, 4), r_(R, 2, 0, 6), r_(R, 3, 0, 6), D},
    {r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 5), r_(R, 0, 10, 1), r_(G, 2, 0, 4), r_(G, 1, 0, 4), r_(G, 0, 10, 1),
     r_(B, 3, 0, 1), r_(G, 3, 0, 4), r_(B, 1, 0, 4), r_(B, 0, 10, 1), r_(B, 3, 1, 1), r_(B, 2, 0, 4), r_(R, 2, 0, 5), r_(B, 3, 2, 1),
     r_(R, 3, 0, 5), r_(B, 3, 3, 1), D},
    {r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 4), r_(R, 0, 10, 1), r_(G, 3, 4, 1), r_(G, 2, 0, 4), r_(G, 1, 0, 5),
     r_(G, 0, 10, 1), r_(G, 3, 0, 4), r_(B, 1, 0, 4), r_(B, 0, 10, 1), r_(B, 3, 1, 1), r_(B, 2, 0, 4), r_(R, 2, 0, 4), r_(B, 3, 0, 1),
     r_(B, 3, 2, 1), r_(R, 3, 0, 4), r_(G, 2, 4, 1), r_(B, 3, 3, 1), D},
    {r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 4), r_(R, 0, 10, 1), r_(B, 2, 4, 1), r_(G, 2, 0, 4), r_(G, 1, 0, 4),
     r_(G, 0, 10, 1), r_(B, 3, 0, 1), r_(G, 3, 0, 4), r_(B, 1, 0, 5), r_(B, 0, 10, 1), r_(B, 2, 0, 4), r_(R, 2, 0, 4), r_(B, 3, 1, 1),
     r_(B, 3, 2, 1), r_(R, 3, 0, 4), r_(B, 3, 4, 1), r_(B, 3, 3, 1), D},
    {r_(R, 0, 0, 9), r_(B, 2, 4, 1), r_(G, 0, 0, 9), r_(G, 2, 4, 1), r_(B, 0, 0, 9), r_(B, 3, 4, 1), r_(R, 1, 0, 5), r_(G, 3, 4, 1),
     r_(G, 2, 0, 4), r_(G, 1, 0, 5), r_(B, 3, 0, 1), r_(G, 3, 0, 4), r_(B, 1, 0, 5), r_(B, 3, 1, 1), r_(B, 2, 0, 4), r_(R, 2, 0, 5),
     r_(B, 3, 2, 1), r_(R, 3, 0, 5), r_(B, 3, 3, 1), D},
    {r_(R, 0, 0, 8), r_(G, 3, 4, 1), r_(B, 2, 4, 1), r_(G, 0, 0, 8), r_(B, 3, 2, 1), r_(G, 2, 4, 1), r_(B, 0, 0, 8), r_(B, 3, 3, 1),
     r_(B, 3, 4, 1), r_(R, 1, 0, 6), r_(G, 2, 0, 4), r_(G, 1, 0, 5), r_(B, 3, 0, 1), r_(G, 3, 0, 4), r_(B, 1, 0, 5), r_(B, 3, 1, 1),
     r_(B, 2, 0, 4), r_(R, 2, 0, 6), r_(R, 3, 0, 6), D},
    {r_(R, 0, 0, 8), r_(B, 3, 0, 1), r_(B, 2, 4, 1), r_(G, 0, 0, 8), r_(G, 2, 5, 1), r_(G, 2, 4, 1), r_(B, 0, 0, 8), r_(G, 3, 5, 1),
     r_(B, 3, 4, 1), r_(R, 1, 0, 5), r_(G, 3, 4, 1), r_(G, 2, 0, 4), r_(G, 1, 0, 6), r_(G, 3, 0, 4), r_(B, 1, 0, 5), r_(B, 3, 1, 1),
     r_(B, 2, 0, 4), r_(R, 2, 0, 5), r_(B, 3, 2, 1), r_(R, 3, 0, 5), r_(B, 3, 3, 1), D},
    {r_(R, 0, 0, 8), r_(B, 3, 1, 1), r_(B, 2, 4, 1), r_(G, 0, 0, 8), r_(B, 2, 5, 1), r_(G, 2, 4, 1), r_(B, 0, 0, 8), r_(B, 3, 5, 1),
     r_(B, 3, 4, 1), r_(R, 1, 0, 5), r_(G, 3, 4, 1), r_(G, 2, 0, 4), r_(G, 1, 0, 5), r_(B, 3, 0, 1), r_(G, 3, 0, 4), r_(B, 1, 0, 6),
     r_(B, 2, 0, 4), r_(R, 2, 0, 5), r_(B, 3, 2, 1), r_(R, 3, 0, 5), r_(B, 3, 3, 1), D},
    {r_(R, 0, 0, 6), r_(G, 3, 4, 1), r_(B, 3, 0, 1), r_(B, 3, 1, 1), r_(B, 2, 4, 1), r_(G, 0, 0, 6), r_(G, 2, 5, 1), r_(B, 2, 5, 1),
     r_(B, 3, 2, 1), r_(G, 2, 4, 1), r_(B, 0, 0, 6), r_(G, 3, 5, 1), r_(B, 3, 3, 1), r_(B, 3, 5, 1), r_(B, 3, 4, 1), r_(R, 1, 0, 6),
     r_(G, 2, 0, 4), r_(G, 1, 0, 6), r_(G, 3, 0, 4), r_(B, 1, 0, 6), r_(B, 2, 0, 4), r_(R, 2, 0, 6), r_(R, 3, 0, 6), D},
    {r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 10), r_(G, 1, 0, 10), r_(B, 1, 0, 10)},
    {r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 9), r_(R, 0, 10, 1), r_(G, 1, 0, 9), r_(G, 0, 10, 1), r_(B, 1, 0, 9),
     r_(B, 0, 10, 1)},
    {r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 8), r_(R, 0, 10, 2), r_(G, 1, 0, 8), r_(G, 0, 10, 2), r_(B, 1, 0, 8),
     r_(B, 0, 10, 2)},
    {r_(R, 0, 0, 10), r_(G, 0, 0, 10), r_(B, 0, 0, 10), r_(R, 1, 0, 4), r_(R, 0, 10, 6), r_(G, 1, 0, 4), r_(G, 0, 10, 6), r_(B, 1, 0, 4),
     r_(B, 0, 10, 6)},
};
#undef r_

uint8_t bc6_index_bits(uint8_t mode) { return mode >= 1 && mode <= 10 ? 3 : mode >= 11 && mode <= 14 ? 4 : 0; }

const char* bc6(const uint8_t* records, size_t stride, uint8_t* dst, size_t blocks, const std::vector<std::pair<Bc6Ep, Bc6Ix>>& dict) {
    std::vector<std::pair<Bc6Ep, Bc6Ix>> history(HISTORY);
    std::pair<Bc6Ep, Bc6Ix> zero{};
    zero.second.mode = 11;
    auto z = [](uint16_t v) { return uint16_t((v >> 1) ^ uint16_t(-(v & 1))); };
    auto wide = [](uint8_t a, uint8_t b) {
        uint16_t acc = 0;
        for (int k = 0; k < 8; ++k) acc |= uint16_t(((a >> k) & 1) << (2 * k) | ((b >> k) & 1) << (2 * k + 1));
        return acc;
    };
    for (size_t i = 0; i < blocks; ++i) {
        const uint8_t* s = records + i * stride;
        size_t er, ir;
        references(s, true, er, ir);
        uint8_t mode = uint8_t(s[30] + 1);
        if (mode < 1 || mode > 14) return "BC6H mode out of range";
        bool two = mode < 11;
        uint32_t shift[4];
        for (int k = 0; k < 4; ++k) shift[k] = 16 - BC6_BITS[mode][k];
        uint16_t e0[3] = {z(wide(s[4], s[5])), z(wide(s[6], s[7])), z(wide(s[8], s[9]))};
        uint16_t e1[3] = {z(wide(s[10], s[11])), z(wide(s[12], s[13])), z(wide(s[14], s[15]))};
        uint16_t e2[3], e3[3];
        if (two) {
            e2[0] = z(s[16]), e2[1] = z(s[18]), e2[2] = z(s[20]);
            e3[0] = z(s[17]), e3[1] = z(s[19]), e3[2] = z(s[21]);
        } else {
            memcpy(e2, e0, 6), memcpy(e3, e1, 6);
        }
        const Bc6Ep& r = pick(i, er, history, dict, zero).first;
        Bc6Ep ep;
        auto mask = [](uint32_t sh) { return uint16_t(0xffffu << sh); };
        for (int c = 0; c < 3; ++c) ep.e[0][c] = uint16_t(uint16_t(r.e[0][c] + uint16_t(e0[c] << shift[0])) & mask(shift[0]));
        const uint16_t* res[3] = {e1, e2, e3};
        for (int k = 0; k < (two ? 3 : 1); ++k)
            for (int c = 0; c < 3; ++c)
                ep.e[k + 1][c] = uint16_t(uint16_t(r.e[k + 1][c] + uint16_t(res[k][c] << shift[1 + c])) & mask(shift[1 + c]));
        if (!two) {
            for (int c = 0; c < 3; ++c) ep.e[2][c] = uint16_t((ep.e[0][c] >> shift[0]) << shift[1]);
            memcpy(ep.e[3], ep.e[1], 6);
        }
        const Bc6Ix& rx = pick(i, ir, history, dict, zero).second;
        uint8_t bits = bc6_index_bits(mode);
        Bc6Ix ix;
        ix.mode = mode;
        for (int k = 0; k < 8; ++k) {
            auto pair = nibbles(s[22 + k], k < 6);
            for (int h = 0; h < 2; ++h) {
                int p = 2 * k + h;
                ix.ix[p] = add_index(pair[h], bits, rx.ix[p], bc6_index_bits(rx.mode));
            }
        }
        history[i % HISTORY] = {ep, ix};
        auto field = [&](size_t k, size_t c) {
            uint32_t sh = k == 0 ? shift[0] : shift[1 + c];
            return uint64_t(uint16_t(ep.e[k][c] ^ 0x8000) >> sh);
        };
        Bits w;
        w.put(BC6_CODE[mode].first, BC6_CODE[mode].second);
        uint8_t part = s[31] & 31;
        for (const Run& run : BC6_LAYOUT[mode]) {
            if (run.ch == 3) w.put(part, 5);
            else w.put(field(run.ep, run.ch) >> run.first, run.len);
        }
        uint8_t anchor = two ? ANCHOR2[part] : 0;
        for (int p = 0; p < 16; ++p) {
            bool shortp = p == 0 || p == anchor;
            uint16_t v = uint16_t(uint16_t(ix.ix[p]) + (1u << (bits - 1)));
            w.put(v, uint32_t(bits) - (shortp ? 1 : 0));
        }
        w.store(dst + i * 16);
    }
    return nullptr;
}

}  // namespace

const char* decode(unsigned bc, const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_len) {
    const char* why = nullptr;
    const Tables* tp = tables(why);
    if (!tp) return why;
    const Tables& t = *tp;
    size_t block;
    switch (bc) {
        case 1: case 4: block = 8; break;
        case 3: case 5: case 6: case 7: block = 16; break;
        default: return "unsupported BC format";
    }
    if (!dst_len || dst_len % block) return "destination is not a whole number of blocks";
    size_t blocks = dst_len / block;
    static const uint8_t W_1[] = {7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7}, W_3[] = {3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7},
                         W_4[] = {7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5}, W_5[] = {3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5},
                         W_6[] = {3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4}, W_7[] = {3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4};
    const uint8_t* widths;
    size_t nw, per;
    switch (bc) {
        case 1: widths = W_1, nw = 12, per = 1; break;
        case 3: widths = W_3, nw = 12, per = 2; break;
        case 4: widths = W_4, nw = 12, per = 1; break;
        case 5: widths = W_5, nw = 12, per = 2; break;
        case 6: widths = W_6, nw = 16, per = 2; break;
        default: widths = W_7, nw = 16, per = 2; break;
    }
    size_t stride = nw * per;
    std::vector<uint8_t> records(blocks * stride);
    if (const char* e = lanes(t, widths, nw, per, blocks, src, src_len, records.data())) return e;
    const uint8_t* rec = records.data();
    switch (bc) {
        case 1:
            plain(rec, stride, dst, block, blocks, t.bc1, W1, [&](const uint8_t* s) { return bc1_residual(t, s); },
                  [](const Plain& v, uint8_t* o) { bc1_color(v.ep, v.ix, o); });
            return nullptr;
        case 3:
            plain(rec, stride, dst, block, blocks, t.bc3, W3, [&](const uint8_t* s) { return bc3_residual(t, s); },
                  [](const Plain& v, uint8_t* o) {
                      bc4_alpha(v.ep[6], v.ep[7], v.ix, o);
                      bc1_color(v.ep, v.ix + 16, o + 8);
                  });
            return nullptr;
        case 4:
            plain(rec, stride, dst, block, blocks, t.bc4, W4, [&](const uint8_t* s) { return bc4_residual(t, s); },
                  [](const Plain& v, uint8_t* o) { bc4_alpha(v.ep[0], v.ep[1], v.ix, o); });
            return nullptr;
        case 5:
            plain(rec, stride, dst, block, blocks, t.bc5, W5, [&](const uint8_t* s) { return bc5_residual(t, s); },
                  [](const Plain& v, uint8_t* o) {
                      bc4_alpha(v.ep[0], v.ep[1], v.ix, o);
                      bc4_alpha(v.ep[2], v.ep[3], v.ix + 16, o + 8);
                  });
            return nullptr;
        case 6: return bc6(rec, stride, dst, blocks, t.bc6);
        default: return bc7(rec, stride, dst, blocks, t);
    }
}

}  // namespace kl::bcpack
