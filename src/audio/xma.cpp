#include "audio/xma.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#include "audio/xma_tables.h"

namespace kl::xma {
namespace {

using namespace tables;
constexpr size_t PACKET_BITS = PACKET_BYTES * 8, HEADER_BITS = 32;
constexpr uint32_t LENGTH_BITS = 15;
constexpr size_t MIN_SUBFRAME = SUBFRAME_SAMPLES;

// Bit positions in a buffer of packets, skipping headers and other streams' packets.
struct Cursor {
    const uint8_t* data;
    size_t len, pos;
    bool bit(uint32_t& out) {
        if (pos % PACKET_BITS == 0) {
            size_t previous = pos / PACKET_BITS;
            if (previous > 0) {
                auto p = packet(data, len, previous - 1);
                if (!p) return false;
                pos = (previous + p->skip) * PACKET_BITS;
            }
            pos += HEADER_BITS;
        }
        if (pos / 8 >= len) return false;
        out = (data[pos / 8] >> (7 - pos % 8)) & 1;
        ++pos;
        return true;
    }
    bool bits(uint32_t n, uint32_t& out) {
        out = 0;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t b;
            if (!bit(b)) return false;
            out = out << 1 | b;
        }
        return true;
    }
};

std::optional<size_t> next_frame(const uint8_t* data, size_t len, size_t start, size_t end) {
    size_t index = start / PACKET_BITS;
    auto header = packet(data, len, index);
    if (!header) return std::nullopt;
    size_t at = index * PACKET_BITS + HEADER_BITS + header->first_frame_bit;
    uint32_t number = 0;
    while (at < start) {
        Cursor c{data, len, at};
        uint32_t bits;
        if (!c.bits(LENGTH_BITS, bits) || bits <= LENGTH_BITS) return std::nullopt;
        at += bits;
        ++number;
    }
    bool last = number + 1 >= header->frames;
    if (end % PACKET_BITS == 0 || (last && end / PACKET_BITS == index)) {
        auto p = packet(data, len, (end - 1) / PACKET_BITS);
        if (!p) return std::nullopt;
        size_t next = (end - 1) / PACKET_BITS + 1 + p->skip;
        auto h = packet(data, len, next);
        if (!h) return std::nullopt;
        end = next * PACKET_BITS + HEADER_BITS + h->first_frame_bit;
    }
    if (end / 8 < len) return end;
    return std::nullopt;
}

// A prefix code built like FFmpeg's ff_vlc_init_from_lengths.
struct Vlc {
    struct Code {
        uint32_t code;
        uint8_t len;
        int32_t sym;
    };
    std::vector<Code> codes;
    void add(int32_t sym, uint8_t len, uint64_t& code) {
        codes.push_back({uint32_t(code), len, sym});
        code += 1ull << (32 - len);
    }
};

struct Codes {
    Vlc scale, scale_rl, coef[2], vec4, vec2, vec1;
};
const Codes& codes() {
    static Codes c = [] {
        Codes c;
        auto pairs = [](Vlc& v, const Pair* t, size_t n, int32_t off) {
            uint64_t code = 0;
            for (size_t i = 0; i < n; ++i) v.add(int32_t(t[i].sym) + off, t[i].len, code);
        };
        pairs(c.scale, SCALE, std::size(SCALE), -60);
        pairs(c.scale_rl, SCALE_RL, std::size(SCALE_RL), 0);
        {
            uint64_t code = 0;
            for (size_t i = 0; i < std::size(COEF0_SYMS); ++i) c.coef[0].add(COEF0_SYMS[i], COEF0_LENS[i], code);
        }
        pairs(c.coef[1], COEF1, std::size(COEF1), 0);
        {
            uint64_t code = 0;
            for (size_t i = 0; i < std::size(VEC4_SYMS); ++i) c.vec4.add(int32_t(VEC4_SYMS[i]) - 1, VEC4_LENS[i], code);
        }
        pairs(c.vec2, VEC2, std::size(VEC2), -1);
        pairs(c.vec1, VEC1, std::size(VEC1), 0);
        return c;
    }();
    return c;
}

uint32_t ilog2(size_t v) {
    uint32_t r = 0;
    v = v ? v : 1;
    while (v >>= 1) ++r;
    return r;
}

// Half-length inverse MDCT as FFmpeg defines it: n coefficients -> the middle n samples, scaled.
void fft(double* re, double* im, size_t n) {
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) j ^= bit, bit >>= 1;
        j |= bit;
        if (i < j) std::swap(re[i], re[j]), std::swap(im[i], im[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        double ws = std::sin(-2.0 * 3.14159265358979323846 / double(len)), wc = std::cos(-2.0 * 3.14159265358979323846 / double(len));
        for (size_t s = 0; s < n; s += len) {
            double cr = 1, ci = 0;
            for (size_t k = 0; k < len / 2; ++k) {
                size_t a = s + k, b = s + k + len / 2;
                double tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
                re[b] = re[a] - tr, im[b] = im[a] - ti;
                re[a] += tr, im[a] += ti;
                double ncr = cr * wc - ci * ws;
                ci = cr * ws + ci * wc;
                cr = ncr;
            }
        }
    }
}
void imdct_half(const float* in, size_t n, float* out, float scale) {
    size_t m = n / 2;
    double re[FRAME_SAMPLES / 2], im[FRAME_SAMPLES / 2];
    const double PI = 3.14159265358979323846;
    for (size_t k = 0; k < m; ++k) {
        double a = in[2 * k], b = in[n - 1 - 2 * k];
        double ang = -PI * double(k) / double(n);
        double s = std::sin(ang), c = std::cos(ang);
        re[k] = a * c - b * s, im[k] = a * s + b * c;
    }
    fft(re, im, m);
    for (size_t k = 0; k < m; ++k) {
        double ang = -PI * double(4 * k + 1) / double(4 * n);
        double s = std::sin(ang), c = std::cos(ang);
        double wr = re[k] * c - im[k] * s, wi = re[k] * s + im[k] * c;
        out[n - 1 - 2 * k] = float(wr) * scale;
        out[2 * k] = -float(wi) * scale;
    }
}

const std::vector<float>& window(uint32_t log2len) {
    static std::vector<float> w[3];
    static std::once_flag once;
    std::call_once(once, [] {
        for (int i = 0; i < 3; ++i) {
            size_t n = size_t(128) << i;
            w[i].resize(n);
            for (size_t k = 0; k < n; ++k) w[i][k] = float(std::sin((double(k) + 0.5) * 3.14159265358979323846 / (2.0 * double(n))));
        }
    });
    return w[log2len - 7];
}

// FFmpeg's vector_fmul_window in place.
void window_overlap(float* block, size_t winlen, const std::vector<float>& w) {
    size_t len = winlen / 2;
    for (size_t k = 0; k < len; ++k) {
        size_t i = k, j = 2 * len - 1 - k;
        float s0 = block[i], s1 = block[j], wi = w[i], wj = w[j];
        block[i] = s0 * wj - s1 * wi;
        block[j] = s0 * wi + s1 * wj;
    }
}

int16_t to_i16(float v) {
    float s = std::round(v * 32768.0f);
    return int16_t(std::clamp(s, -32768.0f, 32767.0f));
}

}  // namespace

std::optional<Packet> packet(const uint8_t* data, size_t len, size_t index) {
    size_t start = index * PACKET_BYTES;
    if (start + 4 > len) return std::nullopt;
    uint32_t h = uint32_t(data[start]) << 24 | uint32_t(data[start + 1]) << 16 | uint32_t(data[start + 2]) << 8 | data[start + 3];
    return Packet{h >> 26, (h >> 11) & 0x7fff, h & 0xff};
}

std::optional<Frame> read_frame(const uint8_t* data, size_t len, size_t start) {
    Cursor c{data, len, start};
    uint32_t bits;
    if (!c.bits(LENGTH_BITS, bits) || bits <= LENGTH_BITS || bits == 0x7fff) return std::nullopt;
    Frame f;
    f.data.assign((bits + 7) / 8, 0);
    f.bits = bits;
    f.start = start;
    Cursor r{data, len, start};
    for (size_t i = 0; i < bits; ++i) {
        uint32_t b;
        if (!r.bit(b)) return std::nullopt;
        f.data[i / 8] |= uint8_t(b << (7 - i % 8));
    }
    f.next = next_frame(data, len, start, r.pos);
    return f;
}

// ------------------------------------------------------------ decoder ----

struct Decoder::Bits {
    const uint8_t* data;
    size_t size, pos, end;
    uint32_t peek32() const {
        uint64_t v = 0;
        size_t byte = pos / 8;
        for (size_t i = 0; i < 5; ++i) v = v << 8 | (byte + i < size ? data[byte + i] : 0);
        return uint32_t((v << (24 + pos % 8)) >> 32);
    }
    uint32_t read(uint32_t n) {
        if (!n) return 0;
        uint32_t v = peek32() >> (32 - n);
        pos += n;
        return v;
    }
    bool bit() { return read(1) != 0; }
    int32_t sread(uint32_t n) { return int32_t(read(n) << (32 - n)) >> (32 - n); }
    bool overread() const { return pos > end; }
    int32_t vlc(const Vlc& v) {
        uint32_t p = peek32();
        auto it = std::upper_bound(v.codes.begin(), v.codes.end(), p, [](uint32_t x, const Vlc::Code& c) { return x < c.code; });
        const auto& c = *(it - 1);
        pos += c.len;
        return c.sym;
    }
    uint32_t large_value() {
        uint32_t n = 8;
        if (bit()) {
            n += 8;
            if (bit()) {
                n += 8;
                if (bit()) n += 7;
            }
        }
        return read(n);
    }
};

Decoder::Decoder(size_t channels, uint32_t sample_rate) {
    channels_ = std::clamp<size_t>(channels, 1, 2);
    uint32_t rate = sample_rate > 44100 ? 48000 : sample_rate > 32000 ? 44100 : sample_rate > 24000 ? 32000 : 24000;
    for (size_t i = 0; i < BLOCK_SIZES; ++i) {
        size_t sub = FRAME_SAMPLES >> i;
        size_t* off = sfb_offsets_[i];
        size_t band = 1;
        for (size_t f = 0; f < MAX_BANDS - 1 && f < std::size(CRITICAL_FREQ); ++f) {
            if (off[band - 1] >= sub) break;
            size_t o = ((sub * 2 * CRITICAL_FREQ[f]) / rate + 2) & ~size_t(3);
            if (o > off[band - 1]) off[band++] = o;
            if (o >= sub) break;
        }
        off[band - 1] = sub;
        num_sfb_[i] = band - 1;
    }
    for (size_t i = 0; i < BLOCK_SIZES; ++i)
        for (size_t b = 0; b < num_sfb_[i]; ++b) {
            size_t o = ((sfb_offsets_[i][b] + sfb_offsets_[i][b + 1] - 1) << i) >> 1;
            for (size_t x = 0; x < BLOCK_SIZES; ++x) {
                size_t v = 0;
                while (v + 1 < MAX_BANDS && (sfb_offsets_[x][v + 1] << x) < o) ++v;
                sf_offsets_[i][x][b] = v;
            }
        }
}

void Decoder::reset() { ch_[0] = Channel(), ch_[1] = Channel(); }

bool Decoder::decode(const uint8_t* data, size_t bits, std::vector<int16_t>& out) {
    Bits b{data, (bits + 7) / 8, 0, bits};
    size_t len = b.read(LENGTH_BITS);
    bool ok = frame(b, len);
    if (!ok)
        for (size_t c = 0; c < channels_; ++c) std::fill(std::begin(ch_[c].out), std::end(ch_[c].out), 0.0f);
    for (size_t i = 0; i < FRAME_SAMPLES; ++i)
        for (size_t c = 0; c < channels_; ++c) out.push_back(to_i16(ch_[c].out[i]));
    for (size_t c = 0; c < channels_; ++c) memmove(ch_[c].out, ch_[c].out + FRAME_SAMPLES, sizeof(float) * FRAME_SAMPLES / 2);
    return ok;
}

void Decoder::flush(std::vector<int16_t>& out) {
    for (size_t i = 0; i < FRAME_SAMPLES / 2; ++i)
        for (size_t c = 0; c < channels_; ++c) out.push_back(to_i16(ch_[c].out[i]));
    reset();
}

bool Decoder::frame(Bits& b, size_t len) {
    if (!tile_header(b)) return false;
    if (channels_ > 1 && b.bit() && b.bit()) b.read(uint32_t(channels_ * channels_ * 4));
    b.read(8);  // dynamic range compression gain
    if (b.bit()) {
        if (b.bit()) b.read(ilog2(FRAME_SAMPLES * 2));
        if (b.bit()) b.read(ilog2(FRAME_SAMPLES * 2));
    }
    parsed_all_subframes_ = false;
    for (size_t c = 0; c < channels_; ++c) ch_[c].decoded_samples = 0, ch_[c].cur_subframe = 0, ch_[c].reuse_sf = false;
    while (!parsed_all_subframes_)
        if (!subframe(b)) return false;
    return !(b.overread() || b.pos > len);
}

bool Decoder::subframe_length(Bits& b, size_t offset, size_t& len) {
    if (offset == FRAME_SAMPLES - MIN_SUBFRAME) return len = MIN_SUBFRAME, true;
    uint32_t shift = b.bit() ? 1 + b.read(1) : 0;
    len = FRAME_SAMPLES >> shift;
    return len >= MIN_SUBFRAME;
}

bool Decoder::tile_header(Bits& b) {
    size_t num_samples[2] = {}, min_len = 0, cur = channels_;
    bool contains[2] = {};
    for (size_t c = 0; c < channels_; ++c) ch_[c].num_subframes = 0;
    bool fixed = b.bit();
    for (;;) {
        for (size_t c = 0; c < channels_; ++c)
            contains[c] = num_samples[c] == min_len && (fixed || cur == 1 || min_len == FRAME_SAMPLES - MIN_SUBFRAME || b.bit());
        size_t sub;
        if (!subframe_length(b, min_len, sub)) return false;
        min_len += sub;
        for (size_t c = 0; c < channels_; ++c) {
            auto& ch = ch_[c];
            if (contains[c]) {
                if (ch.num_subframes >= MAX_SUBFRAMES) return false;
                ch.subframe_len[ch.num_subframes++] = sub;
                num_samples[c] += sub;
                if (num_samples[c] > FRAME_SAMPLES) return false;
            } else if (num_samples[c] <= min_len) {
                if (num_samples[c] < min_len) cur = 0, min_len = num_samples[c];
                ++cur;
            }
        }
        if (min_len >= FRAME_SAMPLES || b.overread()) break;
    }
    return !b.overread();
}

bool Decoder::channel_transform(Bits& b) {
    transform_ = false;
    if (channels_ == 1) return true;
    if (b.bit()) return false;
    if (channels_for_cur_subframe_ == 2) {
        if (b.bit()) {
            if (b.bit()) return false;
        } else {
            transform_ = true;
        }
    }
    if (transform_) {
        if (!b.bit()) {
            for (size_t i = 0; i < num_bands_; ++i) transform_band_[i] = b.bit();
        } else {
            std::fill(std::begin(transform_band_), std::end(transform_band_), true);
        }
    }
    return true;
}

bool Decoder::coefficients(Bits& b, size_t c) {
    const Codes& k = codes();
    size_t table = b.bit() ? 1 : 0;
    const uint16_t* run = table ? COEF1_RUN : COEF0_RUN;
    const float* level = table ? COEF1_LEVEL : COEF0_LEVEL;
    auto& ch = ch_[c];
    size_t sub = subframe_len_;
    bool rl_mode = false;
    size_t cur = 0, zeros = 0;
    while ((transmit_num_vec_coeffs_ || !rl_mode) && cur + 3 < ch.num_vec_coeffs) {
        uint32_t vals[4] = {};
        int32_t idx = b.vlc(k.vec4);
        if (idx < 0) {
            for (int i = 0; i < 4; i += 2) {
                int32_t i2 = b.vlc(k.vec2);
                if (i2 < 0) {
                    for (int j = i; j < i + 2; ++j) {
                        uint32_t x = uint32_t(b.vlc(k.vec1));
                        if (x == std::size(VEC1) - 1) x += b.large_value();
                        vals[j] = x;
                    }
                } else {
                    vals[i] = uint32_t(i2 >> 4), vals[i + 1] = uint32_t(i2 & 0xf);
                }
            }
        } else {
            vals[0] = uint32_t(idx >> 12), vals[1] = uint32_t((idx >> 8) & 0xf), vals[2] = uint32_t((idx >> 4) & 0xf), vals[3] = uint32_t(idx & 0xf);
        }
        for (uint32_t v : vals) {
            if (v) {
                bool neg = !b.bit();
                ch.coeffs[cur] = neg ? -float(v) : float(v);
                zeros = 0;
            } else {
                ch.coeffs[cur] = 0;
                ++zeros;
                rl_mode |= zeros > (sub >> 8);
            }
            ++cur;
        }
        if (b.overread()) return false;
    }
    if (cur < sub) {
        std::fill(ch.coeffs + cur, ch.coeffs + sub, 0.0f);
        size_t mask = sub - 1, off = cur;
        while (off < sub) {
            int32_t code = b.vlc(k.coef[table]);
            if (code > 1) {
                off += run[code];
                bool neg = !b.bit();
                ch.coeffs[off & mask] = neg ? -level[code] : level[code];
            } else if (code == 1) {
                break;
            } else {
                int32_t value = int32_t(b.large_value());
                if (b.bit()) {
                    if (b.bit()) {
                        if (b.bit()) break;
                        off += b.read(esc_len_) + 4;
                    } else {
                        off += b.read(2) + 1;
                    }
                }
                bool neg = !b.bit();
                ch.coeffs[off & mask] = float(neg ? -value : value);
            }
            if (b.overread()) return false;
            ++off;
        }
    }
    return true;
}

bool Decoder::scale_factors(Bits& b) {
    const Codes& k = codes();
    for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
        size_t c = channel_indexes_[i];
        const size_t* sfo = sf_offsets_[table_idx_][ch_[c].table_idx];
        auto& ch = ch_[c];
        size_t target = 1 - ch.scale_factor_idx;
        ch.scale_factors = target;
        if (ch.reuse_sf) {
            int32_t from[MAX_BANDS];
            memcpy(from, ch.saved_scale_factors[ch.scale_factor_idx], sizeof from);
            for (size_t band = 0; band < num_bands_; ++band) ch.saved_scale_factors[target][band] = from[sfo[band]];
        }
        if (ch.cur_subframe == 0 || b.bit()) {
            if (!ch.reuse_sf) {
                ch.scale_factor_step = int32_t(b.read(2)) + 1;
                int32_t val = 45 / ch.scale_factor_step;
                for (size_t band = 0; band < num_bands_; ++band) {
                    val += b.vlc(k.scale);
                    ch.saved_scale_factors[target][band] = val;
                }
            } else {
                size_t band = 0;
                while (band < num_bands_) {
                    int32_t idx = b.vlc(k.scale_rl);
                    uint32_t skip;
                    int32_t val;
                    bool neg;
                    if (idx == 0) {
                        uint32_t code = b.read(14);
                        skip = (code & 0x3f) >> 1, val = int32_t(code >> 6), neg = (code & 1) == 0;
                    } else if (idx == 1) {
                        break;
                    } else {
                        skip = SCALE_RL_RUN[idx], val = SCALE_RL_LEVEL[idx], neg = !b.bit();
                    }
                    band += skip;
                    if (band >= num_bands_) return false;
                    ch.saved_scale_factors[target][band] += neg ? -val : val;
                    ++band;
                }
            }
            ch.scale_factor_idx = target;
            ch.table_idx = table_idx_;
            ch.reuse_sf = true;
        }
        int32_t mx = INT32_MIN;
        for (size_t band = 0; band < num_bands_; ++band) mx = std::max(mx, ch.saved_scale_factors[ch.scale_factors][band]);
        ch.max_scale_factor = num_bands_ ? mx : 0;
    }
    return true;
}

bool Decoder::subframe(Bits& b) {
    size_t offset = FRAME_SAMPLES, sub = FRAME_SAMPLES;
    size_t total = FRAME_SAMPLES * channels_;
    for (size_t c = 0; c < channels_; ++c)
        if (offset > ch_[c].decoded_samples) {
            offset = ch_[c].decoded_samples;
            sub = ch_[c].subframe_len[std::min(ch_[c].cur_subframe, MAX_SUBFRAMES - 1)];
        }
    channels_for_cur_subframe_ = 0;
    for (size_t c = 0; c < channels_; ++c) {
        auto& ch = ch_[c];
        total -= ch.decoded_samples;
        if (offset == ch.decoded_samples && ch.cur_subframe < ch.num_subframes && sub == ch.subframe_len[ch.cur_subframe]) {
            total -= sub;
            ch.decoded_samples += sub;
            channel_indexes_[channels_for_cur_subframe_++] = c;
        }
    }
    if (!channels_for_cur_subframe_) return false;
    if (!total) parsed_all_subframes_ = true;
    table_idx_ = ilog2(FRAME_SAMPLES / sub);
    num_bands_ = num_sfb_[table_idx_];
    subframe_len_ = sub;
    esc_len_ = ilog2(sub - 1) + 1;
    if (b.bit()) {
        size_t fill = b.read(2);
        if (!fill) {
            uint32_t l = b.read(4);
            fill = b.read(l) + 1;
        }
        b.pos += fill;
    }
    if (b.bit()) return false;
    if (!channel_transform(b)) return false;
    bool transmit = false;
    for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
        auto& ch = ch_[channel_indexes_[i]];
        ch.transmit_coefs = b.bit();
        transmit |= ch.transmit_coefs;
    }
    if (transmit) {
        transmit_num_vec_coeffs_ = b.bit();
        if (transmit_num_vec_coeffs_) {
            uint32_t nb = ilog2((sub + 3) / 4) + 1;
            for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
                size_t n = size_t(b.read(nb)) << 2;
                if (n > sub) return false;
                ch_[channel_indexes_[i]].num_vec_coeffs = n;
            }
        } else {
            for (size_t i = 0; i < channels_for_cur_subframe_; ++i) ch_[channel_indexes_[i]].num_vec_coeffs = sub;
        }
        int32_t quant = 90;
        int32_t step = b.sread(6);
        quant += step;
        if (step == -32 || step == 31) {
            bool neg = step == -32;
            int32_t q = 0;
            while (b.pos + 5 < b.end) {
                step = int32_t(b.read(5));
                if (step != 31) break;
                q += 31;
            }
            quant += neg ? -(q + step) : (q + step);
        }
        if (channels_for_cur_subframe_ == 1) {
            ch_[channel_indexes_[0]].quant_step = quant;
        } else {
            uint32_t ml = b.read(3);
            for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
                auto& ch = ch_[channel_indexes_[i]];
                ch.quant_step = quant;
                if (b.bit()) ch.quant_step += ml ? int32_t(b.read(ml)) + 1 : 1;
            }
        }
        if (!scale_factors(b)) return false;
    }
    for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
        size_t c = channel_indexes_[i];
        if (ch_[c].transmit_coefs && b.pos < b.end) {
            if (!coefficients(b, c)) return false;
        } else {
            std::fill(ch_[c].coeffs, ch_[c].coeffs + sub, 0.0f);
        }
    }
    size_t at = offset + FRAME_SAMPLES / 2;
    if (transmit) {
        inverse_channel_transform();
        float scale = 2.0f / float(sub) / 32768.0f;
        const size_t* sfb = sfb_offsets_[table_idx_];
        float tmp[FRAME_SAMPLES];
        for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
            auto& ch = ch_[channel_indexes_[i]];
            const int32_t* sf = ch.saved_scale_factors[ch.scale_factors];
            for (size_t band = 0; band < num_bands_; ++band) {
                size_t end = std::min(sfb[band + 1], sub);
                int32_t exp = ch.quant_step - (ch.max_scale_factor - sf[band]) * ch.scale_factor_step;
                float q = std::pow(10.0f, float(exp) / 20.0f);
                for (size_t y = sfb[band]; y < end; ++y) tmp[y] = ch.coeffs[y] * q;
            }
            imdct_half(tmp, sub, ch.out + at, scale);
        }
    } else {
        for (size_t i = 0; i < channels_for_cur_subframe_; ++i) std::fill(ch_[channel_indexes_[i]].out + at, ch_[channel_indexes_[i]].out + at + sub, 0.0f);
    }
    for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
        auto& ch = ch_[channel_indexes_[i]];
        size_t winlen = ch.prev_block_len, start = at - winlen / 2;
        if (sub < winlen) start += (winlen - sub) / 2, winlen = sub;
        window_overlap(ch.out + start, winlen, window(ilog2(winlen)));
        ch.prev_block_len = sub;
    }
    for (size_t i = 0; i < channels_for_cur_subframe_; ++i) {
        auto& ch = ch_[channel_indexes_[i]];
        if (ch.cur_subframe >= ch.num_subframes) return false;
        ++ch.cur_subframe;
    }
    return !b.overread();
}

void Decoder::inverse_channel_transform() {
    if (!transform_ || channels_for_cur_subframe_ != 2) return;
    const size_t* sfb = sfb_offsets_[table_idx_];
    auto& a = ch_[0];
    auto& bb = ch_[1];
    for (size_t band = 0; band < num_bands_; ++band) {
        size_t end = std::min(sfb[band + 1], subframe_len_);
        for (size_t y = sfb[band]; y < end; ++y) {
            if (transform_band_[band]) {
                float l = a.coeffs[y], r = bb.coeffs[y];
                a.coeffs[y] = l - r, bb.coeffs[y] = l + r;
            } else {
                a.coeffs[y] *= 181.0f / 128.0f, bb.coeffs[y] *= 181.0f / 128.0f;
            }
        }
    }
}

}  // namespace kl::xma
