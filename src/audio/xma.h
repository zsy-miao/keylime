// keylime: XMA2, the console's hardware audio codec: mono or stereo WMA Pro
// frames of 512 samples per channel, packed into 2048-byte packets.
//
// A packet starts with a 32-bit header (MSB first): frames that begin in it
// (6 bits), bits finishing the previous frame before the first new one (15),
// metadata (3), packets of other streams before this stream's next (8). Every
// frame starts with its length in bits (15).
//
// Ported from VIctory's xma.rs, whose frame decoder follows FFmpeg's
// wmaprodec.c (LGPL-2.1-or-later) restricted to what XMA needs.
#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kl::xma {

constexpr size_t PACKET_BYTES = 2048;
constexpr size_t FRAME_SAMPLES = 512;     // per channel
constexpr size_t SUBFRAME_SAMPLES = 128;  // unit of skips and loop ends

struct Packet {
    uint32_t frames, first_frame_bit, skip;
};
std::optional<Packet> packet(const uint8_t* data, size_t len, size_t index);

struct Frame {
    std::vector<uint8_t> data;  // from its length field on, MSB first
    size_t bits = 0;
    size_t start = 0;            // bit offset in the buffer
    std::optional<size_t> next;  // where the next frame starts, if any
};
// The frame starting at bit `start` of a buffer of whole packets of one stream.
std::optional<Frame> read_frame(const uint8_t* data, size_t len, size_t start);

class Decoder {
public:
    Decoder(size_t channels, uint32_t sample_rate);
    size_t channels() const { return channels_; }
    void reset();
    // Appends FRAME_SAMPLES interleaved samples per channel; false on a damaged frame (silence kept for timing).
    bool decode(const uint8_t* data, size_t bits, std::vector<int16_t>& out);
    // The overlap left after the last frame: half a frame per channel.
    void flush(std::vector<int16_t>& out);

private:
    static constexpr size_t BLOCK_SIZES = 3, MAX_BANDS = 29, MAX_SUBFRAMES = 32;
    struct Bits;
    struct Channel {
        size_t prev_block_len = FRAME_SAMPLES;
        bool transmit_coefs = false;
        size_t num_subframes = 0, subframe_len[MAX_SUBFRAMES] = {}, cur_subframe = 0, decoded_samples = 0;
        int32_t quant_step = 0;
        bool reuse_sf = false;
        int32_t scale_factor_step = 1, max_scale_factor = 0;
        int32_t saved_scale_factors[2][MAX_BANDS] = {};
        size_t scale_factor_idx = 0, scale_factors = 0, table_idx = 0, num_vec_coeffs = 0;
        float coeffs[FRAME_SAMPLES] = {};
        float out[FRAME_SAMPLES + FRAME_SAMPLES / 2] = {};
    };
    bool frame(Bits& b, size_t len);
    bool subframe_length(Bits& b, size_t offset, size_t& len);
    bool tile_header(Bits& b);
    bool channel_transform(Bits& b);
    bool coefficients(Bits& b, size_t c);
    bool scale_factors(Bits& b);
    bool subframe(Bits& b);
    void inverse_channel_transform();

    size_t channels_;
    size_t num_sfb_[BLOCK_SIZES] = {};
    size_t sfb_offsets_[BLOCK_SIZES][MAX_BANDS + 1] = {};
    size_t sf_offsets_[BLOCK_SIZES][BLOCK_SIZES][MAX_BANDS] = {};
    Channel ch_[2];
    size_t subframe_len_ = FRAME_SAMPLES, channels_for_cur_subframe_ = 0, channel_indexes_[2] = {0, 1};
    size_t num_bands_ = 0, table_idx_ = 0;
    uint32_t esc_len_ = 0;
    bool transmit_num_vec_coeffs_ = false, transform_ = false, transform_band_[MAX_BANDS] = {};
    bool parsed_all_subframes_ = false;
};

}  // namespace kl::xma
