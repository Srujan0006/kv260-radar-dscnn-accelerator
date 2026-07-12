#ifndef RADAR_CNN_STREAM_LAYERS_H
#define RADAR_CNN_STREAM_LAYERS_H

// Templated streaming layer library for the V2 accelerator. Every layer
// consumes and produces hls::stream<Pixel<C>> tokens (pixel-major channel
// vectors) so the whole network runs as one DATAFLOW pipeline with no
// intermediate feature map ever stored. Only the 3x3 depthwise stages use
// line buffers (2 rows); pointwise, pool and GAP stages are pure
// token-at-a-time operators.
//
// Numerical contract: every stage reproduces the validated V1
// frame-buffered operation order (accumulate in acc_t starting from the
// bias, in ascending input-channel / kernel order, cast to data_t, then
// ReLU), so results are directly comparable against the V1 reference.

#include "radar_cnn_stream.h"
#include "stream_types.h"
#include "hls_stream_compat.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

static inline data_t relu_stream(data_t value) {
#pragma HLS INLINE
    return value > (data_t)0 ? value : (data_t)0;
}

// Streaming 3x3 depthwise convolution, stride 1, zero padding 1 ("same").
// Per-channel generalization of the validated streaming stem: a 2-row
// line buffer plus a 3x3 window per channel, with one extra flush
// row/column so the bottom/right padded outputs are emitted. Consumes
// H*W tokens, emits H*W tokens.
//
// Loop structure (2026-07 optimization round): a single PERFECT loop nest
// (py, px, cb) with PIPELINE II=1 on the innermost channel-block loop, so
// Vitis flattens the whole thing into ONE pipeline of
// (H+1)*(W+1)*(C/UF) iterations -- there is no per-pixel pipeline
// flush any more (the old form left the pixel loops unpipelined, costing
// 228-781 cycles/pixel on DW2-DW4).
//
// State (window, line buffer, weights, biases) is banked CYCLIC by UF on
// the channel dimension instead of COMPLETE: iteration cb touches only
// channels cb*UF+u, so each of the UF lanes reads/writes exactly one
// element per bank per cycle at address cb -- plain memory addressing
// instead of the C:1 channel-select multiplexers that made the old
// complete-partitioned form explode to ~57k LUTs on DW4.
//
// UF channels are convolved in parallel per cycle (UF * 9 multipliers);
// one pixel takes C/UF cycles. UF is the DSP/latency knob, must divide C.
// With UF == C (DW1) the banks collapse to registers and this degenerates
// to the stem's fully-parallel structure.
//
// Note on II: the per-channel window banks carry a true RAW recurrence of
// distance C/UF iterations (write at block cb, read again for the next
// pixel). For C/UF >= 2 that distance exceeds the read/write stage gap;
// if a synthesis report still shows II=2 here, add
//   #pragma HLS DEPENDENCE variable=window inter distance=<C/UF> true
// inside BLOCKS. Even II=2 leaves every DW stage far off the critical
// path (see README), so correctness is never at stake.
template<int C, int H, int W, int UF>
void depthwise_conv_stream(
    hls::stream<Pixel<C> > &in_pixels,
    hls::stream<Pixel<C> > &out_pixels,
    const data_t (&weights)[C][1][3][3],
    const data_t (&biases)[C]
) {
#pragma HLS INLINE off
    static_assert(C % UF == 0, "UF must divide C");
    constexpr int CB_COUNT = C / UF;

    // 2 rows x W columns x C channels. Row parity banks are separate
    // arrays; the channel dim is banked by lane (address = px*CB + cb).
    // Depths are tiny (<= W * C/UF <= 200 words), so LUTRAM, not BRAM.
    data_t line_buffer[2][W][C];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer cyclic factor=UF dim=3
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=lutram
    data_t window[C][3][3];
#pragma HLS ARRAY_PARTITION variable=window cyclic factor=UF dim=1
#pragma HLS ARRAY_PARTITION variable=window complete dim=2
#pragma HLS ARRAY_PARTITION variable=window complete dim=3
    // Weight ROM: UF*9 banks of depth C/UF (registers/LUTRAM) -- see the
    // README for why these must NOT be pushed to URAM (UltraScale+ URAM
    // cannot be initialized from the bitstream, so const ROMs would come
    // up all-zero on the board).
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=UF dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights complete dim=4
#pragma HLS ARRAY_PARTITION variable=biases cyclic factor=UF dim=1

    // Persist across BLOCKS iterations: the token read at cb==0 and the
    // output assembled block by block until cb==CB_COUNT-1.
    Pixel<C> incoming;
    Pixel<C> output;

    int read_count = 0;
    int write_count = 0;

    ROWS: for (int py = 0; py <= H; ++py) {
        COLS: for (int px = 0; px <= W; ++px) {
            BLOCKS: for (int cb = 0; cb < CB_COUNT; ++cb) {
#pragma HLS PIPELINE II=1
                // The value written to line_buffer at (py, px, c) is next
                // read one full row later ((W+1)*CB_COUNT iterations),
                // far beyond the pipeline depth, so the conservative
                // inter-iteration RAW hazard can be waived.
#pragma HLS DEPENDENCE variable=line_buffer inter false
                const bool have_input = (py < H && px < W);
                if (cb == 0 && have_input) {
                    incoming = in_pixels.read();
                    ++read_count;
                }

                LANES: for (int u = 0; u < UF; ++u) {
#pragma HLS UNROLL
                    const int c = cb * UF + u;
                    // Shift the window left. At the row start the old
                    // window is the cleared one (all zeros), exactly as
                    // in the previous CLEAR-then-SHIFT formulation.
                    data_t l0, l1, l2, m0, m1, m2;
                    if (px == 0) {
                        l0 = l1 = l2 = (data_t)0;
                        m0 = m1 = m2 = (data_t)0;
                    } else {
                        l0 = window[c][0][1];
                        l1 = window[c][1][1];
                        l2 = window[c][2][1];
                        m0 = window[c][0][2];
                        m1 = window[c][1][2];
                        m2 = window[c][2][2];
                    }
                    const data_t r0 = (py >= 2 && px < W)
                        ? line_buffer[py % 2][px][c] : (data_t)0;
                    const data_t r1 = (py >= 1 && px < W)
                        ? line_buffer[(py + 1) % 2][px][c] : (data_t)0;
                    const data_t r2 = have_input
                        ? incoming.channel[c] : (data_t)0;
                    window[c][0][0] = l0; window[c][0][1] = m0; window[c][0][2] = r0;
                    window[c][1][0] = l1; window[c][1][1] = m1; window[c][1][2] = r1;
                    window[c][2][0] = l2; window[c][2][1] = m2; window[c][2][2] = r2;
                    if (have_input) {
                        line_buffer[py % 2][px][c] = r2;
                    }

                    if (py >= 1 && px >= 1) {
                        // V1 numerical contract: bias first, then the 9
                        // taps in ascending (ky, kx) order.
                        acc_t acc = biases[c];
                        acc += (acc_t)l0 * (acc_t)weights[c][0][0][0];
                        acc += (acc_t)m0 * (acc_t)weights[c][0][0][1];
                        acc += (acc_t)r0 * (acc_t)weights[c][0][0][2];
                        acc += (acc_t)l1 * (acc_t)weights[c][0][1][0];
                        acc += (acc_t)m1 * (acc_t)weights[c][0][1][1];
                        acc += (acc_t)r1 * (acc_t)weights[c][0][1][2];
                        acc += (acc_t)l2 * (acc_t)weights[c][0][2][0];
                        acc += (acc_t)m2 * (acc_t)weights[c][0][2][1];
                        acc += (acc_t)r2 * (acc_t)weights[c][0][2][2];
                        output.channel[c] = relu_stream((data_t)acc);
                    }
                }

                if (py >= 1 && px >= 1 && cb == CB_COUNT - 1) {
                    out_pixels.write(output);
                    ++write_count;
                }
            }
        }
    }
#ifndef __SYNTHESIS__
    assert(read_count == H * W);
    assert(write_count == H * W);
#endif
}

// Streaming 1x1 (pointwise) convolution: one Pixel<CIN> token in, one
// Pixel<COUT> token out, no feature-map storage of any kind. The weight
// matrix is walked in UF_OUT x UF_IN tiles by a single flat pipelined
// loop ((COUT/UF_OUT) * (CIN/UF_IN) iterations per pixel at II=1, using
// UF_OUT*UF_IN multipliers). Accumulation is chained from the bias in
// ascending input-channel order, so the result is numerically identical
// to the V1 linear loop (bit-exact in ap_fixed, same rounding order in
// float).
template<int CIN, int COUT, int TOKENS, int UF_IN, int UF_OUT, bool RELU>
void pointwise_conv_stream(
    hls::stream<Pixel<CIN> > &in_pixels,
    hls::stream<Pixel<COUT> > &out_pixels,
    const data_t (&weights)[COUT][CIN],
    const data_t (&biases)[COUT]
) {
#pragma HLS INLINE off
    static_assert(CIN % UF_IN == 0, "UF_IN must divide CIN");
    static_assert(COUT % UF_OUT == 0, "UF_OUT must divide COUT");
    // Weight ROM banked to exactly match the UF_OUT x UF_IN access tile:
    // iteration (ob, ib) reads weights[ob*UF_OUT+uo][ib*UF_IN+ui], so with
    // cyclic partitioning each of the UF_OUT*UF_IN banks serves one read
    // per cycle at address (ob, ib) -- no channel-select multiplexers and
    // no over-partitioning (the old `complete dim=2` made CIN banks of
    // depth COUT, i.e. up to 96 half-empty BRAMs for PW4). Banks stay in
    // LUTRAM/BRAM; URAM is deliberately NOT used because UltraScale+ URAM
    // cannot be initialized from the bitstream (const ROMs would be zero).
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=UF_OUT dim=1
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=UF_IN dim=2
#pragma HLS ARRAY_PARTITION variable=biases cyclic factor=UF_OUT dim=1

    constexpr int IB_COUNT = CIN / UF_IN;
    constexpr int OB_COUNT = COUT / UF_OUT;

    int read_count = 0;
    int write_count = 0;

    TOKEN: for (int t = 0; t < TOKENS; ++t) {
        Pixel<CIN> in_vec = in_pixels.read();
        ++read_count;
        data_t in_local[CIN];
#pragma HLS ARRAY_PARTITION variable=in_local complete dim=1
        LOAD: for (int ic = 0; ic < CIN; ++ic) {
#pragma HLS UNROLL
            in_local[ic] = in_vec.channel[ic];
        }

        Pixel<COUT> out_vec;
        acc_t acc[UF_OUT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

        int ob = 0;
        int ib = 0;
        MAC_FLAT: for (int i = 0; i < OB_COUNT * IB_COUNT; ++i) {
#pragma HLS PIPELINE II=1
            OUT_CH: for (int uo = 0; uo < UF_OUT; ++uo) {
#pragma HLS UNROLL
                const int oc = ob * UF_OUT + uo;
                acc_t sum = (ib == 0) ? (acc_t)biases[oc] : acc[uo];
                IN_CH: for (int ui = 0; ui < UF_IN; ++ui) {
#pragma HLS UNROLL
                    const int ic = ib * UF_IN + ui;
                    sum += (acc_t)in_local[ic] * (acc_t)weights[oc][ic];
                }
                acc[uo] = sum;
                if (ib == IB_COUNT - 1) {
                    out_vec.channel[oc] =
                        RELU ? relu_stream((data_t)sum) : (data_t)sum;
                }
            }
            if (ib == IB_COUNT - 1) {
                ib = 0;
                ++ob;
            } else {
                ++ib;
            }
        }
        out_pixels.write(out_vec);
        ++write_count;
    }
#ifndef __SYNTHESIS__
    assert(read_count == TOKENS);
    assert(write_count == TOKENS);
#endif
}

// Streaming 2x2 stride-2 max pool. Emits floor(H_IN/2) x floor(W_IN/2)
// tokens; for odd H_IN/W_IN the trailing row/column is consumed but never
// contributes, exactly like the V1 frame-buffered maxpool_2x2 (which
// never indexes it). One row of horizontal pair-maxima is buffered; no
// feature map is stored. II=1 per input token, all channels in parallel
// (comparators only, no DSPs).
template<int C, int H_IN, int W_IN>
void maxpool_stream(
    hls::stream<Pixel<C> > &in_pixels,
    hls::stream<Pixel<C> > &out_pixels
) {
#pragma HLS INLINE off
    constexpr int H_OUT = H_IN / 2;
    constexpr int W_OUT = W_IN / 2;

    data_t row_max[W_OUT][C];
#pragma HLS ARRAY_PARTITION variable=row_max complete dim=2
    data_t pair_max[C];
#pragma HLS ARRAY_PARTITION variable=pair_max complete dim=1

    int read_count = 0;
    int write_count = 0;

    ROWS: for (int py = 0; py < H_IN; ++py) {
        COLS: for (int px = 0; px < W_IN; ++px) {
#pragma HLS PIPELINE II=1
            Pixel<C> pix = in_pixels.read();
            ++read_count;
            const int ox = px / 2;
            const bool odd_col = (px % 2) == 1;
            const bool odd_row = (py % 2) == 1;
            const bool col_in_pair = ox < W_OUT;
            const bool row_in_pair = (py / 2) < H_OUT;

            Pixel<C> out_pix;
            CH: for (int c = 0; c < C; ++c) {
#pragma HLS UNROLL
                const data_t v = pix.channel[c];
                if (!odd_col) {
                    pair_max[c] = v;
                } else if (col_in_pair) {
                    const data_t h = (pair_max[c] > v) ? pair_max[c] : v;
                    if (!odd_row) {
                        row_max[ox][c] = h;
                    } else if (row_in_pair) {
                        const data_t r = row_max[ox][c];
                        out_pix.channel[c] = (r > h) ? r : h;
                    }
                }
            }
            if (odd_row && odd_col && col_in_pair && row_in_pair) {
                out_pixels.write(out_pix);
                ++write_count;
            }
        }
    }
#ifndef __SYNTHESIS__
    assert(read_count == H_IN * W_IN);
    assert(write_count == H_OUT * W_OUT);
#endif
}

// Streaming global average pool: accumulates C running sums over TOKENS
// pixel tokens, then emits ONE Pixel<C> holding the per-channel mean.
// Matches V1 exactly: accumulate in acc_t in arrival (row-major) order,
// divide by TOKENS, then cast to data_t. UF channels accumulate in
// parallel per cycle.
template<int C, int TOKENS, int UF>
void gap_stream(
    hls::stream<Pixel<C> > &in_pixels,
    hls::stream<Pixel<C> > &out_pixels
) {
#pragma HLS INLINE off
    static_assert(C % UF == 0, "UF must divide C");
    acc_t acc[C];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
    INIT: for (int c = 0; c < C; ++c) {
#pragma HLS UNROLL
        acc[c] = (acc_t)0;
    }
    TOKEN: for (int t = 0; t < TOKENS; ++t) {
        Pixel<C> pix = in_pixels.read();
        ACC_BLOCKS: for (int cb = 0; cb < C / UF; ++cb) {
#pragma HLS PIPELINE II=1
            ACC_CH: for (int u = 0; u < UF; ++u) {
#pragma HLS UNROLL
                const int c = cb * UF + u;
                acc[c] += (acc_t)pix.channel[c];
            }
        }
    }
    Pixel<C> out_pix;
    DIVIDE: for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE II=1
        out_pix.channel[c] = (data_t)(acc[c] / (acc_t)TOKENS);
    }
    out_pixels.write(out_pix);
}

// Final DDR writer: unpacks the single logits token.
inline void write_logits_stream(
    hls::stream<Pixel<NUM_CLASSES> > &logit_pixels,
    data_t *logits
) {
#pragma HLS INLINE off
    Pixel<NUM_CLASSES> pix = logit_pixels.read();
    WRITE: for (int i = 0; i < NUM_CLASSES; ++i) {
#pragma HLS PIPELINE II=1
        logits[i] = pix.channel[i];
    }
}

#endif
