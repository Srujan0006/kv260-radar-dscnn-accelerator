#include "radar_cnn_stream.h"
#include "stream_types.h"
#include "hls_stream_compat.h"
#include "stream_layers.h"
#include "weights.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

void read_input(const data_t *input, hls::stream<InputPixel> &pixels) {
#pragma HLS INLINE off
    int write_count = 0;
    for (int i = 0; i < INPUT_TOKENS; ++i) {
#pragma HLS PIPELINE II=1
        InputPixel pixel;
        pixel.channel[0] = input[i];
        pixels.write(pixel);
        ++write_count;
    }
#ifndef __SYNTHESIS__
    assert(write_count == INPUT_TOKENS);
#endif
}

void stem_conv_stream(
    hls::stream<InputPixel> &input_pixels,
    hls::stream<StemPixel> &stem_pixels
) {
#pragma HLS INLINE off
    // Store only the two previous rows. The current row lives in the window.
    data_t line_buffer[2][IMG_W];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
    data_t window[3][3];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0
    int read_count = 0;
    int write_count = 0;

    // The extra row and column flush the bottom and right zero padding.
    for (int py = 0; py <= IMG_H; ++py) {
        for (int px = 0; px <= IMG_W; ++px) {
#pragma HLS PIPELINE II=1
            if (px == 0) {
                for (int wy = 0; wy < 3; ++wy) {
                    for (int wx = 0; wx < 3; ++wx) {
                        window[wy][wx] = (data_t)0;
                    }
                }
            }

            data_t incoming = (data_t)0;
            if (py < IMG_H && px < IMG_W) {
                InputPixel input_pixel = input_pixels.read();
                incoming = input_pixel.channel[0];
                ++read_count;
            }

            for (int row = 0; row < 3; ++row) {
                window[row][0] = window[row][1];
                window[row][1] = window[row][2];
            }
            window[0][2] = (py >= 2 && px < IMG_W)
                ? line_buffer[py % 2][px] : (data_t)0;
            window[1][2] = (py >= 1 && px < IMG_W)
                ? line_buffer[(py + 1) % 2][px] : (data_t)0;
            window[2][2] = incoming;

            if (px < IMG_W) {
                line_buffer[py % 2][px] = incoming;
            }

            if (py >= 1 && px >= 1) {
                StemPixel output;

                for (int oc = 0; oc < C_STEM; ++oc) {
                    acc_t acc = B_STEM[oc];
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            acc += (acc_t)window[ky][kx] *
                                   (acc_t)W_STEM[oc][0][ky][kx];
                        }
                    }
                    output.channel[oc] = relu_stream((data_t)acc);
                }
                stem_pixels.write(output);
                ++write_count;
            }
        }
    }
#ifndef __SYNTHESIS__
    assert(read_count == INPUT_TOKENS);
    assert(write_count == STEM_TOKENS);
#endif
}

void capture_stem(
    hls::stream<StemPixel> &stem_pixels,
    data_t *stem_output
) {
#pragma HLS INLINE off
    int read_count = 0;
    for (int pixel_index = 0; pixel_index < STEM_TOKENS; ++pixel_index) {
        StemPixel pixel = stem_pixels.read();
        ++read_count;
        for (int channel = 0; channel < C_STEM; ++channel) {
#pragma HLS PIPELINE II=1
            stem_output[pixel_index * C_STEM + channel] = pixel.channel[channel];
        }
    }
#ifndef __SYNTHESIS__
    assert(read_count == STEM_TOKENS);
#endif
}

void capture_dw1(
    hls::stream<StemPixel> &dw1_pixels,
    data_t *dw1_output
) {
#pragma HLS INLINE off
    int read_count = 0;
    for (int pixel_index = 0; pixel_index < DW1_TOKENS; ++pixel_index) {
        StemPixel pixel = dw1_pixels.read();
        ++read_count;
        for (int channel = 0; channel < C_STEM; ++channel) {
#pragma HLS PIPELINE II=1
            dw1_output[pixel_index * C_STEM + channel] = pixel.channel[channel];
        }
    }
#ifndef __SYNTHESIS__
    assert(read_count == DW1_TOKENS);
#endif
}

extern "C" {
void radar_cnn_stream_stem(const data_t *input, data_t *stem_output) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=10000 max_widen_bitwidth=128
#pragma HLS INTERFACE m_axi port=stem_output offset=slave bundle=gmem1 depth=160000 max_widen_bitwidth=128
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=stem_output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW

    hls::stream<InputPixel> input_pixels;
    hls::stream<StemPixel> stem_pixels;
#pragma HLS STREAM variable=input_pixels depth=64
#pragma HLS STREAM variable=stem_pixels depth=4
#pragma HLS BIND_STORAGE variable=input_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=stem_pixels type=fifo impl=srl

    read_input(input, input_pixels);
    stem_conv_stream(input_pixels, stem_pixels);
    capture_stem(stem_pixels, stem_output);
}

// V2B milestone kernel: streaming stem followed by streaming DW1. Stem
// output never touches DDR -- it flows stem_pixels -> DW1 entirely inside
// the DATAFLOW region. Only the DW1 activations (H0 x W0 x C_STEM, since
// DW1 is stride 1 like the validated V1 reference) are captured to DDR
// for verification.
void radar_cnn_stream_dw1(const data_t *input, data_t *dw1_output) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=10000 max_widen_bitwidth=128
#pragma HLS INTERFACE m_axi port=dw1_output offset=slave bundle=gmem1 depth=160000 max_widen_bitwidth=128
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=dw1_output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW

    hls::stream<InputPixel> input_pixels;
    hls::stream<StemPixel> stem_pixels;
    hls::stream<StemPixel> dw1_pixels;
#pragma HLS STREAM variable=input_pixels depth=64
#pragma HLS STREAM variable=stem_pixels depth=4
#pragma HLS STREAM variable=dw1_pixels depth=4
#pragma HLS BIND_STORAGE variable=input_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=stem_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=dw1_pixels type=fifo impl=srl

    read_input(input, input_pixels);
    stem_conv_stream(input_pixels, stem_pixels);
    depthwise_conv_stream<C_STEM, H0, W0, C_STEM>(
        stem_pixels, dw1_pixels, W_DW1, B_DW1);
    capture_dw1(dw1_pixels, dw1_output);
}

// V2C milestone kernel: the COMPLETE streaming CNN. Every stage runs
// concurrently inside one DATAFLOW region; no intermediate feature map
// ever touches DDR or a full-frame buffer. Depthwise stages use 2-row
// line buffers; pointwise/pool/GAP stages are token-at-a-time operators.
//
// Depthwise UF (channels per cycle) and pointwise UF_IN x UF_OUT tile
// sizes are the DSP/throughput knobs. After the 2026-07 balancing round
// the budget is spent where the cycles are: the depthwise stages were
// over-provisioned (2k-10k cycle intervals against a 160k bottleneck,
// ~650 DSPs of dead parallelism), so their UF was cut to land all four
// at ~8k-22k cycles and the reclaimed DSPs widened every pointwise
// tile. Frame rate is set by PW1 and PW2 at 80k cycles/frame each
// (10,000 px x (16/8)*(32/8), resp. 2,500 px x (32/8)*(64/8)):
// ~533 us at 150 MHz, ~1,875 fps. Going wider (40k) prices out at
// ~1.3k DSPs on this 1,248-DSP part -- the design is DSP-limited here.
void radar_cnn_stream_full(const data_t *input, data_t *logits) {
    // Input widened to 128-bit beats (8 samples/beat) so the read burst
    // is 1,250 beats instead of 10,000. The logits port stays narrow on
    // purpose: 4 x 16-bit is a single beat either way.
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=10000 max_widen_bitwidth=128
#pragma HLS INTERFACE m_axi port=logits offset=slave bundle=gmem1 depth=4
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=logits bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW

    hls::stream<InputPixel> input_pixels;
    hls::stream<StemPixel> stem_pixels;
    hls::stream<StemPixel> dw1_pixels;
    hls::stream<B1Pixel> pw1_pixels;
    hls::stream<B1Pixel> pool1_pixels;
    hls::stream<B1Pixel> dw2_pixels;
    hls::stream<B2Pixel> pw2_pixels;
    hls::stream<B2Pixel> pool2_pixels;
    hls::stream<B2Pixel> dw3_pixels;
    hls::stream<B3Pixel> pw3_pixels;
    hls::stream<B3Pixel> pool3_pixels;
    hls::stream<B3Pixel> dw4_pixels;
    hls::stream<B4Pixel> pw4_pixels;
    hls::stream<B4Pixel> gap_pixels;
    hls::stream<LogitsPixel> logit_pixels;
    // FIFO sizing: the pipeline is a single linear chain (no forks or
    // reconvergence), so no depth beyond the depth-2 handshake minimum is
    // ever REQUIRED for deadlock freedom -- extra depth only smooths
    // transient rate jitter at row boundaries (a depthwise stage consumes
    // nothing during its flush column; a pool emits nothing on even
    // rows). Depth 4 covers that jitter. input_pixels is deeper (64
    // tokens = eight 128-bit AXI beats) purely to keep the widened DDR
    // burst moving while the stem sits in flush cycles.
    //
    // Every stream is forced to SRL/register FIFOs: tokens are up to
    // 2,048 bits wide (Pixel<128>), and a BRAM FIFO costs ceil(width/36)
    // BRAM36 REGARDLESS of depth -- 57 BRAMs for pw4/gap alone, ~660
    // across the pipeline in the baseline report, on a part with ~288.
    // At these depths SRL costs ~1 LUT/bit (~14k LUTs total), which the
    // depthwise rewrite pays for many times over.
#pragma HLS STREAM variable=input_pixels depth=64
#pragma HLS STREAM variable=stem_pixels depth=4
#pragma HLS STREAM variable=dw1_pixels depth=4
#pragma HLS STREAM variable=pw1_pixels depth=4
#pragma HLS STREAM variable=pool1_pixels depth=4
#pragma HLS STREAM variable=dw2_pixels depth=4
#pragma HLS STREAM variable=pw2_pixels depth=4
#pragma HLS STREAM variable=pool2_pixels depth=4
#pragma HLS STREAM variable=dw3_pixels depth=4
#pragma HLS STREAM variable=pw3_pixels depth=4
#pragma HLS STREAM variable=pool3_pixels depth=4
#pragma HLS STREAM variable=dw4_pixels depth=4
#pragma HLS STREAM variable=pw4_pixels depth=2
#pragma HLS STREAM variable=gap_pixels depth=2
#pragma HLS STREAM variable=logit_pixels depth=2
#pragma HLS BIND_STORAGE variable=input_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=stem_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=dw1_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=pw1_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=pool1_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=dw2_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=pw2_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=pool2_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=dw3_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=pw3_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=pool3_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=dw4_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=pw4_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=gap_pixels type=fifo impl=srl
#pragma HLS BIND_STORAGE variable=logit_pixels type=fifo impl=srl

    read_input(input, input_pixels);
    stem_conv_stream(input_pixels, stem_pixels);
    // Block 1 at 100x100: DW1 (8 ch/cycle -> 20,402 cy), PW1 16->32
    // (8x8 tile -> 80,000 cy, the critical path), pool to 50x50.
    depthwise_conv_stream<C_STEM, H0, W0, 8>(
        stem_pixels, dw1_pixels, W_DW1, B_DW1);
    pointwise_conv_stream<C_STEM, C_B1, H0 * W0, 8, 8, true>(
        dw1_pixels, pw1_pixels, W_PW1, B_PW1);
    maxpool_stream<C_B1, H0, W0>(pw1_pixels, pool1_pixels);
    // Block 2 at 50x50: DW2 (4 ch/cycle -> 20,808 cy), PW2 32->64
    // (8x8 tile -> 80,000 cy, ties PW1), pool to 25x25.
    depthwise_conv_stream<C_B1, H1, W1, 4>(
        pool1_pixels, dw2_pixels, W_DW2, B_DW2);
    pointwise_conv_stream<C_B1, C_B2, H1 * W1, 8, 8, true>(
        dw2_pixels, pw2_pixels, W_PW2, B_PW2);
    maxpool_stream<C_B2, H1, W1>(pw2_pixels, pool2_pixels);
    // Block 3 at 25x25: DW3 (2 ch/cycle -> 21,632 cy), PW3 64->96
    // (8x8 tile -> 60,000 cy), pool to 12x12 (floor of 25/2).
    depthwise_conv_stream<C_B2, H2, W2, 2>(
        pool2_pixels, dw3_pixels, W_DW3, B_DW3);
    pointwise_conv_stream<C_B2, C_B3, H2 * W2, 8, 8, true>(
        dw3_pixels, pw3_pixels, W_PW3, B_PW3);
    maxpool_stream<C_B3, H2, W2>(pw3_pixels, pool3_pixels);
    // Block 4 at 12x12: DW4 (2 ch/cycle -> 8,112 cy), PW4 96->128
    // (8x4 tile -> 55,296 cy, well under the 80k critical path).
    depthwise_conv_stream<C_B3, H3, W3, 2>(
        pool3_pixels, dw4_pixels, W_DW4, B_DW4);
    pointwise_conv_stream<C_B3, C_B4, H3 * W3, 8, 4, true>(
        dw4_pixels, pw4_pixels, W_PW4, B_PW4);
    // Head: global average over the 144 remaining pixels, then the 1x1
    // classifier (a single-token pointwise layer with no ReLU).
    gap_stream<C_B4, H3 * W3, 8>(pw4_pixels, gap_pixels);
    pointwise_conv_stream<C_B4, NUM_CLASSES, 1, 8, 2, false>(
        gap_pixels, logit_pixels, W_CLS, B_CLS);
    write_logits_stream(logit_pixels, logits);
}
}
