#ifndef RADAR_CNN_STREAM_H
#define RADAR_CNN_STREAM_H

#ifdef RADAR_CNN_SOFTWARE_FLOAT
typedef float data_t;
typedef float acc_t;
#else
#include <ap_fixed.h>
typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<32, 12> acc_t;
#endif

// Spatial sizes. Downsampling happens ONLY in the three 2x2 max-pool
// stages, exactly as in the validated V1 reference: 100 -> 50 -> 25 -> 12
// (25/2 floors to 12; the trailing row/column is consumed but unused).
constexpr int H0 = 100;  // stem, DW1, PW1 resolution
constexpr int W0 = 100;
constexpr int H1 = 50;   // after Pool1: DW2, PW2 resolution
constexpr int W1 = 50;
constexpr int H2 = 25;   // after Pool2: DW3, PW3 resolution
constexpr int W2 = 25;
constexpr int H3 = 12;   // after Pool3: DW4, PW4, GAP resolution
constexpr int W3 = 12;

// Channel widths per block. These match weights.h and the trained
// checkpoint (radar_dscnn2d_topk3_capped8_best.pth): note PW3 expands
// 64 -> 96 and PW4 expands 96 -> 128.
constexpr int C_STEM = 16;
constexpr int C_B1 = 32;
constexpr int C_B2 = 64;
constexpr int C_B3 = 96;
constexpr int C_B4 = 128;
constexpr int NUM_CLASSES = 4;

constexpr int IMG_H = H0;
constexpr int IMG_W = W0;
constexpr int INPUT_TOKENS = H0 * W0;
constexpr int STEM_TOKENS = H0 * W0;
constexpr int STEM_VALUES = STEM_TOKENS * C_STEM;

// DW1 is a stride-1, padding-1, 3x3 depthwise convolution, exactly as in
// the validated V1 reference (v1_frame_buffered/radar_cnn_hls.cpp):
// spatial size is unchanged and the 100 -> 50 downsample happens later,
// in Pool1 after PW1. An earlier V2B draft implemented DW1 with stride 2;
// that assumption did not match the trained checkpoint and was corrected.
constexpr int DW1_TOKENS = H0 * W0;
constexpr int DW1_VALUES = DW1_TOKENS * C_STEM;

extern "C" {
void radar_cnn_stream_stem(const data_t *input, data_t *stem_output);
void radar_cnn_stream_dw1(const data_t *input, data_t *dw1_output);
void radar_cnn_stream_full(const data_t *input, data_t *logits);
}

#endif
