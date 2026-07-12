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

constexpr int H0 = 100;
constexpr int W0 = 100;
constexpr int H1 = 50;
constexpr int W1 = 50;
constexpr int H2 = 25;
constexpr int W2 = 25;
constexpr int H3 = 12;
constexpr int W3 = 12;
constexpr int C_STEM = 16;

constexpr int IMG_H = H0;
constexpr int IMG_W = W0;
constexpr int INPUT_TOKENS = H0 * W0;
constexpr int STEM_TOKENS = H0 * W0;
constexpr int STEM_VALUES = STEM_TOKENS * C_STEM;

extern "C" {
void radar_cnn_stream_stem(const data_t *input, data_t *stem_output);
}

#endif
