#ifndef RADAR_CNN_STREAM_TYPES_H
#define RADAR_CNN_STREAM_TYPES_H

#include "radar_cnn_stream.h"

template<int C>
struct Pixel {
    data_t channel[C];
};

// Pixel-major ordering: each token contains all channels for one (y, x).
typedef Pixel<1> InputPixel;
typedef Pixel<C_STEM> StemPixel;
// DW1 is depthwise (stride 1), so it keeps C_STEM channels at the same
// H0 x W0 resolution. Reuses the StemPixel layout.
typedef Pixel<C_STEM> DW1Pixel;
// Block tokens for the rest of the pipeline; the channel count changes at
// each pointwise expansion, the spatial token count at each max pool.
typedef Pixel<C_B1> B1Pixel;      // PW1 out, Pool1 out, DW2 out
typedef Pixel<C_B2> B2Pixel;      // PW2 out, Pool2 out, DW3 out
typedef Pixel<C_B3> B3Pixel;      // PW3 out, Pool3 out, DW4 out
typedef Pixel<C_B4> B4Pixel;      // PW4 out, GAP out
typedef Pixel<NUM_CLASSES> LogitsPixel;

#endif
