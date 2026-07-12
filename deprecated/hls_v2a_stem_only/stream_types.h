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

#endif
