#include "radar_cnn_stream.h"
#include "stream_types.h"
#include "weights.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

#ifdef RADAR_CNN_SOFTWARE_FLOAT
#include <queue>
namespace hls {
template<typename T>
class stream {
public:
    void write(const T &value) { values.push(value); }
    T read() { T value = values.front(); values.pop(); return value; }
private:
    std::queue<T> values;
};
}
#else
#include <hls_stream.h>
#endif

static data_t relu_stream(data_t value) {
#pragma HLS INLINE
    return value > (data_t)0 ? value : (data_t)0;
}

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

extern "C" {
void radar_cnn_stream_stem(const data_t *input, data_t *stem_output) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=10000
#pragma HLS INTERFACE m_axi port=stem_output offset=slave bundle=gmem1 depth=160000
#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=stem_output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW

    hls::stream<InputPixel> input_pixels;
    hls::stream<StemPixel> stem_pixels;
#pragma HLS STREAM variable=input_pixels depth=32
#pragma HLS STREAM variable=stem_pixels depth=16

    read_input(input, input_pixels);
    stem_conv_stream(input_pixels, stem_pixels);
    capture_stem(stem_pixels, stem_output);
}
}
