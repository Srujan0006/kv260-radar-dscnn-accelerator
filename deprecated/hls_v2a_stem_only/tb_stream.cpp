#include "radar_cnn_stream.h"
#include "weights.h"

#include <cmath>
#include <iostream>

static data_t relu_reference(data_t value) {
    return value > (data_t)0 ? value : (data_t)0;
}

int main() {
    static data_t input[STEM_TOKENS];
    static data_t actual[STEM_VALUES];
    static data_t expected[STEM_VALUES];

    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            input[y * IMG_W + x] = (data_t)(
                0.5f + 0.5f * std::sin(0.05f * y) * std::cos(0.05f * x));
        }
    }

    radar_cnn_stream_stem(input, actual);

    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            for (int oc = 0; oc < C_STEM; ++oc) {
                acc_t acc = B_STEM[oc];
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        const int iy = y + ky - 1;
                        const int ix = x + kx - 1;
                        if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W) {
                            acc += (acc_t)input[iy * IMG_W + ix] *
                                   (acc_t)W_STEM[oc][0][ky][kx];
                        }
                    }
                }
                expected[(y * IMG_W + x) * C_STEM + oc] =
                    relu_reference((data_t)acc);
            }
        }
    }

    float maximum_error = 0.0f;
    int mismatches = 0;
    for (int i = 0; i < STEM_VALUES; ++i) {
        const float error = std::fabs((float)actual[i] - (float)expected[i]);
        if (error > maximum_error) maximum_error = error;
#ifdef RADAR_CNN_SOFTWARE_FLOAT
        if (error > 1.0e-6f) ++mismatches;
#else
        if (actual[i] != expected[i]) ++mismatches;
#endif
    }

    const int corners[4] = {
        0,
        (IMG_W - 1) * C_STEM,
        ((IMG_H - 1) * IMG_W) * C_STEM,
        (IMG_H * IMG_W - 1) * C_STEM
    };
    for (int i = 0; i < 4; ++i) {
        std::cout << "CORNER " << i << " CH0 "
                  << (float)actual[corners[i]] << std::endl;
    }
    std::cout << "STEM_VALUES " << STEM_VALUES << std::endl;
    std::cout << "MAX_ERROR " << maximum_error << std::endl;
    std::cout << "MISMATCHES " << mismatches << std::endl;
    return mismatches == 0 ? 0 : 1;
}
