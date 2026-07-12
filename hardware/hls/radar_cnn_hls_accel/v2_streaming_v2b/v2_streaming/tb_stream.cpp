// Testbench for the V2 streaming accelerator.
//
// 1. Rebuilds the ENTIRE network in the exact V1 frame-buffered operation
//    order (the validated numerical reference) on a synthetic input.
// 2. Checks the stem and DW1 milestone kernels against it.
// 3. Checks every streaming layer template in isolation (each stage is
//    fed its reference input, so a failure pinpoints the broken stage).
// 4. Checks the full-pipeline kernel end to end.
// 5. Optionally (float mode, when given file arguments) replays the 8
//    golden radar samples and compares logits against the PyTorch
//    reference:
//      tb <golden/input_samples.txt> <.../pytorch_reference.csv>
//
// In float mode (-DRADAR_CNN_SOFTWARE_FLOAT) every streaming stage
// reproduces the reference floating-point rounding order, so all
// reference comparisons demand agreement within 1e-6 (expected: exactly
// 0). In ap_fixed mode results must match the reference bit-for-bit.

#include "radar_cnn_stream.h"
#include "stream_types.h"
#include "hls_stream_compat.h"
#include "stream_layers.h"
#include "weights.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

static data_t relu_reference(data_t value) {
    return value > (data_t)0 ? value : (data_t)0;
}

// ---------------------------------------------------------------------
// V1-order reference layers (mirror v1_frame_buffered/radar_cnn_hls.cpp).
// ---------------------------------------------------------------------

static void ref_stem(const data_t *input, data_t out[C_STEM][H0][W0]) {
    for (int oc = 0; oc < C_STEM; ++oc) {
        for (int y = 0; y < H0; ++y) {
            for (int x = 0; x < W0; ++x) {
                acc_t acc = B_STEM[oc];
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        const int iy = y + ky - 1;
                        const int ix = x + kx - 1;
                        if (iy >= 0 && iy < H0 && ix >= 0 && ix < W0) {
                            acc += (acc_t)input[iy * W0 + ix] *
                                   (acc_t)W_STEM[oc][0][ky][kx];
                        }
                    }
                }
                out[oc][y][x] = relu_reference((data_t)acc);
            }
        }
    }
}

template<int C, int S>
static void ref_depthwise(
    const data_t in[C][S][S], data_t out[C][S][S],
    const data_t weights[C][1][3][3], const data_t biases[C]
) {
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < S; ++y) {
            for (int x = 0; x < S; ++x) {
                acc_t acc = biases[c];
                for (int ky = 0; ky < 3; ++ky) {
                    for (int kx = 0; kx < 3; ++kx) {
                        const int iy = y + ky - 1;
                        const int ix = x + kx - 1;
                        if (iy >= 0 && iy < S && ix >= 0 && ix < S) {
                            acc += (acc_t)in[c][iy][ix] *
                                   (acc_t)weights[c][0][ky][kx];
                        }
                    }
                }
                out[c][y][x] = relu_reference((data_t)acc);
            }
        }
    }
}

template<int IN_C, int OUT_C, int S>
static void ref_pointwise(
    const data_t in[IN_C][S][S], data_t out[OUT_C][S][S],
    const data_t weights[OUT_C][IN_C], const data_t biases[OUT_C]
) {
    for (int oc = 0; oc < OUT_C; ++oc) {
        for (int y = 0; y < S; ++y) {
            for (int x = 0; x < S; ++x) {
                acc_t acc = biases[oc];
                for (int ic = 0; ic < IN_C; ++ic) {
                    acc += (acc_t)in[ic][y][x] * (acc_t)weights[oc][ic];
                }
                out[oc][y][x] = relu_reference((data_t)acc);
            }
        }
    }
}

template<int C, int IN_S, int OUT_S>
static void ref_maxpool(const data_t in[C][IN_S][IN_S],
                        data_t out[C][OUT_S][OUT_S]) {
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < OUT_S; ++y) {
            for (int x = 0; x < OUT_S; ++x) {
                data_t m = in[c][2 * y][2 * x];
                const data_t v1 = in[c][2 * y][2 * x + 1];
                const data_t v2 = in[c][2 * y + 1][2 * x];
                const data_t v3 = in[c][2 * y + 1][2 * x + 1];
                if (v1 > m) m = v1;
                if (v2 > m) m = v2;
                if (v3 > m) m = v3;
                out[c][y][x] = m;
            }
        }
    }
}

static void ref_gap(const data_t in[C_B4][H3][W3], data_t out[C_B4]) {
    for (int c = 0; c < C_B4; ++c) {
        acc_t acc = 0;
        for (int y = 0; y < H3; ++y) {
            for (int x = 0; x < W3; ++x) {
                acc += in[c][y][x];
            }
        }
        out[c] = (data_t)(acc / (acc_t)(H3 * W3));
    }
}

static void ref_classifier(const data_t in[C_B4], data_t out[NUM_CLASSES]) {
    for (int oc = 0; oc < NUM_CLASSES; ++oc) {
        acc_t acc = B_CLS[oc];
        for (int ic = 0; ic < C_B4; ++ic) {
            acc += (acc_t)in[ic] * (acc_t)W_CLS[oc][ic];
        }
        out[oc] = (data_t)acc;
    }
}

// ---------------------------------------------------------------------
// Stream helpers for per-stage checks.
// ---------------------------------------------------------------------

#ifdef RADAR_CNN_SOFTWARE_FLOAT
static const float VALUE_TOLERANCE = 1.0e-6f;
#endif

static int total_failures = 0;

template<int C>
static void fill_pixels(hls::stream<Pixel<C> > &s, const data_t *planar,
                        int height, int width) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel<C> p;
            for (int c = 0; c < C; ++c) {
                p.channel[c] = planar[(c * height + y) * width + x];
            }
            s.write(p);
        }
    }
}

template<int C>
static void check_pixels(hls::stream<Pixel<C> > &s, const data_t *planar,
                         int height, int width, const char *name) {
    float maximum_error = 0.0f;
    int mismatches = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel<C> p = s.read();
            for (int c = 0; c < C; ++c) {
                const float expected =
                    (float)planar[(c * height + y) * width + x];
                const float error =
                    std::fabs((float)p.channel[c] - expected);
                if (error > maximum_error) maximum_error = error;
#ifdef RADAR_CNN_SOFTWARE_FLOAT
                if (error > VALUE_TOLERANCE) ++mismatches;
#else
                if (p.channel[c] !=
                    planar[(c * height + y) * width + x]) ++mismatches;
#endif
            }
        }
    }
    if (!s.empty()) {
        std::cout << name << " LEFTOVER_TOKENS" << std::endl;
        ++mismatches;
    }
    std::cout << name << " MAX_ERROR " << maximum_error
              << " MISMATCHES " << mismatches << std::endl;
    if (mismatches != 0) ++total_failures;
}

// ---------------------------------------------------------------------
// Reference activations (static: several MB, too big for the stack).
// ---------------------------------------------------------------------

static data_t input_image[INPUT_TOKENS];
static data_t stem_ref[C_STEM][H0][W0];
static data_t dw1_ref[C_STEM][H0][W0];
static data_t pw1_ref[C_B1][H0][W0];
static data_t p1_ref[C_B1][H1][W1];
static data_t dw2_ref[C_B1][H1][W1];
static data_t pw2_ref[C_B2][H1][W1];
static data_t p2_ref[C_B2][H2][W2];
static data_t dw3_ref[C_B2][H2][W2];
static data_t pw3_ref[C_B3][H2][W2];
static data_t p3_ref[C_B3][H3][W3];
static data_t dw4_ref[C_B3][H3][W3];
static data_t pw4_ref[C_B4][H3][W3];
static data_t gap_ref[C_B4];
static data_t logits_ref[NUM_CLASSES];

static void run_reference(const data_t *input) {
    ref_stem(input, stem_ref);
    ref_depthwise<C_STEM, H0>(stem_ref, dw1_ref, W_DW1, B_DW1);
    ref_pointwise<C_STEM, C_B1, H0>(dw1_ref, pw1_ref, W_PW1, B_PW1);
    ref_maxpool<C_B1, H0, H1>(pw1_ref, p1_ref);
    ref_depthwise<C_B1, H1>(p1_ref, dw2_ref, W_DW2, B_DW2);
    ref_pointwise<C_B1, C_B2, H1>(dw2_ref, pw2_ref, W_PW2, B_PW2);
    ref_maxpool<C_B2, H1, H2>(pw2_ref, p2_ref);
    ref_depthwise<C_B2, H2>(p2_ref, dw3_ref, W_DW3, B_DW3);
    ref_pointwise<C_B2, C_B3, H2>(dw3_ref, pw3_ref, W_PW3, B_PW3);
    ref_maxpool<C_B3, H2, H3>(pw3_ref, p3_ref);
    ref_depthwise<C_B3, H3>(p3_ref, dw4_ref, W_DW4, B_DW4);
    ref_pointwise<C_B3, C_B4, H3>(dw4_ref, pw4_ref, W_PW4, B_PW4);
    ref_gap(pw4_ref, gap_ref);
    ref_classifier(gap_ref, logits_ref);
}

// ---------------------------------------------------------------------
// Checks.
// ---------------------------------------------------------------------

static void check_flat_capture(const data_t *actual,
                               const data_t *planar_ref,
                               int height, int width, int channels,
                               const char *name) {
    float maximum_error = 0.0f;
    int mismatches = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                const data_t a = actual[(y * width + x) * channels + c];
                const data_t e = planar_ref[(c * height + y) * width + x];
                const float error = std::fabs((float)a - (float)e);
                if (error > maximum_error) maximum_error = error;
#ifdef RADAR_CNN_SOFTWARE_FLOAT
                if (error > VALUE_TOLERANCE) ++mismatches;
#else
                if (a != e) ++mismatches;
#endif
            }
        }
    }
    std::cout << name << " MAX_ERROR " << maximum_error
              << " MISMATCHES " << mismatches << std::endl;
    if (mismatches != 0) ++total_failures;
}

static void check_stage_templates() {
    // Each stage gets its REFERENCE input, so errors do not compound and
    // a failure localizes the broken stage.
    {
        hls::stream<StemPixel> in, out;
        fill_pixels<C_STEM>(in, &stem_ref[0][0][0], H0, W0);
        depthwise_conv_stream<C_STEM, H0, W0, C_STEM>(in, out, W_DW1, B_DW1);
        check_pixels<C_STEM>(out, &dw1_ref[0][0][0], H0, W0, "STAGE_DW1");
    }
    {
        hls::stream<StemPixel> in;
        hls::stream<B1Pixel> out;
        fill_pixels<C_STEM>(in, &dw1_ref[0][0][0], H0, W0);
        pointwise_conv_stream<C_STEM, C_B1, H0 * W0, 8, 4, true>(
            in, out, W_PW1, B_PW1);
        check_pixels<C_B1>(out, &pw1_ref[0][0][0], H0, W0, "STAGE_PW1");
    }
    {
        hls::stream<B1Pixel> in, out;
        fill_pixels<C_B1>(in, &pw1_ref[0][0][0], H0, W0);
        maxpool_stream<C_B1, H0, W0>(in, out);
        check_pixels<C_B1>(out, &p1_ref[0][0][0], H1, W1, "STAGE_POOL1");
    }
    {
        hls::stream<B1Pixel> in, out;
        fill_pixels<C_B1>(in, &p1_ref[0][0][0], H1, W1);
        depthwise_conv_stream<C_B1, H1, W1, 8>(in, out, W_DW2, B_DW2);
        check_pixels<C_B1>(out, &dw2_ref[0][0][0], H1, W1, "STAGE_DW2");
    }
    {
        hls::stream<B1Pixel> in;
        hls::stream<B2Pixel> out;
        fill_pixels<C_B1>(in, &dw2_ref[0][0][0], H1, W1);
        pointwise_conv_stream<C_B1, C_B2, H1 * W1, 8, 4, true>(
            in, out, W_PW2, B_PW2);
        check_pixels<C_B2>(out, &pw2_ref[0][0][0], H1, W1, "STAGE_PW2");
    }
    {
        hls::stream<B2Pixel> in, out;
        fill_pixels<C_B2>(in, &pw2_ref[0][0][0], H1, W1);
        maxpool_stream<C_B2, H1, W1>(in, out);
        check_pixels<C_B2>(out, &p2_ref[0][0][0], H2, W2, "STAGE_POOL2");
    }
    {
        hls::stream<B2Pixel> in, out;
        fill_pixels<C_B2>(in, &p2_ref[0][0][0], H2, W2);
        depthwise_conv_stream<C_B2, H2, W2, 8>(in, out, W_DW3, B_DW3);
        check_pixels<C_B2>(out, &dw3_ref[0][0][0], H2, W2, "STAGE_DW3");
    }
    {
        hls::stream<B2Pixel> in;
        hls::stream<B3Pixel> out;
        fill_pixels<C_B2>(in, &dw3_ref[0][0][0], H2, W2);
        pointwise_conv_stream<C_B2, C_B3, H2 * W2, 8, 4, true>(
            in, out, W_PW3, B_PW3);
        check_pixels<C_B3>(out, &pw3_ref[0][0][0], H2, W2, "STAGE_PW3");
    }
    {
        // 25 -> 12: exercises the odd-dimension floor behavior.
        hls::stream<B3Pixel> in, out;
        fill_pixels<C_B3>(in, &pw3_ref[0][0][0], H2, W2);
        maxpool_stream<C_B3, H2, W2>(in, out);
        check_pixels<C_B3>(out, &p3_ref[0][0][0], H3, W3, "STAGE_POOL3");
    }
    {
        hls::stream<B3Pixel> in, out;
        fill_pixels<C_B3>(in, &p3_ref[0][0][0], H3, W3);
        depthwise_conv_stream<C_B3, H3, W3, 8>(in, out, W_DW4, B_DW4);
        check_pixels<C_B3>(out, &dw4_ref[0][0][0], H3, W3, "STAGE_DW4");
    }
    {
        hls::stream<B3Pixel> in;
        hls::stream<B4Pixel> out;
        fill_pixels<C_B3>(in, &dw4_ref[0][0][0], H3, W3);
        pointwise_conv_stream<C_B3, C_B4, H3 * W3, 8, 2, true>(
            in, out, W_PW4, B_PW4);
        check_pixels<C_B4>(out, &pw4_ref[0][0][0], H3, W3, "STAGE_PW4");
    }
    {
        hls::stream<B4Pixel> in, out;
        fill_pixels<C_B4>(in, &pw4_ref[0][0][0], H3, W3);
        gap_stream<C_B4, H3 * W3, 8>(in, out);
        check_pixels<C_B4>(out, &gap_ref[0], 1, 1, "STAGE_GAP");
    }
    {
        hls::stream<B4Pixel> in;
        hls::stream<LogitsPixel> out;
        fill_pixels<C_B4>(in, &gap_ref[0], 1, 1);
        pointwise_conv_stream<C_B4, NUM_CLASSES, 1, 8, 2, false>(
            in, out, W_CLS, B_CLS);
        check_pixels<NUM_CLASSES>(out, &logits_ref[0], 1, 1, "STAGE_CLS");
    }
}

static int argmax4(const float *v) {
    int best = 0;
    for (int i = 1; i < 4; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

#ifdef RADAR_CNN_SOFTWARE_FLOAT
// Golden replay: run the 8 real radar samples through the full streaming
// kernel and compare against the PyTorch logits (same 2e-4 tolerance as
// the recorded V1 host-float equivalence report).
static void check_golden(const char *samples_path, const char *csv_path) {
    FILE *fs = std::fopen(samples_path, "r");
    FILE *fc = std::fopen(csv_path, "r");
    if (!fs || !fc) {
        std::cout << "GOLDEN SKIPPED (files not found)" << std::endl;
        if (fs) std::fclose(fs);
        if (fc) std::fclose(fc);
        return;
    }

    char line[512];
    if (!std::fgets(line, sizeof(line), fc)) {  // CSV header
        std::cout << "GOLDEN SKIPPED (empty csv)" << std::endl;
        std::fclose(fs); std::fclose(fc);
        return;
    }

    int sample_count = 0;
    if (std::fscanf(fs, "%d", &sample_count) != 1) sample_count = 0;

    static data_t sample_input[INPUT_TOKENS];
    static data_t sample_logits[NUM_CLASSES];
    float global_max_error = 0.0f;
    int failed_samples = 0;

    for (int s = 0; s < sample_count; ++s) {
        int cache_index = -1, label = -1;
        if (std::fscanf(fs, "%d %d", &cache_index, &label) != 2) break;
        for (int i = 0; i < INPUT_TOKENS; ++i) {
            float v = 0.0f;
            if (std::fscanf(fs, "%f", &v) != 1) v = 0.0f;
            sample_input[i] = (data_t)v;
        }

        int csv_cache = -1, csv_label = -1, csv_pred = -1;
        float expected[4] = {0, 0, 0, 0};
        if (!std::fgets(line, sizeof(line), fc) ||
            std::sscanf(line, "%d,%*[^,],%d,%*[^,],%d,%*[^,],%f,%f,%f,%f",
                        &csv_cache, &csv_label, &csv_pred,
                        &expected[0], &expected[1],
                        &expected[2], &expected[3]) != 7 ||
            csv_cache != cache_index) {
            std::cout << "GOLDEN SAMPLE " << s
                      << " CSV_PARSE_MISMATCH" << std::endl;
            ++failed_samples;
            continue;
        }

        radar_cnn_stream_full(sample_input, sample_logits);

        float actual[4];
        float max_error = 0.0f;
        for (int i = 0; i < 4; ++i) {
            actual[i] = (float)sample_logits[i];
            const float error = std::fabs(actual[i] - expected[i]);
            if (error > max_error) max_error = error;
        }
        if (max_error > global_max_error) global_max_error = max_error;
        const bool pred_ok = argmax4(actual) == csv_pred;
        const bool ok = (max_error <= 2.0e-4f) && pred_ok;
        if (!ok) ++failed_samples;
        std::cout << "GOLDEN SAMPLE " << s << " cache=" << cache_index
                  << " MAX_ERROR " << max_error
                  << " PRED " << argmax4(actual) << "/" << csv_pred
                  << (ok ? " OK" : " FAIL") << std::endl;
    }
    std::fclose(fs);
    std::fclose(fc);
    std::cout << "GOLDEN MAX_ERROR " << global_max_error
              << " FAILED " << failed_samples << "/" << sample_count
              << std::endl;
    if (failed_samples != 0) ++total_failures;
}
#endif

int main(int argc, char **argv) {
    for (int y = 0; y < IMG_H; ++y) {
        for (int x = 0; x < IMG_W; ++x) {
            input_image[y * IMG_W + x] = (data_t)(
                0.5f + 0.5f * std::sin(0.05f * y) * std::cos(0.05f * x));
        }
    }

    run_reference(input_image);

    // --- Milestone kernels -------------------------------------------
    static data_t stem_actual[STEM_VALUES];
    radar_cnn_stream_stem(input_image, stem_actual);
    check_flat_capture(stem_actual, &stem_ref[0][0][0], H0, W0, C_STEM,
                       "KERNEL_STEM");

    static data_t dw1_actual[DW1_VALUES];
    radar_cnn_stream_dw1(input_image, dw1_actual);
    check_flat_capture(dw1_actual, &dw1_ref[0][0][0], H0, W0, C_STEM,
                       "KERNEL_DW1");

    // --- Every streaming stage in isolation --------------------------
    check_stage_templates();

    // --- Full pipeline end to end ------------------------------------
    static data_t logits_actual[NUM_CLASSES];
    radar_cnn_stream_full(input_image, logits_actual);
    {
        float maximum_error = 0.0f;
        int mismatches = 0;
        for (int i = 0; i < NUM_CLASSES; ++i) {
            const float error = std::fabs(
                (float)logits_actual[i] - (float)logits_ref[i]);
            if (error > maximum_error) maximum_error = error;
#ifdef RADAR_CNN_SOFTWARE_FLOAT
            if (error > VALUE_TOLERANCE) ++mismatches;
#else
            if (logits_actual[i] != logits_ref[i]) ++mismatches;
#endif
            std::cout << "LOGIT " << i << " " << (float)logits_actual[i]
                      << " (ref " << (float)logits_ref[i] << ")"
                      << std::endl;
        }
        std::cout << "KERNEL_FULL MAX_ERROR " << maximum_error
                  << " MISMATCHES " << mismatches << std::endl;
        if (mismatches != 0) ++total_failures;
    }

    // --- Golden radar samples vs PyTorch (float mode only) -----------
#ifdef RADAR_CNN_SOFTWARE_FLOAT
    if (argc >= 3) {
        check_golden(argv[1], argv[2]);
    } else {
        std::cout << "GOLDEN SKIPPED (no file arguments)" << std::endl;
    }
#else
    (void)argc;
    (void)argv;
#endif

    std::cout << (total_failures == 0 ? "ALL CHECKS PASSED"
                                      : "CHECKS FAILED")
              << " (failures=" << total_failures << ")" << std::endl;
    return total_failures == 0 ? 0 : 1;
}
