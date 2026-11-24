#include <stdint.h>
#include <string.h>

#include "xil_printf.h"
#include "xtime_l.h"

#include "radar_accelerator.h"
#include "radar_classes.h"
#include "radar_expected_logits.h"
#include "radar_fixed_point.h"
#include "radar_test_vector.h"

#define WARMUP_RUNS 10U
#define MEASURED_RUNS 100U

static int16_t input_buffer[RADAR_ACCELERATOR_INPUT_COUNT]
    __attribute__((aligned(64)));
static int16_t output_buffer[RADAR_ACCELERATOR_OUTPUT_COUNT]
    __attribute__((aligned(64)));

static void print_u64(uint64_t value)
{
    char digits[21];
    unsigned int used = 0U;
    do {
        digits[used++] = (char)('0' + (value % 10U));
        value /= 10U;
    } while (value != 0U);
    while (used != 0U) {
        xil_printf("%c", digits[--used]);
    }
}

static void print_fixed(int16_t raw)
{
    radar_fixed_printable_t value = radar_fixed_to_printable(raw);
    if (value.integer_part == INT32_MIN) {
        xil_printf("-0.%06lu", (unsigned long)value.fractional_magnitude);
    } else {
        xil_printf("%ld.%06lu", (long)value.integer_part,
                   (unsigned long)value.fractional_magnitude);
    }
}

static int run_once(uint64_t *counts)
{
    memset(output_buffer, 0, sizeof(output_buffer));
    return radar_accelerator_run(input_buffer, output_buffer, counts);
}

int main(void)
{
    uint64_t counts;
    uint64_t total = 0U;
    uint64_t minimum = UINT64_MAX;
    uint64_t maximum = 0U;
    uint32_t run;
    int predicted;
    int correctness = 1;
    int status;
    int i;

    xil_printf("\r\nKV260 Streaming Radar CNN Bare-Metal Test\r\n");
    xil_printf("Sample: %s, fixed Q%d.%d, %lu inputs, 4 classes\r\n",
               RADAR_TEST_SAMPLE_ID, RADAR_FIXED_INTEGER_BITS,
               RADAR_FIXED_FRACTIONAL_BITS,
               (unsigned long)RADAR_TEST_INPUT_COUNT);

    memcpy(input_buffer, radar_test_input, sizeof(input_buffer));
    memset(output_buffer, 0, sizeof(output_buffer));
    status = radar_accelerator_initialize();
    if (status != RADAR_ACCELERATOR_OK) {
        xil_printf("ERROR: accelerator initialization failed (%d)\r\n", status);
        return 1;
    }

    status = run_once(&counts);
    if (status != RADAR_ACCELERATOR_OK) {
        xil_printf("ERROR: correctness run failed (%d)\r\n", status);
        return 2;
    }
    xil_printf("Correctness run counts: ");
    print_u64(counts);
    xil_printf("\r\n");
    for (i = 0; i < RADAR_NUM_CLASSES; ++i) {
        int32_t difference = radar_fixed_difference(
            output_buffer[i], radar_expected_logits[i]);
        xil_printf("logit[%d]: raw=%d decoded=", i, output_buffer[i]);
        print_fixed(output_buffer[i]);
        xil_printf(" expected_raw=%d expected_decoded=", radar_expected_logits[i]);
        print_fixed(radar_expected_logits[i]);
        xil_printf(" difference=%ld\r\n", (long)difference);
        if (!radar_fixed_equal(output_buffer[i], radar_expected_logits[i],
                               RADAR_EXPECTED_LOGIT_TOLERANCE_RAW)) {
            correctness = 0;
        }
    }
    predicted = radar_fixed_argmax4(output_buffer);
    xil_printf("Expected class: %d (%s); predicted: %d (%s)\r\n",
               RADAR_TEST_EXPECTED_CLASS, radar_class_name(RADAR_TEST_EXPECTED_CLASS),
               predicted, radar_class_name(predicted));
    if (predicted != radar_expected_predicted_class ||
        predicted != RADAR_TEST_EXPECTED_CLASS) {
        correctness = 0;
    }
    xil_printf("CORRECTNESS: %s\r\n", correctness ? "PASS" : "FAIL");
    if (!correctness) {
        return 3;
    }

    for (run = 0U; run < WARMUP_RUNS; ++run) {
        status = run_once(&counts);
        if (status != RADAR_ACCELERATOR_OK) {
            xil_printf("ERROR: warm-up %lu failed (%d)\r\n",
                       (unsigned long)run, status);
            return 4;
        }
    }
    for (run = 0U; run < MEASURED_RUNS; ++run) {
        status = run_once(&counts);
        if (status != RADAR_ACCELERATOR_OK) {
            xil_printf("ERROR: measured run %lu failed (%d)\r\n",
                       (unsigned long)run, status);
            return 5;
        }
        if (counts < minimum) minimum = counts;
        if (counts > maximum) maximum = counts;
        total += counts;
    }

    {
        uint64_t average = total / MEASURED_RUNS;
        uint64_t min_us = (minimum * 1000000ULL) / COUNTS_PER_SECOND;
        uint64_t max_us = (maximum * 1000000ULL) / COUNTS_PER_SECOND;
        uint64_t avg_us = (average * 1000000ULL) / COUNTS_PER_SECOND;
        uint64_t throughput_milli = (average == 0U) ? 0U :
            ((uint64_t)COUNTS_PER_SECOND * 1000ULL) / average;
        xil_printf("Latency: min="); print_u64(min_us);
        xil_printf(" us max="); print_u64(max_us);
        xil_printf(" us avg="); print_u64(avg_us);
        xil_printf(" us\r\nThroughput: ");
        print_u64(throughput_milli / 1000ULL);
        xil_printf(".%03lu inferences/s\r\n",
                   (unsigned long)(throughput_milli % 1000ULL));
        xil_printf("CSV,%s,%d,%d,%d,%d,%d,%d,",
                   RADAR_TEST_SAMPLE_ID, RADAR_TEST_EXPECTED_CLASS, predicted,
                   output_buffer[0], output_buffer[1], output_buffer[2],
                   output_buffer[3]);
        print_u64(average); xil_printf(","); print_u64(avg_us);
        xil_printf(",PASS\r\n");
    }
    return 0;
}
