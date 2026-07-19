#ifndef RADAR_FIXED_POINT_H
#define RADAR_FIXED_POINT_H

#include <stdint.h>
#include <limits.h>

#define RADAR_FIXED_TOTAL_BITS 16
#define RADAR_FIXED_INTEGER_BITS 6
#define RADAR_FIXED_FRACTIONAL_BITS 10
#define RADAR_FIXED_SCALE (1L << RADAR_FIXED_FRACTIONAL_BITS)
#define RADAR_NUM_CLASSES 4

/* Bit-exact storage contract for ap_fixed<16,6,AP_TRN,AP_WRAP>. Conversion
 * truncates toward negative infinity (AP_TRN) and wraps modulo 2^16
 * (AP_WRAP). Range checks are diagnostics only; they do not model saturation. */
#if RADAR_FIXED_FRACTIONAL_BITS != 10
#error "Radar fixed-point fractional-bit contract must be 10"
#endif
#if INT16_MAX != 32767 || INT16_MIN != (-32767 - 1)
#error "Radar deployment requires a 16-bit int16_t"
#endif

typedef struct {
    int32_t integer_part;
    uint32_t fractional_magnitude;
} radar_fixed_printable_t;

radar_fixed_printable_t radar_fixed_to_printable(int16_t raw);
int32_t radar_fixed_difference(int16_t actual, int16_t expected);
int radar_fixed_equal(int16_t actual, int16_t expected, uint16_t tolerance_raw);
int radar_fixed_argmax4(const int16_t logits[RADAR_NUM_CLASSES]);

#endif
