#include "radar_fixed_point.h"

radar_fixed_printable_t radar_fixed_to_printable(int16_t raw)
{
    radar_fixed_printable_t result;
    int32_t value = (int32_t)raw;
    int32_t magnitude = (value < 0) ? -value : value;
    result.integer_part = magnitude / RADAR_FIXED_SCALE;
    result.fractional_magnitude =
        (uint32_t)(((magnitude % RADAR_FIXED_SCALE) * 1000000L) /
                   RADAR_FIXED_SCALE);
    if (value < 0) {
        result.integer_part = -result.integer_part;
        /* Preserve the sign for values in (-1, 0) when printed separately. */
        if (result.integer_part == 0) {
            result.integer_part = INT32_MIN;
        }
    }
    return result;
}

int32_t radar_fixed_difference(int16_t actual, int16_t expected)
{
    return (int32_t)actual - (int32_t)expected;
}

int radar_fixed_equal(int16_t actual, int16_t expected, uint16_t tolerance_raw)
{
    int32_t difference = radar_fixed_difference(actual, expected);
    if (difference < 0) {
        difference = -difference;
    }
    return (uint32_t)difference <= (uint32_t)tolerance_raw;
}

int radar_fixed_argmax4(const int16_t logits[RADAR_NUM_CLASSES])
{
    int best = 0;
    int i;
    for (i = 1; i < RADAR_NUM_CLASSES; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}
