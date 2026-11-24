#ifndef RADAR_EXPECTED_LOGITS_H
#define RADAR_EXPECTED_LOGITS_H

#include <stdint.h>
#include "radar_classes.h"

/* Bit-exact Vitis 2025.1 ap_fixed C-simulation log, synthetic smoke sample. */
static const int16_t radar_expected_logits[4] = {-2703, 4557, -8259, 888};
static const int radar_expected_predicted_class = RADAR_CLASS_BIRD;
#define RADAR_EXPECTED_LOGIT_TOLERANCE_RAW 0U

#endif
