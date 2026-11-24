#ifndef RADAR_ACCELERATOR_H
#define RADAR_ACCELERATOR_H

#include <stdint.h>

#define RADAR_ACCELERATOR_INPUT_COUNT 10000U
#define RADAR_ACCELERATOR_OUTPUT_COUNT 4U
#define RADAR_ACCELERATOR_DEFAULT_TIMEOUT 100000000U

#if RADAR_ACCELERATOR_INPUT_COUNT != 10000U
#error "Radar accelerator input length must be 10000"
#endif
#if RADAR_ACCELERATOR_OUTPUT_COUNT != 4U
#error "Radar accelerator output length must be 4"
#endif

enum radar_accelerator_status {
    RADAR_ACCELERATOR_OK = 0,
    RADAR_ACCELERATOR_ERROR_NOT_INITIALIZED = -1,
    RADAR_ACCELERATOR_ERROR_INITIALIZATION = -2,
    RADAR_ACCELERATOR_ERROR_ARGUMENT = -3,
    RADAR_ACCELERATOR_ERROR_TIMEOUT = -4
};

int radar_accelerator_initialize(void);
int radar_accelerator_run(const int16_t *input, int16_t *output,
                          uint64_t *elapsed_counts);
int radar_accelerator_wait_done(uint32_t timeout_iterations);

#endif
