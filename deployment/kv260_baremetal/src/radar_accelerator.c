#include "radar_accelerator.h"

#include "xil_cache.h"
#include "xil_types.h"
#include "xtime_l.h"

#include "radar_driver_binding.h"

static radar_driver_instance_t accelerator_instance;
static int accelerator_initialized;

int radar_accelerator_initialize(void)
{
    int status = RADAR_DRIVER_INITIALIZE(&accelerator_instance,
                                         RADAR_DRIVER_DEVICE_ID);
    if (status != 0) {
        accelerator_initialized = 0;
        return RADAR_ACCELERATOR_ERROR_INITIALIZATION;
    }
    accelerator_initialized = 1;
    return RADAR_ACCELERATOR_OK;
}

int radar_accelerator_wait_done(uint32_t timeout_iterations)
{
    uint32_t iteration;
    if (!accelerator_initialized) {
        return RADAR_ACCELERATOR_ERROR_NOT_INITIALIZED;
    }
    for (iteration = 0U; iteration < timeout_iterations; ++iteration) {
        if (RADAR_DRIVER_IS_DONE(&accelerator_instance) != 0U) {
            return RADAR_ACCELERATOR_OK;
        }
    }
    return RADAR_ACCELERATOR_ERROR_TIMEOUT;
}

int radar_accelerator_run(const int16_t *input, int16_t *output,
                          uint64_t *elapsed_counts)
{
    XTime start_time;
    XTime end_time;
    int status;
    if (!accelerator_initialized) {
        return RADAR_ACCELERATOR_ERROR_NOT_INITIALIZED;
    }
    if (input == 0 || output == 0 || elapsed_counts == 0) {
        return RADAR_ACCELERATOR_ERROR_ARGUMENT;
    }

    /* HP0_FPD is non-coherent: make PS writes visible to the PL master. */
    Xil_DCacheFlushRange((INTPTR)input,
        RADAR_ACCELERATOR_INPUT_COUNT * (uint32_t)sizeof(input[0]));
    Xil_DCacheFlushRange((INTPTR)output,
        RADAR_ACCELERATOR_OUTPUT_COUNT * (uint32_t)sizeof(output[0]));

    /* UINTPTR preserves the A53's 64-bit DDR address; never cast to u32. */
    RADAR_DRIVER_SET_INPUT(&accelerator_instance, (UINTPTR)input);
    RADAR_DRIVER_SET_OUTPUT(&accelerator_instance, (UINTPTR)output);

    XTime_GetTime(&start_time);
    RADAR_DRIVER_START(&accelerator_instance);
    status = radar_accelerator_wait_done(RADAR_ACCELERATOR_DEFAULT_TIMEOUT);
    XTime_GetTime(&end_time);
    if (status != RADAR_ACCELERATOR_OK) {
        return status;
    }
    *elapsed_counts = (uint64_t)(end_time - start_time);

    /* Discard stale PS cache lines before reading PL-written logits. */
    Xil_DCacheInvalidateRange((INTPTR)output,
        RADAR_ACCELERATOR_OUTPUT_COUNT * (uint32_t)sizeof(output[0]));
    return RADAR_ACCELERATOR_OK;
}
