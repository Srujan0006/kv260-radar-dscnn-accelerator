#ifndef RADAR_DRIVER_BINDING_H
#define RADAR_DRIVER_BINDING_H

/*
 * This is the only application file that may name generated Vitis HLS driver
 * symbols.  The generated BSP artifacts needed to fill this binding are not
 * present in this repository, so compilation stops instead of guessing them.
 */
#error "Generated Vitis accelerator driver files are missing; configure radar_driver_binding.h from xparameters.h and the generated driver"

/* Required polling binding after the generated files are supplied:
 *
 * #include "<exact-generated-driver-header>.h"
 * typedef <exact-generated-instance-type> radar_driver_instance_t;
 * #define RADAR_DRIVER_DEVICE_ID       <exact-XPAR-device-id-or-config-macro>
 * #define RADAR_DRIVER_INITIALIZE(p,id) <exact-initialize-function>((p),(id))
 * #define RADAR_DRIVER_SET_INPUT(p,a)  <exact-64-bit-input-setter>((p),(a))
 * #define RADAR_DRIVER_SET_OUTPUT(p,a) <exact-64-bit-output-setter>((p),(a))
 * #define RADAR_DRIVER_START(p)        <exact-start-function>((p))
 * #define RADAR_DRIVER_IS_DONE(p)      <exact-done-function>((p))
 * #define RADAR_DRIVER_IS_IDLE(p)      <exact-idle-function>((p))
 *
 * Do not add register offsets here. The setter argument casts must match the
 * generated declarations exactly and must preserve UINTPTR addresses.
 */

#ifndef RADAR_USE_INTERRUPTS
#define RADAR_USE_INTERRUPTS 0
#endif

/* Interrupt symbol aliases belong inside this guard if interrupts are later
 * enabled. The polling build neither defines nor references interrupt APIs. */
#if RADAR_USE_INTERRUPTS
#error "Interrupt bindings have not been configured from a generated driver"
#endif

#endif
