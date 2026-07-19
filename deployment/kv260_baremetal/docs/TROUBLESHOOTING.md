# Troubleshooting

## Driver or build problems

- **Accelerator header not found:** inspect the built standalone BSP include
  tree. Confirm the IP is present in the selected XSA and regenerate the BSP.
  Copy its exact header token into `radar_driver_binding.h`.
- **Device-ID macro not found:** search the generated `xparameters.h` for the
  accelerator instance. A missing entry usually means the wrong XSA/domain or
  an IP absent from the address map. Never invent the macro.
- **XTime linker errors:** ensure the application uses the standalone A53 BSP
  and includes `xtime_l.h`; rebuild the BSP libraries. Do not add a Linux timer.
- **Float formatting:** `xil_printf` does not provide normal floating `%f`.
  This application prints scaled integer/fractional fields and needs no float
  printf linker option.

## Execution and data problems

- **All-zero outputs:** confirm the output pointer setter, HP0_FPD DDR routing,
  start/done status, and that the bitstream matches the XSA. Check the PL clock
  and reset. Zero output does not prove successful inference.
- **Stale outputs:** flush/initialize the output range before start and invalidate
  it only after done. Confirm byte lengths cover all four `int16_t` logits.
- **Never reaches done / application hangs:** retain the polling timeout. Check
  AXI-Lite base mapping, PL reset/clock, master access to DDR, and ILA evidence
  for AXI faults. Confirm `IsDone`, not an unverified register offset, is used.
- **DDR address truncation:** use `UINTPTR`/`INTPTR` and the driver's actual
  setter parameter type. A cast to `u32` is invalid for a 64-bit A53 address.
- **Cache coherency errors:** HP0_FPD is non-coherent. Flush the complete input
  and initialized output ranges before start; invalidate output after done.
- **Wrong fixed-point scaling:** the final source declares
  `ap_fixed<16,6,AP_TRN,AP_WRAP>`: signed raw `int16_t`, 10 fractional bits,
  scale 1024. Do not pass IEEE float bytes.
- **Wrong input ordering:** the HLS top consumes flattened row-major
  `[y * 100 + x]`. Regenerate from the exact testbench sample without transpose.
- **Incorrect class order:** 0 Drone, 1 Bird, 2 Human, 3 CornerReflector.

## Board and debug problems

- **JTAG target not found:** verify power, USB data cable, drivers, board boot
  mode, and that no other debugger owns the cable. Refresh hardware targets.
- **PS initialization failure:** select the initialization generated from the
  same XSA, target `psu_cortexa53_0`, power-cycle, then retry.
- **Wrong KV260 preset/DDR configuration:** reopen the Vivado design and verify
  the KV260/K26 preset, DDR configuration, clocks, resets, and HP0_FPD mapping.
  Export a new hardware XSA only after correcting a genuine mismatch.
