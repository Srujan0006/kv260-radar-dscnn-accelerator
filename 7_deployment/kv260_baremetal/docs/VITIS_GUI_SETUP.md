# Vitis Unified IDE 2025.1 GUI setup

Keep the generated workspace at `C:\KV260_Radar_Workspace`; keep the source of
truth in `deployment\kv260_baremetal`.

1. Start Vitis Unified IDE 2025.1 and select the external workspace directory.
2. Choose **File > New Component > Platform**. Select **Hardware Design** and
   browse to
   `hardware\vivado\export\radar_cnn_kv260.xsa`. Confirm that the XSA contains
   the implemented bitstream.
3. Select processor `psu_cortexa53_0`. Create a **Standalone** domain for the
   64-bit A53 architecture; do not select Linux. Build the Platform Component.
4. Open the platform's generated BSP sources/include view. Verify
   `xparameters.h` and the four HLS driver files: the generated `x*.h`, `x*.c`,
   and `x*_hw.h`. Record the exact instance typedef, device-ID macro, initialize,
   input/output setter, start, done, idle, and interrupt function names.
5. Update only `include\radar_driver_binding.h` with those exact symbols.
   Ensure the address setters accept the full pointer width. Rebuild the platform
   after any BSP setting change.
6. Choose **File > New Component > Application > Empty Application**. Target the
   platform and its 64-bit standalone domain.
7. Add or copy `src\main.c`, `src\radar_accelerator.c`, and
   `src\radar_fixed_point.c` into the Application Component. Add
   `deployment\kv260_baremetal\include` to the compiler include directories.
   Do not add the HLS C++ kernel or weights to the application.
8. Enable C99 (or GNU99) and normal warnings. Build the application and resolve
   all warnings. Float printf support is unnecessary.
9. Connect KV260 USB/JTAG and UART. Create a **Hardware Launch** configuration
   for the application ELF and the correct hardware target. Select programming
   of the FPGA and the platform's PS initialization sequence.
10. Program the FPGA, download the ELF to `psu_cortexa53_0`, and run it.
11. Open the Vitis serial terminal/console for the KV260 UART using the board's
    configured baud rate. Save the complete correctness and benchmark output.

If the driver is absent, do not create names manually: confirm the packaged IP
is in the XSA address map and regenerate the platform/BSP.
