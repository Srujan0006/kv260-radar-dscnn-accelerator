# KV260 bare-metal radar CNN deployment

This package runs the packaged `radar_cnn_stream_full`-class IP from a 64-bit
standalone Cortex-A53 application over AXI4-Lite, with input/output buffers in
DDR accessed through non-coherent HP0_FPD. It uses polling, static aligned
storage, cache maintenance, and no Linux, DMA, allocation, interrupts, or
floating-point printing.

## Prerequisites and hardware contract

The completed Vivado design targets `xck26-sfvc784-2LV-c` and exports an XSA
with bitstream and address map (current system: `v3_wrapper.xsa`, accelerator
`s_axi_control` at `0xA000_0000`, PL clock ~97 MHz from `pl_clk0`, interrupt
wired to `pl_ps_irq0`, data through `S_AXI_HPC0_FPD`). Generate a Vitis
platform for `psu_cortexa53_0`, 64-bit standalone, from that XSA.

The generated driver identifiers for `include/radar_driver_binding.h` are
(from the packaged IP's generated driver, checked in under `driver/radar_cnn_stream_full_v1_0/`):
instance type `XRadar_cnn_stream_full`, device-ID macro
`XPAR_XRADAR_CNN_STREAM_FULL_0_DEVICE_ID`,
`XRadar_cnn_stream_full_Initialize`, `_Set_input_r(u64)`, `_Set_logits(u64)`,
`_Start`, `_IsDone`, `_IsIdle`, plus the `_InterruptGlobalEnable/Disable`,
`_InterruptEnable/Disable/Clear/GetEnabled/GetStatus` family (unused by this
MVP). Register map: 0x00 AP_CTRL, 0x04 GIE, 0x08 IER, 0x0C ISR,
0x10/0x14 input_r, 0x1C/0x20 logits. Verify against the platform's generated
`xparameters.h` after the BSP build. Do not rename driver files.

Add these application sources to Vitis:

- `src/main.c`
- `src/radar_accelerator.c`
- `src/radar_fixed_point.c`

Add `deployment\kv260_baremetal\include` as an include directory. Keep the
Vitis workspace outside Git at `C:\KV260_Radar_Workspace`.

## Fixed-point golden data

Both final HLS headers define `data_t` as
`ap_fixed<16,6,AP_TRN,AP_WRAP>` (10 fractional bits, scale 1024). The checked-in
header is a deterministic sine/cosine sample executed by the V2 Vitis 2025.1
fixed-point C simulation; it is not random. Its saved bit-exact logits are
`[-2703, 4557, -8259, 888]`, predicting class 1. This is an interface smoke test,
not a labeled real-radar accuracy claim.

The eight real samples are in
`reports/fixed_point_validation/input_samples.txt`, but the repository explicitly
states their exact V1 fixed-point logits were not retained. Re-run the real
sample C testbench with `--export-golden <fixed_point_validation-directory>`.
It writes `raw_input_int16.txt`, `raw_logits_int16.txt`, and
`sample_metadata.txt`; the raw values come directly from the `ap_fixed` bit
ranges. Pass those three files to `tools/export_test_vector.py`. The tool does
no Python rounding or floating-point conversion.

## Correctness and timing

Correctness requires all four raw hardware logits to match the fixed-point HLS
reference (zero raw tolerance) and argmax to match the expected prediction.
Terminal output includes raw/decoded actual and expected logits, differences,
class names, `PASS`/`FAIL`, 10 warm-ups, 100 measured runs, min/max/average, and
inferences per second. The CSV-friendly summary is intended for the results
template.

Measured accelerator latency is strictly the interval from the generated start
call through observation of generated done. Address setup, cache operations,
and decoding are excluded. No co-simulation cycle count is presented as board
latency.

## Remaining work

1. Provide/build the final XSA and standalone BSP.
2. Replace adapter TODOs using the generated files.
3. Re-run real-sample fixed-point C simulation with exact logits and export a
   labeled real vector.
4. Build, program, run, and archive genuine UART measurements/screenshots.

See `docs/VITIS_GUI_SETUP.md`, `docs/BOARD_TEST_CHECKLIST.md`, and
`docs/TROUBLESHOOTING.md`.
