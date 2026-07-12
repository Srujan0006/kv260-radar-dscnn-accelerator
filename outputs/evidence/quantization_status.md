# Quantization evidence

- Quantized xmodel: `outputs/real/vai_quant_dpu_stride32/Radar2DCNN_int.xmodel`
- Quantizer configuration: `outputs/real/vai_quant_dpu_stride32/quant_info.json`
- Bias correction: `outputs/real/vai_quant_dpu_stride32/bias_corr.pth`
- Compiled KV260 model: `outputs/real/compiled_kv260_dpu/radar_2dcnn.xmodel`
- Compile target: `DPUCZDX8G_ISA1_B4096`

The existing artifacts prove quantization and compilation completed. A fresh
console log can be captured when `run_quantization.sh` is rerun inside the Vitis
AI PyTorch environment.
