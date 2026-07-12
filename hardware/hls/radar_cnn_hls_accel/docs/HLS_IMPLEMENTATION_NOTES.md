# HLS Implementation Notes

## Why use the selected DS-CNN model?
The selected model uses depthwise 3x3, pointwise 1x1, BatchNorm, ReLU,
MaxPool, global average pooling, and a 1x1 convolution classifier. BatchNorm is
folded into convolution weights for inference. This graph is materially smaller
than regular convolutions and maps directly to readable HLS loops.

## Hardware Accelerator Boundary
This HLS IP accelerates only the CNN inference after spectrogram generation. The flow is:

```text
Raw IQ -> spectrogram generation on CPU/PS -> HLS CNN accelerator -> logits -> class decision
```

Do not put STFT/FFT into this HLS block for the first version. That would significantly increase schedule risk.

## Fixed-Point Format
Default:

```cpp
typedef ap_fixed<16, 6> data_t;
typedef ap_fixed<32, 12> acc_t;
```

This is conservative. If accuracy is good, reduce to `ap_fixed<12,4>` or `ap_fixed<10,4>` after validation.

## Validation Order
1. Generate trained `weights.h`.
2. Run the host-float equivalence check against PyTorch.
3. Run C simulation with real fixed-point weights.
4. Compare fixed-point HLS logits with PyTorch on several spectrograms.
5. Run C synthesis and inspect timing/resource reports.
6. Export IP.
7. Integrate it with the ZynqMP PS in Vivado.
8. Test hardware through memory-mapped DDR buffers.

## Expected Bottlenecks
The direct nested-loop implementation has high latency. It is deliberately readable first. Optimize only after functional validation.
