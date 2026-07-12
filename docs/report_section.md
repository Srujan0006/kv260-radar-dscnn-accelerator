# Report Section: KV260-Based 2D CNN Radar Classifier

## Objective

The objective of this implementation is to upgrade a resource-constrained PYNQ-Z2 radar classifier to a Kria KV260-based 2D CNN classifier using 100x100 micro-Doppler spectrogram images. The upgrade is intended to move closer to the reference 2D CNN approach while maintaining edge deployment feasibility.

## Methodology

Each radar input segment is treated as a complex IQ vector containing 1280 samples. The vector is reshaped into five range cells with 256 azimuth/Doppler sweeps per range cell. A representative range signal is selected using center-cell, mean-cell, or maximum-energy-cell selection. Short-time Fourier transform processing is then applied to generate a micro-Doppler time-frequency representation. The resulting power spectrogram is converted to logarithmic scale, resized to 100x100 pixels, and normalized to the range [0, 1].

The 100x100 spectrogram is classified using a compact 2D CNN consisting of convolution, batch normalization, ReLU activation, max pooling, average pooling, and a final fully connected classifier. The selected operators are intentionally compatible with Vitis AI DPU deployment on Kria KV260.

## Implementation Flow

The implementation flow consists of five major stages:

1. Raw IQ dataset loading and superclass label mapping.
2. Spectrogram cache generation.
3. PyTorch 2D CNN training and validation.
4. ONNX export and Vitis AI quantization.
5. KV260 DPU inference using VART runtime.

## Expected Outcome

The expected outcome is a working demonstrator showing radar target classification using spectrogram-based 2D CNN inference. The final output class is one of four categories: Drone, Bird, Human, and Corner Reflector. The performance target is improved accuracy compared with the compact handcrafted-feature 1D CNN baseline, subject to dataset split, preprocessing consistency, and quantization quality.

## Practical Limitation

The implementation does not guarantee paper-level accuracy unless the exact paper preprocessing, class grouping, training setup, and validation procedure are reproduced. The main engineering objective for the one-week deadline is to demonstrate a complete working path from radar IQ data to 2D CNN inference on or for the KV260 platform.
