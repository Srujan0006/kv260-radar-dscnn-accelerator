# Hardware tensor contract

This contract is frozen with `6_hardware/hls/radar_cnn_hls_accel/golden/`.

| Property | Contract |
|---|---|
| Input shape | `1 x 100 x 100` (`C x H x W`) |
| External input order | Row-major: `input[y * 100 + x]`; the single channel is implicit |
| Internal stream order | Pixel-major HWC tokens: `(y,x)` first, then every channel in `Pixel<C>::channel[]` |
| Input numeric type | V1 baseline `ap_fixed<16,6,AP_TRN,AP_WRAP>`; signed, 10 fractional bits |
| Input range | Normalized host `float32` nominally in `[0,1]`, cast to `data_t` before transfer |
| Output shape | Four signed logits |
| Output order | Drone, Bird, Human, CornerReflector |
| Output numeric type | V1 baseline `ap_fixed<16,6,AP_TRN,AP_WRAP>` |

## Exact preprocessing

For each `5 x 256` complex IQ window, rank the five range bins by mean squared
magnitude and select the top three. For each selected bin: remove its complex
mean, compute a Hann STFT (`nperseg=64`, `noverlap=56`, `nfft=128`, no boundary
extension or padding, two-sided), FFT-shift, take magnitude squared, and apply
`10*log10(value + 1e-12)`. Bilinearly resize to `100 x 100`, subtract that
image's minimum, and divide by its shifted maximum plus `1e-8`. Average the
three independently normalized images using `float32` arithmetic.

Convolutions use zero padding outside the image. V1 and V2 must preserve the
same accumulation, cast, activation, folded-weight, and class ordering.
