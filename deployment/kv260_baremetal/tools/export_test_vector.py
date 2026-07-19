#!/usr/bin/env python3
"""Convert a raw ap_fixed C-simulation export into bare-metal C headers.

No floating-point conversion occurs here. Both arrays must be raw signed
int16 values extracted directly from ap_fixed<16,6,AP_TRN,AP_WRAP> objects.
"""

import argparse
import pathlib
import sys

INPUT_COUNT = 10000
OUTPUT_COUNT = 4
FRACTIONAL_BITS = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", required=True, type=pathlib.Path)
    parser.add_argument("--raw-logits", required=True, type=pathlib.Path)
    parser.add_argument("--metadata", required=True, type=pathlib.Path)
    parser.add_argument("--output-input-header", required=True, type=pathlib.Path)
    parser.add_argument("--output-logits-header", required=True, type=pathlib.Path)
    return parser.parse_args()


def read_raw(path, expected_count):
    try:
        values = [int(token) for token in path.read_text(encoding="utf-8").split()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read raw int16 file {path}: {exc}") from exc
    if len(values) != expected_count:
        raise ValueError(f"{path}: expected exactly {expected_count} values, found {len(values)}")
    if any(value < -32768 or value > 32767 for value in values):
        raise ValueError(f"{path}: value outside signed int16 range")
    return values


def read_metadata(path):
    try:
        fields = dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines() if line)
        sample_id = fields["sample_id"]
        true_class = int(fields["true_class"])
        predicted_class = int(fields["predicted_class"])
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"invalid sample metadata {path}: {exc}") from exc
    if not 0 <= true_class < OUTPUT_COUNT or not 0 <= predicted_class < OUTPUT_COUNT:
        raise ValueError("metadata classes must be in [0, 3]")
    return sample_id, true_class, predicted_class


def format_array(values):
    return "\n".join(
        "    " + ", ".join(map(str, values[i:i + 16])) + ("," if i + 16 < len(values) else "")
        for i in range(0, len(values), 16)
    )


def main():
    args = parse_args()
    raw_input = read_raw(args.raw_input, INPUT_COUNT)
    logits = read_raw(args.raw_logits, OUTPUT_COUNT)
    sample_id, true_class, exported_prediction = read_metadata(args.metadata)
    computed_prediction = max(range(OUTPUT_COUNT), key=lambda index: logits[index])
    if computed_prediction != exported_prediction:
        raise ValueError("metadata predicted_class does not match raw-logit argmax")

    input_text = f"""#ifndef RADAR_TEST_VECTOR_H
#define RADAR_TEST_VECTOR_H
#include <stdint.h>
#define RADAR_TEST_INPUT_COUNT {INPUT_COUNT}U
#define RADAR_TEST_SAMPLE_ID \"hls_real_sample_{sample_id}\"
#define RADAR_TEST_EXPECTED_CLASS {true_class}
#define RADAR_TEST_FRACTIONAL_BITS {FRACTIONAL_BITS}
#define RADAR_TEST_SCALE (1U << RADAR_TEST_FRACTIONAL_BITS)
#define RADAR_TEST_VECTOR_IS_SYNTHETIC 0
static const int16_t radar_test_input[RADAR_TEST_INPUT_COUNT] = {{
{format_array(raw_input)}
}};
#endif
"""
    logits_text = f"""#ifndef RADAR_EXPECTED_LOGITS_H
#define RADAR_EXPECTED_LOGITS_H
#include <stdint.h>
static const int16_t radar_expected_logits[{OUTPUT_COUNT}] = {{{', '.join(map(str, logits))}}};
static const int radar_expected_predicted_class = {exported_prediction};
#define RADAR_EXPECTED_LOGIT_TOLERANCE_RAW 0U
#endif
"""
    args.output_input_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_logits_header.parent.mkdir(parents=True, exist_ok=True)
    args.output_input_header.write_text(input_text, encoding="utf-8")
    args.output_logits_header.write_text(logits_text, encoding="utf-8")
    print(f"exported sample_id={sample_id}, class={true_class}, prediction={exported_prediction}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
