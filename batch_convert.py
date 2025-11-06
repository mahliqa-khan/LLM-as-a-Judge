#!/usr/bin/env python3
"""
Quick batch converter - paste all your data at once in a specific format.

Format:
===1===
TEXT: your model output here
HUMAN: 8
GPT5: 7
NOTES: optional notes

===2===
TEXT: another output
HUMAN: 9
GPT5: 9
"""

import json
import re


def parse_batch_format(text):
    """Parse the batch format into JSON."""
    samples = []

    # Split by sample markers
    pattern = r'===(\d+)==='
    sections = re.split(pattern, text)

    # Process each sample (skip first empty element)
    for i in range(1, len(sections), 2):
        sample_id = int(sections[i])
        content = sections[i + 1].strip()

        # Extract fields
        text_match = re.search(r'TEXT:\s*(.+?)(?=\nHUMAN:|$)', content, re.DOTALL)
        human_match = re.search(r'HUMAN:\s*(\d+)', content)
        gpt5_match = re.search(r'GPT5:\s*(\d+)', content)
        notes_match = re.search(r'NOTES:\s*(.+?)(?=\n===|$)', content, re.DOTALL)

        if text_match and human_match and gpt5_match:
            sample = {
                "id": sample_id,
                "model_output": text_match.group(1).strip(),
                "human_score": int(human_match.group(1)),
                "gpt5_score": int(gpt5_match.group(1))
            }

            if notes_match:
                sample["notes"] = notes_match.group(1).strip()

            samples.append(sample)

    return samples


def main():
    print("="*80)
    print("BATCH TEXT TO JSON CONVERTER")
    print("="*80)
    print("\nPaste your data in this format:")
    print("""
===1===
TEXT: your model output here
HUMAN: 8
GPT5: 7

===2===
TEXT: another output
HUMAN: 9
GPT5: 9
NOTES: optional notes
    """)
    print("\nWhen done, type 'END' on a new line and press Enter.\n")
    print("="*80)

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    text = "\n".join(lines)

    if not text.strip():
        print("No data entered. Exiting.")
        return

    # Parse the data
    samples = parse_batch_format(text)

    if not samples:
        print("\n✗ No valid samples found. Check your format.")
        return

    # Save to JSON
    output_file = "evaluation_data.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"\n✓ Successfully converted {len(samples)} samples!")
    print(f"✓ Saved to: {output_file}")
    print("\nYou can now upload this file to the web app or run:")
    print(f"  python eval_stats.py {output_file}")


if __name__ == "__main__":
    main()
