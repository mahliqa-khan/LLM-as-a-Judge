#!/usr/bin/env python3
"""
Helper script to convert plain text paragraphs + scores into JSON format.

This makes it easy to prepare your evaluation data.
"""

import json
import sys


def convert_text_to_json():
    """Interactive script to convert text data to JSON format."""
    print("="*80)
    print("TEXT TO JSON CONVERTER")
    print("="*80)
    print("\nThis tool helps you convert your text data into JSON format for evaluation.\n")

    samples = []
    sample_id = 1

    print("INSTRUCTIONS:")
    print("1. For each sample, you'll paste the model output text")
    print("2. Then enter the human score")
    print("3. Then enter the GPT-5 score")
    print("4. Type 'DONE' when you've entered all samples\n")
    print("="*80)

    while True:
        print(f"\n--- Sample {sample_id} ---")

        # Check if user wants to finish
        check = input("Enter 'DONE' to finish, or press Enter to add sample: ").strip()
        if check.upper() == 'DONE':
            break

        # Get model output
        print(f"\nPaste the model output text (press Enter twice when done):")
        lines = []
        empty_count = 0
        while empty_count < 2:
            line = input()
            if line.strip() == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)

        model_output = "\n".join(lines).strip()

        if not model_output:
            print("No text entered, skipping...")
            continue

        # Get human score
        while True:
            try:
                human_score = int(input(f"Enter human score (1-10): ").strip())
                if 1 <= human_score <= 10:
                    break
                print("Score must be between 1 and 10")
            except ValueError:
                print("Please enter a valid number")

        # Get GPT-5 score
        while True:
            try:
                gpt5_score = int(input(f"Enter GPT-5 score (1-10): ").strip())
                if 1 <= gpt5_score <= 10:
                    break
                print("Score must be between 1 and 10")
            except ValueError:
                print("Please enter a valid number")

        # Optional notes
        notes = input("Optional notes (press Enter to skip): ").strip()

        # Create sample
        sample = {
            "id": sample_id,
            "model_output": model_output,
            "human_score": human_score,
            "gpt5_score": gpt5_score
        }

        if notes:
            sample["notes"] = notes

        samples.append(sample)
        print(f"✓ Sample {sample_id} added")
        sample_id += 1

    if not samples:
        print("\nNo samples entered. Exiting.")
        return

    # Save to file
    output_file = "evaluation_data.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print("\n" + "="*80)
    print(f"✓ Saved {len(samples)} samples to {output_file}")
    print("="*80)
    print("\nYou can now:")
    print(f"1. Upload {output_file} to the web app")
    print(f"2. Run: python eval_stats.py {output_file}")
    print(f"3. Edit {output_file} to add more samples")


def convert_from_csv():
    """Convert CSV format to JSON."""
    print("\n=== CSV TO JSON CONVERTER ===\n")
    csv_file = input("Enter CSV file path: ").strip()

    try:
        import pandas as pd
        df = pd.read_csv(csv_file)

        # Check required columns
        required = ['model_output', 'human_score', 'gpt5_score']
        if not all(col in df.columns for col in required):
            print(f"Error: CSV must have columns: {required}")
            return

        # Convert to JSON format
        samples = []
        for idx, row in df.iterrows():
            sample = {
                "id": idx + 1,
                "model_output": str(row['model_output']),
                "human_score": int(row['human_score']),
                "gpt5_score": int(row['gpt5_score'])
            }

            if 'notes' in df.columns and pd.notna(row['notes']):
                sample['notes'] = str(row['notes'])

            samples.append(sample)

        output_file = "evaluation_data.json"
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)

        print(f"\n✓ Converted {len(samples)} samples to {output_file}")

    except ImportError:
        print("Error: pandas not installed. Run: pip install pandas")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point."""
    print("\nChoose conversion method:")
    print("1. Interactive text input (enter samples one by one)")
    print("2. Convert from CSV file")
    print("3. Exit")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        convert_text_to_json()
    elif choice == "2":
        convert_from_csv()
    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
