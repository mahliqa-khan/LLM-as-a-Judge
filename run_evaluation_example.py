"""
Example: How to run the GPT-5 evaluation against human labels.
"""

from eval_stats import LLMJudgeEvaluator


def main():
    """Run evaluation on your data."""

    # 1. Make sure your data file is ready
    # Use the evaluation_data_template.json as a starting point
    data_file = "evaluation_data.json"  # Change to your actual data file

    print("="*80)
    print("GPT-5 JUDGE EVALUATION")
    print("="*80)
    print()

    # 2. Load the data and run evaluation
    print(f"Loading data from: {data_file}")
    try:
        evaluator = LLMJudgeEvaluator(data_file)
        print(f"✓ Successfully loaded {len(evaluator.data)} samples\n")
    except FileNotFoundError:
        print(f"✗ Error: Could not find '{data_file}'")
        print("\nPlease:")
        print("1. Copy 'evaluation_data_template.json' to 'evaluation_data.json'")
        print("2. Fill in your actual data (90-100 samples)")
        print("3. Run this script again")
        return
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return

    # 3. Print comprehensive report
    evaluator.print_report(detailed=True)

    # 4. Save results to JSON file
    output_file = "evaluation_report.json"
    evaluator.save_report(output_file)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"• Detailed metrics saved to: {output_file}")
    print("• Review the worst predictions to understand where GPT-5 struggles")
    print("• Check correlation metrics to assess overall alignment with humans")
    print("• MAE < 1.0 is excellent, < 1.5 is good, < 2.0 is acceptable")


if __name__ == "__main__":
    main()
