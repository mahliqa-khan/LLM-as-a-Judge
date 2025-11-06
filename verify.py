#!/usr/bin/env python3
"""
Simple CLI tool to verify model extractions against source PDFs.

Usage:
    python verify.py

Then follow the prompts to paste your PDF path and extracted data.
"""

import sys
from pdf_verifier import PDFVerifier


def main():
    print("=" * 80)
    print("PDF EXTRACTION VERIFIER")
    print("=" * 80)
    print()
    print("This tool verifies extracted information against the source PDF.")
    print()

    # Get PDF path
    print("Step 1: Enter the path to your PDF file")
    pdf_path = input("PDF Path: ").strip()

    if not pdf_path:
        print("Error: PDF path is required")
        sys.exit(1)

    # Get provider choice
    print("\nStep 2: Choose LLM provider")
    print("1. OpenAI GPT-4o-mini (cheapest - ~$0.15/$0.60 per 1M tokens)")
    print("2. Anthropic Claude Haiku (~$0.25/$1.25 per 1M tokens)")
    provider_choice = input("Enter choice (1 or 2) [default: 1]: ").strip() or "1"

    provider = "openai" if provider_choice == "1" else "anthropic"
    model_name = "gpt-4o-mini" if provider == "openai" else "claude-3-5-haiku-20241022"

    # Initialize verifier
    print(f"\nInitializing verifier with {provider} ({model_name})...")
    try:
        verifier = PDFVerifier(provider=provider)
    except Exception as e:
        print(f"Error initializing verifier: {e}")
        print("\nMake sure you have set your API key:")
        print("  export OPENAI_API_KEY='your-key'  (for OpenAI)")
        print("  export ANTHROPIC_API_KEY='your-key'  (for Anthropic)")
        sys.exit(1)

    # Load PDF
    print(f"\nLoading PDF: {pdf_path}")
    try:
        verifier.load_pdf(pdf_path)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        sys.exit(1)

    # Get extracted data
    print("\n" + "=" * 80)
    print("Step 3: Paste your model's extracted output below")
    print("=" * 80)
    print("You can paste:")
    print("  - Plain text (one claim per line)")
    print("  - JSON format")
    print("  - Bullet points")
    print("\nPaste your extraction below, then press Enter and type 'END' on a new line:")
    print()

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    extracted_data = "\n".join(lines)

    if not extracted_data.strip():
        print("\nError: No extraction data provided")
        sys.exit(1)

    # Verify the extraction
    print("\n" + "=" * 80)
    print("VERIFYING EXTRACTION...")
    print("=" * 80)
    print()

    try:
        results = verifier.verify_extraction(extracted_data)

        if not results:
            print("No claims found to verify. Please check your input format.")
            sys.exit(1)

        # Print the report
        verifier.print_verification_report(results)

        # Save results to file
        save_choice = input("\nWould you like to save the results to a file? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_file = input("Enter filename [default: verification_results.txt]: ").strip() or "verification_results.txt"

            with open(output_file, 'w') as f:
                f.write("VERIFICATION REPORT\n")
                f.write("=" * 80 + "\n\n")

                verified_count = sum(1 for r in results if r.is_verified)
                total_count = len(results)

                f.write(f"Total Claims: {total_count}\n")
                f.write(f"Verified: {verified_count}\n")
                f.write(f"Not Verified: {total_count - verified_count}\n")
                f.write(f"Success Rate: {(verified_count/total_count*100):.1f}%\n\n")

                f.write("=" * 80 + "\n")
                f.write("DETAILED RESULTS\n")
                f.write("=" * 80 + "\n\n")

                for i, result in enumerate(results, 1):
                    status = "VERIFIED" if result.is_verified else "NOT VERIFIED"
                    f.write(f"{i}. {status} (Confidence: {result.confidence:.1f}/10)\n")
                    f.write(f"   Claim: {result.claim}\n")
                    f.write(f"   Explanation: {result.explanation}\n")
                    if result.page_numbers:
                        f.write(f"   Relevant Pages: {', '.join(map(str, result.page_numbers))}\n")
                    f.write("\n")

            print(f"\nResults saved to {output_file}")

    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
