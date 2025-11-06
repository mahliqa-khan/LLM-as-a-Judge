"""
Example: Using the PDF Verifier to check extracted information.
"""

from pdf_verifier import PDFVerifier


def example_basic_verification():
    """Basic example of verifying extractions against a PDF."""
    print("=== PDF Extraction Verification Example ===\n")

    # Initialize the verifier
    verifier = PDFVerifier(provider="openai")

    # Load your PDF file
    pdf_path = "your_document.pdf"  # Replace with your PDF path
    print(f"Loading PDF: {pdf_path}")
    verifier.load_pdf(pdf_path)

    # Example: Verify a single claim
    print("\n--- Verifying Single Claim ---")
    claim = "The company revenue increased by 50% in Q4 2023"
    result = verifier.verify_claim(claim)

    print(f"Claim: {result.claim}")
    print(f"Verified: {result.is_verified}")
    print(f"Confidence: {result.confidence}/10")
    print(f"Explanation: {result.explanation}")
    print(f"Relevant Pages: {result.page_numbers}")

    # Example: Verify multiple claims from model output
    print("\n--- Verifying Multiple Claims ---")

    # This is what you'd copy-paste from your extraction model
    extracted_data = """
    Company Name: TechCorp Inc.
    CEO: Jane Smith
    Revenue: $1.5M in Q4 2023
    Employee Count: 250 employees
    Headquarters: San Francisco, CA
    """

    results = verifier.verify_extraction(extracted_data)

    # Print detailed report
    verifier.print_verification_report(results)


def example_json_format():
    """Example with JSON formatted extraction."""
    print("\n=== JSON Format Example ===\n")

    verifier = PDFVerifier(provider="anthropic")  # Using Claude
    verifier.load_pdf("your_document.pdf")

    # JSON formatted extraction
    extracted_json = """{
        "company_name": "TechCorp Inc.",
        "ceo": "Jane Smith",
        "revenue": "$1.5M",
        "employees": "250",
        "location": "San Francisco"
    }"""

    results = verifier.verify_extraction(extracted_json)
    verifier.print_verification_report(results)


def example_with_context():
    """Example showing how context can help verification."""
    print("\n=== Verification with Context ===\n")

    verifier = PDFVerifier(provider="openai")
    verifier.load_pdf("financial_report.pdf")

    # Verify with context about what we're checking
    claim = "Q4 revenue was $1.5M"
    result = verifier.verify_claim(
        claim=claim,
        context="This is from the quarterly financial summary section"
    )

    print(f"Verified: {result.is_verified}")
    print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    print("PDF Verifier Examples")
    print("=" * 80)
    print("\nNote: Make sure to:")
    print("1. Set your API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    print("2. Replace 'your_document.pdf' with your actual PDF path")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("\n" + "=" * 80 + "\n")

    try:
        # Run examples (uncomment the one you want to try)

        # example_basic_verification()
        # example_json_format()
        # example_with_context()

        print("\nUncomment the example functions to run them!")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("- Set your API key in environment variables")
        print("- Installed dependencies: pip install -r requirements.txt")
        print("- Updated the PDF path to point to your actual file")
