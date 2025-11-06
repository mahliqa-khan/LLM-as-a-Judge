"""
Example usage of the LLM-as-a-Judge framework.
"""

from llm_judge import LLMJudge


def example_single_evaluation():
    """Example: Evaluate a single text."""
    print("=== Single Text Evaluation ===\n")

    # Initialize judge (uses OpenAI by default)
    judge = LLMJudge(provider="openai")

    # Text to evaluate
    text = """
    Machine learning is a subset of artificial intelligence that enables computers
    to learn from data without being explicitly programmed. It has applications in
    various fields including healthcare, finance, and autonomous vehicles.
    """

    # Evaluate for clarity
    result = judge.judge(
        text=text,
        criteria="clarity and coherence",
        scale="1-10"
    )

    print(f"Criteria: {result.criteria}")
    print(f"Score: {result.score}/10")
    print(f"Explanation: {result.explanation}")
    print()


def example_comparison():
    """Example: Compare two texts."""
    print("=== Text Comparison ===\n")

    judge = LLMJudge(provider="openai")

    text_a = "Python is a programming language that is easy to learn and widely used."
    text_b = "Python is a versatile, high-level programming language known for its readable syntax and extensive ecosystem of libraries, making it ideal for beginners and experts alike."

    result = judge.compare(
        text_a=text_a,
        text_b=text_b,
        criteria="informativeness and clarity"
    )

    print(f"Winner: {result['winner']}")
    print(f"Text A Score: {result['score_a']}/10")
    print(f"Text B Score: {result['score_b']}/10")
    print(f"Explanation: {result['explanation']}")
    print()


def example_batch_evaluation():
    """Example: Evaluate multiple texts."""
    print("=== Batch Evaluation ===\n")

    judge = LLMJudge(provider="openai")

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "AI is transforming industries through automation and data analysis.",
        "Climate change is real."
    ]

    results = judge.batch_judge(
        texts=texts,
        criteria="informativeness",
        scale="1-10"
    )

    for i, result in enumerate(results, 1):
        print(f"Text {i}: {result.score}/10")
        print(f"Explanation: {result.explanation}\n")


def example_with_context():
    """Example: Evaluate with additional context."""
    print("=== Evaluation with Context ===\n")

    judge = LLMJudge(provider="openai")

    text = "Our product increased revenue by 50% last quarter."
    context = "This is a marketing claim that should be evaluated for credibility and specificity."

    result = judge.judge(
        text=text,
        criteria="credibility and specificity",
        scale="1-10",
        context=context
    )

    print(f"Score: {result.score}/10")
    print(f"Explanation: {result.explanation}")
    print()


def example_anthropic():
    """Example: Using Anthropic's Claude instead of OpenAI."""
    print("=== Using Anthropic Claude ===\n")

    # Initialize with Anthropic
    judge = LLMJudge(provider="anthropic")

    text = "Quantum computing leverages quantum mechanics to process information."

    result = judge.judge(
        text=text,
        criteria="technical accuracy",
        scale="1-10"
    )

    print(f"Score: {result.score}/10")
    print(f"Explanation: {result.explanation}")
    print()


if __name__ == "__main__":
    print("LLM-as-a-Judge Examples\n")
    print("=" * 50)
    print()

    # Run examples
    # Note: Make sure to set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment

    try:
        example_single_evaluation()
    except Exception as e:
        print(f"Error in single evaluation: {e}\n")

    try:
        example_comparison()
    except Exception as e:
        print(f"Error in comparison: {e}\n")

    try:
        example_batch_evaluation()
    except Exception as e:
        print(f"Error in batch evaluation: {e}\n")

    try:
        example_with_context()
    except Exception as e:
        print(f"Error in context evaluation: {e}\n")

    # Uncomment to test Anthropic
    # try:
    #     example_anthropic()
    # except Exception as e:
    #     print(f"Error with Anthropic: {e}\n")
