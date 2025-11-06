#!/usr/bin/env python3
"""
Automatic Answer Scoring Tool

Compares model answers vs human answers using LLM to generate numerical scores.
Perfect for ESG data extraction, Q&A systems, etc.
"""

import json
import csv
import os
import sys
from typing import List, Dict
import time


def score_answer_pair(question: str, human_answer: str, model_answer: str,
                      provider: str = "openai", model: str = None) -> Dict:
    """
    Use an LLM to score how accurate the model answer is compared to human answer.

    Args:
        question: The question being answered
        human_answer: Ground truth answer from human
        model_answer: Answer from your AI model
        provider: "openai" or "anthropic"
        model: Specific model name (optional)

    Returns:
        Dict with 'score' (0-100) and 'explanation'
    """
    if provider == "openai":
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = model or "gpt-4o-mini"
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        model = model or "claude-3-5-haiku-20241022"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    prompt = f"""You are evaluating the accuracy of an AI model's answer compared to a human expert's answer.

Question: {question}

Human Expert Answer (Ground Truth):
{human_answer}

AI Model Answer:
{model_answer}

Rate the AI model's answer on a scale of 0-100 based on:
1. Factual accuracy - Does it contain the same key facts?
2. Completeness - Does it include all important information?
3. Correctness - Is anything wrong or contradictory?

Scoring guide:
- 95-100: Perfect or near-perfect match
- 85-94: Correct with minor differences in wording
- 70-84: Mostly correct, missing some details
- 50-69: Partially correct, significant gaps
- 30-49: Incorrect but related information
- 0-29: Wrong or completely different

Respond in JSON format:
{{
    "score": <number 0-100>,
    "explanation": "<brief explanation of the score>"
}}"""

    try:
        if provider == "openai":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            result_text = response.choices[0].message.content
        else:  # anthropic
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            result_text = response.content[0].text

        # Parse JSON response
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        json_str = result_text[json_start:json_end]
        result = json.loads(json_str)

        return {
            'score': float(result['score']),
            'explanation': result['explanation']
        }

    except Exception as e:
        print(f"Warning: Error scoring answer: {e}")
        return {
            'score': 50.0,
            'explanation': f"Error during scoring: {str(e)}"
        }


def process_csv(csv_path: str, provider: str = "openai") -> List[Dict]:
    """
    Process CSV file with questions and answers, generate scores.

    Expected CSV columns:
    - question (or Question)
    - kpi (or KPI) - optional
    - human_answer (or Human Answer)
    - model_answer (or Model Answer)

    Returns:
        List of scored samples in JSON format
    """
    samples = []

    print(f"Reading CSV from {csv_path}...")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} rows")
    print(f"\nUsing {provider} to score answers...")
    print("This may take a few minutes...\n")

    for idx, row in enumerate(rows, 1):
        # Handle different column name formats
        question = row.get('question') or row.get('Question') or row.get('QUESTION') or ''
        kpi = row.get('kpi') or row.get('KPI') or row.get('Kpi') or ''
        human_answer = row.get('human_answer') or row.get('Human Answer') or row.get('HUMAN_ANSWER') or ''
        model_answer = row.get('model_answer') or row.get('Model Answer') or row.get('MODEL_ANSWER') or ''

        if not question or not human_answer or not model_answer:
            print(f"⚠️  Skipping row {idx}: Missing required fields")
            continue

        print(f"[{idx}/{len(rows)}] Scoring: {question[:60]}...")

        # Score the answer pair
        result = score_answer_pair(question, human_answer, model_answer, provider)

        # Create sample in benchmark format
        sample = {
            "id": idx,
            "model_output": f"Q: {question}\nKPI: {kpi}\nAnswer: {model_answer[:200]}..." if len(model_answer) > 200 else f"Q: {question}\nKPI: {kpi}\nAnswer: {model_answer}",
            "human_score": 100,  # Human answer is the ground truth (100%)
            "gpt5_score": int(round(result['score'])),  # Model's score
            "notes": f"Scoring: {result['explanation']}"
        }

        samples.append(sample)

        # Rate limiting
        time.sleep(0.5)

    return samples


def main():
    """Main entry point."""
    print("="*80)
    print("AUTOMATIC ANSWER SCORING TOOL")
    print("="*80)
    print("\nThis tool automatically scores your model's answers against human answers")
    print("using an LLM judge, then outputs JSON for the benchmark tool.\n")

    # Get CSV file path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("Enter path to your CSV file: ").strip()

    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found: {csv_path}")
        sys.exit(1)

    # Choose provider
    print("\nChoose LLM provider for scoring:")
    print("1. OpenAI (gpt-4o-mini) - Fast and cheap")
    print("2. Anthropic (Claude Haiku) - Alternative option")
    choice = input("Enter choice (1 or 2) [default: 1]: ").strip() or "1"

    provider = "openai" if choice == "1" else "anthropic"

    # Check API key
    api_key_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(api_key_name):
        print(f"\n❌ Error: {api_key_name} not set")
        print(f"Set it with: export {api_key_name}='your-key-here'")
        sys.exit(1)

    try:
        # Process the CSV
        samples = process_csv(csv_path, provider)

        if not samples:
            print("\n❌ No valid samples found in CSV")
            sys.exit(1)

        # Save to JSON
        output_file = "scored_evaluation_data.json"
        with open(output_file, 'w') as f:
            json.dump(samples, f, indent=2)

        print("\n" + "="*80)
        print("✅ SCORING COMPLETE!")
        print("="*80)
        print(f"\nProcessed {len(samples)} samples")
        print(f"Saved to: {output_file}")

        # Show statistics
        scores = [s['gpt5_score'] for s in samples]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        print(f"\nModel Performance Preview:")
        print(f"  Average Score: {avg_score:.1f}/100")
        print(f"  Range: {min_score}-{max_score}")
        print(f"  Total Samples: {len(samples)}")

        print(f"\nNext steps:")
        print(f"1. Upload {output_file} to the web app")
        print(f"2. Or run: python eval_stats.py {output_file}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
