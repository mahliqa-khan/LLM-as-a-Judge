
# LLM-as-a-Judge

A simple, flexible framework for using Large Language Models to evaluate and judge text quality based on custom criteria.

## 🌐 Web App Available!

**Try it now:** Deploy your own evaluation dashboard in minutes!

```bash
streamlit run app.py
```

Then visit `http://localhost:8501` to:
- 📤 Upload your evaluation data
- 📊 View interactive charts and statistics
- 📈 Analyze GPT-5 judge accuracy vs human labels
- 📥 Download detailed reports

**[See Deployment Guide →](DEPLOYMENT.md)** for deploying to Streamlit Cloud (free!)

## Features

### LLM Judging
- **Single Text Evaluation**: Judge individual texts based on any criteria
- **Text Comparison**: Compare two texts and determine which is better
- **Batch Evaluation**: Evaluate multiple texts efficiently
- **Multi-Provider Support**: Works with OpenAI (GPT-4) and Anthropic (Claude)
- **Flexible Criteria**: Define custom evaluation criteria
- **Structured Output**: Get scores and detailed explanations

### PDF Verification
- **PDF Extraction Verification**: Verify extracted information against source PDFs
- **Interactive CLI Tool**: Easy copy-paste verification workflow

### Evaluation & Analytics
- **Statistical Analysis**: Compare LLM judges against human ground truth
- **Accuracy Metrics**: Exact match, within-N accuracy, MAE, RMSE
- **Correlation Analysis**: Pearson and Spearman correlations
- **Error Breakdown**: Detailed analysis of prediction errors
- **Interactive Web Dashboard**: Beautiful Streamlit interface with charts

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
```bash
# For OpenAI
export OPENAI_API_KEY='your-api-key-here'

# For Anthropic
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Quick Start

```python
from llm_judge import LLMJudge

# Initialize the judge
judge = LLMJudge(provider="openai")

# Evaluate a text
text = "Machine learning enables computers to learn from data."
result = judge.judge(
    text=text,
    criteria="clarity and informativeness",
    scale="1-10"
)

print(f"Score: {result.score}/10")
print(f"Explanation: {result.explanation}")
```

## PDF Extraction Verification

### Quick Start - Copy & Paste Verification

The easiest way to verify extractions against a PDF:

```bash
python verify.py
```

Then follow the prompts:
1. Enter your PDF file path
2. Choose your LLM provider (OpenAI or Anthropic)
3. Paste your model's extracted output
4. Type `END` on a new line when done

The tool will automatically:
- Find relevant sections in the PDF for each claim
- Verify each claim against the source
- Provide a detailed verification report with confidence scores
- Show which pages contain supporting evidence

**Example Output:**
```
VERIFICATION REPORT
================================================================================

Total Claims: 5
Verified: 4
Not Verified: 1
Success Rate: 80.0%

DETAILED RESULTS
================================================================================

1. ✓ VERIFIED (Confidence: 9.0/10)
   Claim: Company revenue increased by 50% in Q4 2023
   Explanation: This claim is directly supported by the financial data on page 12
   Relevant Pages: 12, 13

2. ✗ NOT VERIFIED (Confidence: 2.0/10)
   Claim: The CEO is John Smith
   Explanation: No mention of this name found in the relevant sections
   Relevant Pages: 5, 6
```

### Programmatic Usage

You can also use the PDF verifier in your own code:

```python
from pdf_verifier import PDFVerifier

# Initialize the verifier
verifier = PDFVerifier(provider="openai")

# Load your PDF
verifier.load_pdf("document.pdf")

# Verify a single claim
result = verifier.verify_claim("The revenue was $1.5M in Q4 2023")

print(f"Verified: {result.is_verified}")
print(f"Confidence: {result.confidence}/10")
print(f"Explanation: {result.explanation}")
print(f"Found on pages: {result.page_numbers}")

# Or verify multiple claims at once
extracted_data = """
Revenue: $1.5M in Q4 2023
CEO: John Smith
Employees: 250 people
Location: San Francisco, CA
"""

results = verifier.verify_extraction(extracted_data)
verifier.print_verification_report(results)
```

### Supported Input Formats

The verifier accepts various formats:

**Plain text (one claim per line):**
```
Company revenue was $1.5M
CEO is John Smith
Office located in San Francisco
```

**JSON format:**
```json
{
  "revenue": "$1.5M in Q4 2023",
  "ceo": "John Smith",
  "location": "San Francisco, CA"
}
```

**Bullet points:**
```
- Revenue: $1.5M
- CEO: John Smith
- Location: San Francisco
```

## Usage Examples

### Single Text Evaluation

```python
from llm_judge import LLMJudge

judge = LLMJudge(provider="openai")

result = judge.judge(
    text="Your text here",
    criteria="helpfulness",
    scale="1-10"
)

print(f"Score: {result.score}")
print(f"Explanation: {result.explanation}")
```

### Comparing Two Texts

```python
result = judge.compare(
    text_a="First response",
    text_b="Second response",
    criteria="accuracy and completeness"
)

print(f"Winner: {result['winner']}")
print(f"Score A: {result['score_a']}")
print(f"Score B: {result['score_b']}")
```

### Batch Evaluation

```python
texts = [
    "Text 1",
    "Text 2",
    "Text 3"
]

results = judge.batch_judge(
    texts=texts,
    criteria="coherence",
    scale="1-10"
)

for i, result in enumerate(results):
    print(f"Text {i+1}: {result.score}/10")
```

### With Context

```python
result = judge.judge(
    text="Our product is the best!",
    criteria="credibility",
    context="This is a marketing claim",
    scale="1-10"
)
```

### Using Different Providers

```python
# OpenAI (default)
judge_openai = LLMJudge(provider="openai", model="gpt-4o")

# Anthropic Claude
judge_claude = LLMJudge(provider="anthropic", model="claude-3-5-sonnet-20241022")
```

## API Reference

### LLMJudge

**Constructor Parameters:**
- `provider` (str): API provider - "openai" or "anthropic"
- `model` (str, optional): Specific model name
- `api_key` (str, optional): API key (uses environment variable if not provided)

**Methods:**

#### `judge(text, criteria, scale="1-10", context=None, reference=None)`
Evaluate a single text.

**Parameters:**
- `text` (str): Text to evaluate
- `criteria` (str): Evaluation criteria (e.g., "clarity", "helpfulness")
- `scale` (str): Scoring scale (default "1-10")
- `context` (str, optional): Additional context for evaluation
- `reference` (str, optional): Reference text for comparison

**Returns:** `JudgmentResult` with `score`, `explanation`, `criteria`, and `raw_response`

#### `compare(text_a, text_b, criteria, context=None)`
Compare two texts.

**Parameters:**
- `text_a` (str): First text
- `text_b` (str): Second text
- `criteria` (str): Comparison criteria
- `context` (str, optional): Additional context

**Returns:** Dictionary with `winner`, `score_a`, `score_b`, and `explanation`

#### `batch_judge(texts, criteria, scale="1-10", context=None)`
Evaluate multiple texts.

**Parameters:**
- `texts` (List[str]): List of texts to evaluate
- `criteria` (str): Evaluation criteria
- `scale` (str): Scoring scale
- `context` (str, optional): Additional context

**Returns:** List of `JudgmentResult` objects

## Common Use Cases

### Evaluating Model Outputs
```python
judge = LLMJudge()

model_output = "Response from your AI model"
result = judge.judge(
    text=model_output,
    criteria="helpfulness and accuracy",
    context="User asked: How does photosynthesis work?"
)
```

### Content Quality Assessment
```python
result = judge.judge(
    text=article_text,
    criteria="readability and engagement",
    scale="1-10"
)
```

### Choosing Best Response
```python
result = judge.compare(
    text_a=response_1,
    text_b=response_2,
    criteria="relevance and completeness"
)
```

## Running Examples

Run the example script to see all features in action:

```bash
python example_usage.py
```

## Evaluating Judge Accuracy

### Statistical Evaluation Framework

Compare GPT-5 (or any LLM judge) accuracy against human ground truth:

**Quick Start:**
```bash
# Run the web app
streamlit run app.py

# Or use the CLI
python eval_stats.py evaluation_data.json
```

**Data Format:**
```json
[
  {
    "id": 1,
    "model_output": "Your model's generated text",
    "human_score": 8,
    "gpt5_score": 7
  }
]
```

**What You Get:**
- ✅ Exact match accuracy (% of perfect predictions)
- 📊 Within ±1 and ±2 point accuracy
- 📈 Correlation metrics (Pearson, Spearman)
- 📉 Error analysis (MAE, RMSE, bias detection)
- 🔍 Worst/best predictions analysis
- 📋 Confusion matrix
- 💾 Exportable reports (JSON/text)

**See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for complete documentation.**

## Notes

- The framework uses temperature=0.3 for more consistent evaluations
- Responses are structured as JSON for easy parsing
- Fallback handling if JSON parsing fails
- Both OpenAI and Anthropic models are supported

## License

MIT
