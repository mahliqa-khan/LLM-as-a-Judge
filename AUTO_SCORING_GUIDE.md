# Automatic Answer Scoring Guide

For when your data is **text answers** (not numerical scores yet).

## Use Case

You have:
- ✅ Questions
- ✅ Human answers (ground truth)
- ✅ Model answers (AI-generated)
- ❌ No numerical scores yet

This tool **automatically scores** your model's answers by comparing them to human answers using an LLM judge.

## Quick Start

### Step 1: Export from Numbers to CSV

1. Open your Numbers spreadsheet
2. Make sure you have these columns:
   - `question` - The question being answered
   - `kpi` - KPI category (optional)
   - `human_answer` - Ground truth answer
   - `model_answer` - Your AI model's answer

3. Go to **File → Export To → CSV**
4. Save as `my_data.csv`

### Step 2: Run the Scoring Tool

```bash
python auto_score_answers.py my_data.csv
```

**Or without arguments:**
```bash
python auto_score_answers.py
# Then enter your CSV path when prompted
```

### Step 3: Choose LLM Provider

- **Option 1:** OpenAI (gpt-4o-mini) - Fast, cheap, recommended
- **Option 2:** Anthropic (Claude Haiku) - Alternative

### Step 4: Wait for Scoring

The tool will:
- ✅ Read your CSV
- ✅ Score each answer pair using LLM
- ✅ Generate `scored_evaluation_data.json`
- ✅ Show you preview statistics

**Time:** ~5-10 seconds per answer (for 100 answers = ~10 minutes)

### Step 5: Upload to Benchmark Tool

Upload `scored_evaluation_data.json` to your web app!

## CSV Format Requirements

Your CSV **must have** these columns (case-insensitive):

### Required:
- `question` - The question
- `human_answer` - Human's answer (ground truth)
- `model_answer` - Your model's answer

### Optional:
- `kpi` - KPI category for context

### Example CSV:

```csv
question,kpi,human_answer,model_answer
How does the company assess materiality?,Sustainability,Results on page 30 of ESG report,Assessment results found on page 30
What are emission targets?,Climate,Net-zero by 2050 with 30% by 2030,Net-zero 2050 goal with interim 30% target
```

## Template Files

**Blank template:**
```bash
answer_scoring_template.csv
```

**Sample ESG data:**
```bash
sample_esg_data.csv  # 10 realistic examples
```

## How Scoring Works

The LLM judge evaluates each model answer on:

1. **Factual Accuracy** - Same key facts as human answer?
2. **Completeness** - All important information included?
3. **Correctness** - No wrong or contradictory info?

### Scoring Scale:
- **95-100:** Perfect or near-perfect match
- **85-94:** Correct with minor wording differences
- **70-84:** Mostly correct, missing some details
- **50-69:** Partially correct, significant gaps
- **30-49:** Incorrect but related information
- **0-29:** Wrong or completely different

## Output Format

The tool creates `scored_evaluation_data.json`:

```json
[
  {
    "id": 1,
    "model_output": "Q: How does company assess materiality?\nAnswer: ...",
    "human_score": 100,
    "gpt5_score": 92,
    "notes": "Scoring: Correct info with minor wording differences"
  }
]
```

**Note:** `human_score` is always 100 (perfect/ground truth)
`gpt5_score` is your model's score (0-100)

## Example Usage

### From Numbers → Web App

```bash
# 1. Export Numbers to CSV
# File → Export To → CSV → Save as "esg_data.csv"

# 2. Score the answers
python auto_score_answers.py esg_data.csv

# 3. Wait for completion
# ✅ Creates: scored_evaluation_data.json

# 4. Upload to web app
# Or run: python eval_stats.py scored_evaluation_data.json
```

## Cost Estimate

Using OpenAI gpt-4o-mini:
- **~$0.0001 per answer pair**
- **100 answers ≈ $0.01** (1 cent!)
- **1000 answers ≈ $0.10** (10 cents!)

Very cheap! 💰

## Troubleshooting

### "API key not set"
```bash
export OPENAI_API_KEY='your-key-here'
# OR
export ANTHROPIC_API_KEY='your-key-here'
```

### "File not found"
Make sure CSV path is correct. Use full path if needed:
```bash
python auto_score_answers.py /path/to/my_data.csv
```

### "Missing required fields"
Check your CSV has columns: `question`, `human_answer`, `model_answer`

### "Slow scoring"
- Each answer takes ~5-10 seconds
- For 100 answers, expect ~10 minutes
- Grab a coffee! ☕

## Tips

✅ **Clean your data first** - Remove empty rows
✅ **Use clear column names** - question, human_answer, model_answer
✅ **Keep answers focused** - Long answers work but take longer
✅ **Review sample data** - Check `sample_esg_data.csv` for format

## What Happens Next

After scoring:

1. **Review the scores** - Check if they make sense
2. **Upload to web app** - Get statistical analysis
3. **See benchmarks:**
   - Correlation between model and human
   - Average accuracy
   - Error patterns
   - Best/worst predictions

## Questions?

Check the scoring for a few samples manually to verify it's working correctly!
