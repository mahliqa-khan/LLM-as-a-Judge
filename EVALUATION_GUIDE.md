# GPT-5 Judge Evaluation Guide

This guide explains how to evaluate GPT-5's judging accuracy against human ground truth labels.

## Quick Start

### Step 1: Prepare Your Data

1. Copy the template:
```bash
cp evaluation_data_template.json evaluation_data.json
```

2. Fill in your data in `evaluation_data.json`:
```json
[
  {
    "id": 1,
    "model_output": "Your model's generated text here...",
    "human_score": 8,
    "gpt5_score": 7,
    "notes": "Optional notes"
  },
  {
    "id": 2,
    "model_output": "Another paragraph from your model...",
    "human_score": 9,
    "gpt5_score": 9
  }
]
```

**Required Fields:**
- `id`: Unique identifier (integer)
- `model_output`: The text your model generated (string)
- `human_score`: Human-assigned score (integer, recommend 1-10 scale)
- `gpt5_score`: GPT-5's judgment score (integer, same scale as human)

**Optional Fields:**
- `notes`: Any additional context or notes

### Step 2: Run the Evaluation

**Option A - Using the example script:**
```bash
python run_evaluation_example.py
```

**Option B - Using eval_stats.py directly:**
```bash
python eval_stats.py evaluation_data.json
```

**Option C - In your own code:**
```python
from eval_stats import LLMJudgeEvaluator

evaluator = LLMJudgeEvaluator("evaluation_data.json")
evaluator.print_report(detailed=True)
evaluator.save_report("my_results.json")
```

## Understanding the Metrics

### Accuracy Metrics

**Exact Match Accuracy**
- Percentage of times GPT-5 score = Human score
- 80%+ is excellent, 60-80% is good

**Within ±1 Point Accuracy**
- Percentage within 1 point of human score
- Should be 90%+ for reliable judge

**Within ±2 Points Accuracy**
- Percentage within 2 points
- Should be 95%+ for acceptable judge

### Error Metrics

**Mean Absolute Error (MAE)**
- Average difference between GPT-5 and human scores
- < 1.0 = excellent, < 1.5 = good, < 2.0 = acceptable

**Root Mean Square Error (RMSE)**
- Penalizes large errors more heavily
- Should be similar to MAE if no outliers

**Mean Error (Bias)**
- Positive = GPT-5 overestimates
- Negative = GPT-5 underestimates
- Close to 0 = unbiased

### Correlation Metrics

**Pearson Correlation**
- Measures linear relationship
- 0.9+ = very strong, 0.7-0.9 = strong, 0.5-0.7 = moderate

**Spearman Correlation**
- Measures rank-order relationship
- More robust to outliers
- Similar interpretation to Pearson

### What Makes a Good Judge?

A good LLM judge should have:
- ✓ Pearson correlation > 0.8
- ✓ MAE < 1.5
- ✓ Within ±1 accuracy > 85%
- ✓ Mean error close to 0 (unbiased)

## Example Output

```
================================================================================
GPT-5 JUDGE EVALUATION REPORT
================================================================================

Total Samples: 95

ACCURACY METRICS
--------------------------------------------------------------------------------
Exact Match Accuracy:        42.11% (40/95)
Within ±1 Point Accuracy:    87.37%
Within ±2 Points Accuracy:   95.79%

ERROR METRICS
--------------------------------------------------------------------------------
Mean Absolute Error (MAE):   0.842
Root Mean Square Error:      1.124
Mean Error (Bias):           -0.126 (GPT-5 tends to UNDERESTIMATE)

CORRELATION METRICS
--------------------------------------------------------------------------------
Pearson Correlation:         0.876 (p=0.0000)
Spearman Correlation:        0.891 (p=0.0000)
  → Strong positive correlation
```

## Interpreting Results

### High Correlation, Low Exact Match
- GPT-5 understands quality ranking but uses different scoring scale
- Consider: Does score magnitude matter or just relative ranking?

### Low Correlation
- GPT-5 evaluating different aspects than humans
- Review worst predictions to identify patterns
- May need to refine judging criteria/prompts

### Systematic Bias (Mean Error ≠ 0)
- Consistent over/underestimation
- Can be corrected with simple calibration
- Not as problematic as low correlation

### High MAE with Good Correlation
- GPT-5 gets the ranking right but magnitudes wrong
- Scores are still useful for ranking/comparison
- May need score calibration for absolute values

## Troubleshooting

**"FileNotFoundError"**
- Make sure you created `evaluation_data.json` from the template

**"Missing required field"**
- Each sample must have: id, model_output, human_score, gpt5_score

**Very low correlation (< 0.5)**
- Check if scoring scales match (both 1-10?)
- Review if GPT-5 understands the evaluation criteria
- Look at worst predictions to find patterns

## Advanced Usage

### Get Specific Metrics

```python
evaluator = LLMJudgeEvaluator("evaluation_data.json")

# Calculate all metrics
metrics = evaluator.calculate_metrics()
print(f"MAE: {metrics.mean_absolute_error}")

# Get confusion matrix
confusion = evaluator.get_confusion_matrix()

# Get worst predictions
worst = evaluator.get_worst_predictions(n=10)
for pred in worst:
    print(f"ID {pred['id']}: Human={pred['human_score']}, GPT5={pred['gpt5_score']}")

# Get error breakdown
breakdown = evaluator.get_error_breakdown()
print(f"Perfect matches: {breakdown['perfect_match']}")
```

### Custom Analysis

```python
# Access raw data
evaluator = LLMJudgeEvaluator("evaluation_data.json")
human_scores = evaluator.human_scores
gpt5_scores = evaluator.gpt5_scores

# Your custom analysis
import matplotlib.pyplot as plt
plt.scatter(human_scores, gpt5_scores)
plt.xlabel("Human Scores")
plt.ylabel("GPT-5 Scores")
plt.savefig("correlation_plot.png")
```

## Tips for Better Evaluation

1. **Sample Size**: 90-100 samples is good. More is better for statistical significance.

2. **Diverse Examples**: Include range of quality levels (low, medium, high scores)

3. **Clear Criteria**: Make sure both humans and GPT-5 judge on same criteria

4. **Multiple Human Raters**: If possible, average multiple human ratings for more reliable ground truth

5. **Review Outliers**: Examine worst predictions to understand systematic failures

## Dependencies

```bash
pip install numpy scipy
```

## Questions?

Review the code in `eval_stats.py` - it's well-commented and shows exactly how each metric is calculated.
