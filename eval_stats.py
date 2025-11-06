"""
Statistical Evaluation Framework for LLM Judge Accuracy

Compares GPT-5 judgments against human ground truth labels.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats
from collections import defaultdict


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    # Basic metrics
    exact_match_accuracy: float
    within_1_accuracy: float
    within_2_accuracy: float

    # Error metrics
    mean_absolute_error: float
    root_mean_square_error: float
    mean_error: float  # Positive = GPT-5 overestimates, Negative = underestimates

    # Correlation metrics
    pearson_correlation: float
    pearson_p_value: float
    spearman_correlation: float
    spearman_p_value: float

    # Distribution analysis
    gpt5_mean: float
    human_mean: float
    gpt5_std: float
    human_std: float

    # Detailed breakdowns
    overestimate_count: int
    underestimate_count: int
    exact_match_count: int

    total_samples: int


class LLMJudgeEvaluator:
    """
    Evaluates LLM judge accuracy against human ground truth.
    """

    def __init__(self, data_path: str):
        """
        Initialize evaluator with data file.

        Args:
            data_path: Path to JSON file with evaluation data
        """
        self.data = self._load_data(data_path)
        self.human_scores = np.array([item['human_score'] for item in self.data])
        self.gpt5_scores = np.array([item['gpt5_score'] for item in self.data])

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load evaluation data from JSON file."""
        with open(data_path, 'r') as f:
            data = json.load(f)

        # Validate data
        required_fields = ['id', 'model_output', 'human_score', 'gpt5_score']
        for item in data:
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field '{field}' in item {item.get('id', 'unknown')}")

        return data

    def calculate_metrics(self) -> EvaluationMetrics:
        """Calculate all evaluation metrics."""
        n = len(self.human_scores)

        # Exact match accuracy
        exact_matches = np.sum(self.human_scores == self.gpt5_scores)
        exact_match_accuracy = exact_matches / n

        # Within-threshold accuracies
        within_1 = np.sum(np.abs(self.human_scores - self.gpt5_scores) <= 1)
        within_1_accuracy = within_1 / n

        within_2 = np.sum(np.abs(self.human_scores - self.gpt5_scores) <= 2)
        within_2_accuracy = within_2 / n

        # Error metrics
        errors = self.gpt5_scores - self.human_scores
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        mean_error = np.mean(errors)

        # Correlation metrics
        pearson_r, pearson_p = stats.pearsonr(self.human_scores, self.gpt5_scores)
        spearman_r, spearman_p = stats.spearmanr(self.human_scores, self.gpt5_scores)

        # Distribution stats
        gpt5_mean = np.mean(self.gpt5_scores)
        human_mean = np.mean(self.human_scores)
        gpt5_std = np.std(self.gpt5_scores)
        human_std = np.std(self.human_scores)

        # Over/under estimation
        overestimate_count = np.sum(errors > 0)
        underestimate_count = np.sum(errors < 0)

        return EvaluationMetrics(
            exact_match_accuracy=exact_match_accuracy,
            within_1_accuracy=within_1_accuracy,
            within_2_accuracy=within_2_accuracy,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            mean_error=mean_error,
            pearson_correlation=pearson_r,
            pearson_p_value=pearson_p,
            spearman_correlation=spearman_r,
            spearman_p_value=spearman_p,
            gpt5_mean=gpt5_mean,
            human_mean=human_mean,
            gpt5_std=gpt5_std,
            human_std=human_std,
            overestimate_count=overestimate_count,
            underestimate_count=underestimate_count,
            exact_match_count=exact_matches,
            total_samples=n
        )

    def get_confusion_matrix(self, score_range: Tuple[int, int] = (1, 10)) -> np.ndarray:
        """
        Generate confusion matrix for score predictions.

        Args:
            score_range: (min_score, max_score) tuple

        Returns:
            Confusion matrix as numpy array
        """
        min_score, max_score = score_range
        size = max_score - min_score + 1

        confusion = np.zeros((size, size), dtype=int)

        for human, gpt5 in zip(self.human_scores, self.gpt5_scores):
            h_idx = int(human - min_score)
            g_idx = int(gpt5 - min_score)
            confusion[h_idx, g_idx] += 1

        return confusion

    def get_error_breakdown(self) -> Dict:
        """
        Get detailed breakdown of errors by magnitude.

        Returns:
            Dictionary with error distribution
        """
        errors = self.gpt5_scores - self.human_scores

        breakdown = {
            'perfect_match': np.sum(errors == 0),
            'off_by_1': np.sum(np.abs(errors) == 1),
            'off_by_2': np.sum(np.abs(errors) == 2),
            'off_by_3': np.sum(np.abs(errors) == 3),
            'off_by_4_plus': np.sum(np.abs(errors) >= 4),
            'overestimate_by_1': np.sum(errors == 1),
            'overestimate_by_2': np.sum(errors == 2),
            'overestimate_by_3_plus': np.sum(errors >= 3),
            'underestimate_by_1': np.sum(errors == -1),
            'underestimate_by_2': np.sum(errors == -2),
            'underestimate_by_3_plus': np.sum(errors <= -3),
        }

        return breakdown

    def get_worst_predictions(self, n: int = 10) -> List[Dict]:
        """
        Get the samples with largest prediction errors.

        Args:
            n: Number of worst predictions to return

        Returns:
            List of samples with largest errors
        """
        errors = np.abs(self.gpt5_scores - self.human_scores)
        worst_indices = np.argsort(errors)[::-1][:n]

        worst = []
        for idx in worst_indices:
            worst.append({
                'id': self.data[idx]['id'],
                'model_output': self.data[idx]['model_output'][:200] + '...' if len(self.data[idx]['model_output']) > 200 else self.data[idx]['model_output'],
                'human_score': int(self.human_scores[idx]),
                'gpt5_score': int(self.gpt5_scores[idx]),
                'error': int(self.gpt5_scores[idx] - self.human_scores[idx]),
                'abs_error': int(errors[idx])
            })

        return worst

    def get_best_predictions(self, n: int = 10) -> List[Dict]:
        """
        Get the samples with exact or near-perfect predictions.

        Args:
            n: Number of best predictions to return

        Returns:
            List of samples with smallest errors
        """
        errors = np.abs(self.gpt5_scores - self.human_scores)
        best_indices = np.argsort(errors)[:n]

        best = []
        for idx in best_indices:
            best.append({
                'id': self.data[idx]['id'],
                'model_output': self.data[idx]['model_output'][:200] + '...' if len(self.data[idx]['model_output']) > 200 else self.data[idx]['model_output'],
                'human_score': int(self.human_scores[idx]),
                'gpt5_score': int(self.gpt5_scores[idx]),
                'error': int(self.gpt5_scores[idx] - self.human_scores[idx]),
                'abs_error': int(errors[idx])
            })

        return best

    def print_report(self, detailed: bool = True):
        """
        Print comprehensive evaluation report.

        Args:
            detailed: Whether to include detailed breakdowns
        """
        metrics = self.calculate_metrics()

        print("\n" + "="*80)
        print("GPT-5 JUDGE EVALUATION REPORT")
        print("="*80 + "\n")

        print(f"Total Samples: {metrics.total_samples}\n")

        # Accuracy metrics
        print("ACCURACY METRICS")
        print("-" * 80)
        print(f"Exact Match Accuracy:        {metrics.exact_match_accuracy*100:.2f}% ({metrics.exact_match_count}/{metrics.total_samples})")
        print(f"Within ±1 Point Accuracy:    {metrics.within_1_accuracy*100:.2f}%")
        print(f"Within ±2 Points Accuracy:   {metrics.within_2_accuracy*100:.2f}%")
        print()

        # Error metrics
        print("ERROR METRICS")
        print("-" * 80)
        print(f"Mean Absolute Error (MAE):   {metrics.mean_absolute_error:.3f}")
        print(f"Root Mean Square Error:      {metrics.root_mean_square_error:.3f}")
        print(f"Mean Error (Bias):           {metrics.mean_error:.3f}", end="")
        if metrics.mean_error > 0:
            print(" (GPT-5 tends to OVERESTIMATE)")
        elif metrics.mean_error < 0:
            print(" (GPT-5 tends to UNDERESTIMATE)")
        else:
            print(" (No bias)")
        print()

        # Correlation metrics
        print("CORRELATION METRICS")
        print("-" * 80)
        print(f"Pearson Correlation:         {metrics.pearson_correlation:.3f} (p={metrics.pearson_p_value:.4f})")
        print(f"Spearman Correlation:        {metrics.spearman_correlation:.3f} (p={metrics.spearman_p_value:.4f})")

        if metrics.pearson_correlation >= 0.9:
            print("  → Very strong positive correlation")
        elif metrics.pearson_correlation >= 0.7:
            print("  → Strong positive correlation")
        elif metrics.pearson_correlation >= 0.5:
            print("  → Moderate positive correlation")
        else:
            print("  → Weak correlation - GPT-5 judgments may not align well with humans")
        print()

        # Distribution comparison
        print("SCORE DISTRIBUTIONS")
        print("-" * 80)
        print(f"Human Scores:  Mean={metrics.human_mean:.2f}, Std={metrics.human_std:.2f}")
        print(f"GPT-5 Scores:  Mean={metrics.gpt5_mean:.2f}, Std={metrics.gpt5_std:.2f}")
        print()

        # Over/under estimation
        print("BIAS ANALYSIS")
        print("-" * 80)
        print(f"Exact Matches:      {metrics.exact_match_count} ({metrics.exact_match_count/metrics.total_samples*100:.1f}%)")
        print(f"Overestimates:      {metrics.overestimate_count} ({metrics.overestimate_count/metrics.total_samples*100:.1f}%)")
        print(f"Underestimates:     {metrics.underestimate_count} ({metrics.underestimate_count/metrics.total_samples*100:.1f}%)")
        print()

        if detailed:
            # Error breakdown
            breakdown = self.get_error_breakdown()
            print("ERROR BREAKDOWN")
            print("-" * 80)
            print(f"Perfect matches (0 error):   {breakdown['perfect_match']}")
            print(f"Off by 1 point:              {breakdown['off_by_1']}")
            print(f"Off by 2 points:             {breakdown['off_by_2']}")
            print(f"Off by 3 points:             {breakdown['off_by_3']}")
            print(f"Off by 4+ points:            {breakdown['off_by_4_plus']}")
            print()

            print("DIRECTIONAL ERRORS")
            print("-" * 80)
            print(f"Overestimate by 1:           {breakdown['overestimate_by_1']}")
            print(f"Overestimate by 2:           {breakdown['overestimate_by_2']}")
            print(f"Overestimate by 3+:          {breakdown['overestimate_by_3_plus']}")
            print(f"Underestimate by 1:          {breakdown['underestimate_by_1']}")
            print(f"Underestimate by 2:          {breakdown['underestimate_by_2']}")
            print(f"Underestimate by 3+:         {breakdown['underestimate_by_3_plus']}")
            print()

            # Worst predictions
            print("WORST PREDICTIONS (Top 5)")
            print("-" * 80)
            worst = self.get_worst_predictions(5)
            for i, pred in enumerate(worst, 1):
                print(f"{i}. Sample ID {pred['id']}")
                print(f"   Human: {pred['human_score']}, GPT-5: {pred['gpt5_score']}, Error: {pred['error']}")
                print(f"   Text preview: {pred['model_output'][:150]}...")
                print()

            # Confusion matrix
            print("CONFUSION MATRIX")
            print("-" * 80)
            self._print_confusion_matrix()
            print()

        print("="*80)
        print()

    def _print_confusion_matrix(self):
        """Print formatted confusion matrix."""
        confusion = self.get_confusion_matrix()

        # Determine score range from data
        min_score = int(min(min(self.human_scores), min(self.gpt5_scores)))
        max_score = int(max(max(self.human_scores), max(self.gpt5_scores)))

        print("        GPT-5 Predicted Score")
        print("       ", end="")
        for i in range(min_score, max_score + 1):
            print(f"{i:4d}", end="")
        print()
        print("Human  " + "-" * (4 * (max_score - min_score + 1)))

        for i, row in enumerate(confusion):
            score = i + min_score
            print(f"  {score:2d}  |", end="")
            for val in row:
                if val > 0:
                    print(f"{val:4d}", end="")
                else:
                    print("   .", end="")
            print()

    def save_report(self, output_path: str):
        """
        Save evaluation report to file.

        Args:
            output_path: Path to save the report
        """
        metrics = self.calculate_metrics()
        breakdown = self.get_error_breakdown()
        worst = self.get_worst_predictions(10)
        best = self.get_best_predictions(10)

        # Convert numpy types to native Python types
        breakdown_native = {k: int(v) for k, v in breakdown.items()}

        report = {
            'metrics': {
                'exact_match_accuracy': float(metrics.exact_match_accuracy),
                'within_1_accuracy': float(metrics.within_1_accuracy),
                'within_2_accuracy': float(metrics.within_2_accuracy),
                'mean_absolute_error': float(metrics.mean_absolute_error),
                'root_mean_square_error': float(metrics.root_mean_square_error),
                'mean_error': float(metrics.mean_error),
                'pearson_correlation': float(metrics.pearson_correlation),
                'pearson_p_value': float(metrics.pearson_p_value),
                'spearman_correlation': float(metrics.spearman_correlation),
                'spearman_p_value': float(metrics.spearman_p_value),
                'total_samples': int(metrics.total_samples)
            },
            'distributions': {
                'gpt5_mean': float(metrics.gpt5_mean),
                'gpt5_std': float(metrics.gpt5_std),
                'human_mean': float(metrics.human_mean),
                'human_std': float(metrics.human_std)
            },
            'error_breakdown': breakdown_native,
            'worst_predictions': worst,
            'best_predictions': best
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to {output_path}")


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_stats.py <data_file.json>")
        print("\nExample: python eval_stats.py evaluation_data.json")
        sys.exit(1)

    data_file = sys.argv[1]

    print(f"Loading data from {data_file}...")
    evaluator = LLMJudgeEvaluator(data_file)

    print(f"Loaded {len(evaluator.data)} samples")

    # Print detailed report
    evaluator.print_report(detailed=True)

    # Save to JSON
    output_file = data_file.replace('.json', '_report.json')
    evaluator.save_report(output_file)


if __name__ == "__main__":
    main()
