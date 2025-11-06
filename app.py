"""
Streamlit web app for GPT-5 Judge Evaluation
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from eval_stats import LLMJudgeEvaluator
import tempfile
import os


st.set_page_config(
    page_title="GPT-5 Judge Evaluator",
    page_icon="⚖️",
    layout="wide"
)


def main():
    st.title("⚖️ LLM Judge Evaluator")
    st.markdown("""
    Evaluate GPT-5's judging accuracy against human ground truth labels.
    Upload your evaluation data to get comprehensive statistical analysis.
    """)

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This tool compares GPT-5 judgments against human ratings to assess:
    - Accuracy and correlation
    - Error patterns
    - Bias analysis
    - Worst/best predictions
    """)

    st.sidebar.markdown("---")
    st.sidebar.header("Data Format")
    st.sidebar.markdown("""
    Upload a **JSON** file with this structure:
    ```json
    [
      {
        "id": 1,
        "model_output": "text...",
        "human_score": 8,
        "gpt5_score": 7
      }
    ]
    ```
    """)

    # File upload
    st.header("📁 Upload Evaluation Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            help="Upload your evaluation data in JSON format"
        )

    with col2:
        st.download_button(
            label="📥 Download Template",
            data=json.dumps([
                {
                    "id": 1,
                    "model_output": "Example text from your model",
                    "human_score": 8,
                    "gpt5_score": 7,
                    "notes": "Optional notes"
                }
            ], indent=2),
            file_name="evaluation_template.json",
            mime="application/json"
        )

    if uploaded_file is not None:
        try:
            # Load data
            data = json.load(uploaded_file)

            # Save to temporary file for evaluator
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                json.dump(data, tmp_file)
                tmp_path = tmp_file.name

            # Initialize evaluator
            evaluator = LLMJudgeEvaluator(tmp_path)

            st.success(f"✅ Loaded {len(evaluator.data)} samples successfully!")

            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                df = pd.DataFrame(data)
                st.dataframe(df.head(10), use_container_width=True)

            st.markdown("---")

            # Calculate metrics
            metrics = evaluator.calculate_metrics()

            # Display metrics in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Overview",
                "🎯 Accuracy",
                "📊 Error Analysis",
                "🔍 Predictions",
                "📋 Full Report"
            ])

            with tab1:
                display_overview(metrics, evaluator)

            with tab2:
                display_accuracy_metrics(metrics, evaluator)

            with tab3:
                display_error_analysis(metrics, evaluator)

            with tab4:
                display_predictions(evaluator)

            with tab5:
                display_full_report(evaluator)

            # Clean up temp file
            os.unlink(tmp_path)

        except json.JSONDecodeError:
            st.error("❌ Invalid JSON file. Please check your file format.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        # Show example/demo
        st.info("👆 Upload your evaluation data to get started, or try the sample data below.")

        if st.button("🎮 Try Sample Data"):
            try:
                evaluator = LLMJudgeEvaluator("sample_evaluation_data.json")
                st.success(f"✅ Loaded {len(evaluator.data)} sample records!")

                metrics = evaluator.calculate_metrics()

                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Overview",
                    "🎯 Accuracy",
                    "📊 Error Analysis",
                    "🔍 Predictions",
                    "📋 Full Report"
                ])

                with tab1:
                    display_overview(metrics, evaluator)

                with tab2:
                    display_accuracy_metrics(metrics, evaluator)

                with tab3:
                    display_error_analysis(metrics, evaluator)

                with tab4:
                    display_predictions(evaluator)

                with tab5:
                    display_full_report(evaluator)

            except FileNotFoundError:
                st.warning("Sample data file not found. Please upload your own data.")


def display_overview(metrics, evaluator):
    """Display overview metrics."""
    st.header("📈 Evaluation Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Samples",
            value=metrics.total_samples
        )

    with col2:
        st.metric(
            label="Pearson Correlation",
            value=f"{metrics.pearson_correlation:.3f}",
            delta="Strong" if metrics.pearson_correlation >= 0.8 else "Moderate" if metrics.pearson_correlation >= 0.6 else "Weak"
        )

    with col3:
        st.metric(
            label="Mean Absolute Error",
            value=f"{metrics.mean_absolute_error:.2f}",
            delta="Good" if metrics.mean_absolute_error < 1.5 else "Fair" if metrics.mean_absolute_error < 2.0 else "Poor",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Within ±1 Accuracy",
            value=f"{metrics.within_1_accuracy*100:.1f}%"
        )

    st.markdown("---")

    # Score distribution comparison
    col1, col2 = st.columns(2)

    with col1:
        # Scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=evaluator.human_scores,
            y=evaluator.gpt5_scores,
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.6),
            name='Predictions'
        ))

        # Add diagonal line (perfect agreement)
        min_val = min(min(evaluator.human_scores), min(evaluator.gpt5_scores))
        max_val = max(max(evaluator.human_scores), max(evaluator.gpt5_scores))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Agreement'
        ))

        fig.update_layout(
            title="Human vs GPT-5 Scores",
            xaxis_title="Human Score",
            yaxis_title="GPT-5 Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution comparison
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=evaluator.human_scores,
            name='Human Scores',
            opacity=0.6,
            marker_color='blue'
        ))
        fig.add_trace(go.Histogram(
            x=evaluator.gpt5_scores,
            name='GPT-5 Scores',
            opacity=0.6,
            marker_color='orange'
        ))

        fig.update_layout(
            title="Score Distributions",
            xaxis_title="Score",
            yaxis_title="Count",
            barmode='overlay',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def display_accuracy_metrics(metrics, evaluator):
    """Display accuracy metrics."""
    st.header("🎯 Accuracy Metrics")

    # Accuracy breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Match Accuracy")

        accuracy_data = {
            'Metric': ['Exact Match', 'Within ±1', 'Within ±2'],
            'Percentage': [
                metrics.exact_match_accuracy * 100,
                metrics.within_1_accuracy * 100,
                metrics.within_2_accuracy * 100
            ]
        }

        fig = go.Figure(data=[
            go.Bar(
                x=accuracy_data['Metric'],
                y=accuracy_data['Percentage'],
                text=[f"{v:.1f}%" for v in accuracy_data['Percentage']],
                textposition='auto',
                marker_color=['#2ecc71', '#3498db', '#9b59b6']
            )
        ])

        fig.update_layout(
            yaxis_title="Percentage",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        if metrics.exact_match_accuracy >= 0.6:
            st.success("✅ Excellent exact match rate!")
        elif metrics.exact_match_accuracy >= 0.4:
            st.info("ℹ️ Good exact match rate")
        else:
            st.warning("⚠️ Low exact match rate - check for systematic bias")

    with col2:
        st.subheader("Correlation Metrics")

        st.metric("Pearson Correlation", f"{metrics.pearson_correlation:.3f}")
        st.caption(f"p-value: {metrics.pearson_p_value:.6f}")

        st.metric("Spearman Correlation", f"{metrics.spearman_correlation:.3f}")
        st.caption(f"p-value: {metrics.spearman_p_value:.6f}")

        # Interpretation
        if metrics.pearson_correlation >= 0.9:
            st.success("✅ Very strong correlation - GPT-5 aligns well with humans!")
        elif metrics.pearson_correlation >= 0.7:
            st.info("ℹ️ Strong correlation - good alignment")
        elif metrics.pearson_correlation >= 0.5:
            st.warning("⚠️ Moderate correlation - some alignment issues")
        else:
            st.error("❌ Weak correlation - GPT-5 judgments may not be reliable")

    # Bias analysis
    st.markdown("---")
    st.subheader("Bias Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Exact Matches", metrics.exact_match_count)

    with col2:
        st.metric(
            "Overestimates",
            metrics.overestimate_count,
            delta=f"{metrics.overestimate_count/metrics.total_samples*100:.1f}%"
        )

    with col3:
        st.metric(
            "Underestimates",
            metrics.underestimate_count,
            delta=f"{metrics.underestimate_count/metrics.total_samples*100:.1f}%"
        )

    # Bias indicator
    if abs(metrics.mean_error) < 0.2:
        st.success("✅ No significant bias detected")
    elif metrics.mean_error > 0:
        st.warning(f"⚠️ GPT-5 tends to OVERESTIMATE by {metrics.mean_error:.2f} points on average")
    else:
        st.warning(f"⚠️ GPT-5 tends to UNDERESTIMATE by {abs(metrics.mean_error):.2f} points on average")


def display_error_analysis(metrics, evaluator):
    """Display error analysis."""
    st.header("📊 Error Analysis")

    breakdown = evaluator.get_error_breakdown()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Error Magnitude Distribution")

        error_data = {
            'Error': ['Perfect', 'Off by 1', 'Off by 2', 'Off by 3', 'Off by 4+'],
            'Count': [
                breakdown['perfect_match'],
                breakdown['off_by_1'],
                breakdown['off_by_2'],
                breakdown['off_by_3'],
                breakdown['off_by_4_plus']
            ]
        }

        fig = px.pie(
            error_data,
            values='Count',
            names='Error',
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Directional Errors")

        directional_data = {
            'Type': [
                'Over by 1', 'Over by 2', 'Over by 3+',
                'Under by 1', 'Under by 2', 'Under by 3+'
            ],
            'Count': [
                breakdown['overestimate_by_1'],
                breakdown['overestimate_by_2'],
                breakdown['overestimate_by_3_plus'],
                breakdown['underestimate_by_1'],
                breakdown['underestimate_by_2'],
                breakdown['underestimate_by_3_plus']
            ],
            'Direction': ['Over', 'Over', 'Over', 'Under', 'Under', 'Under']
        }

        fig = go.Figure(data=[
            go.Bar(
                x=directional_data['Type'],
                y=directional_data['Count'],
                marker_color=['#e74c3c' if d == 'Over' else '#3498db' for d in directional_data['Direction']]
            )
        ])
        fig.update_layout(
            yaxis_title="Count",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Error metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Absolute Error", f"{metrics.mean_absolute_error:.3f}")

    with col2:
        st.metric("Root Mean Square Error", f"{metrics.root_mean_square_error:.3f}")

    with col3:
        st.metric("Mean Error (Bias)", f"{metrics.mean_error:.3f}")


def display_predictions(evaluator):
    """Display worst and best predictions."""
    st.header("🔍 Prediction Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("❌ Worst Predictions")
        worst = evaluator.get_worst_predictions(10)

        for i, pred in enumerate(worst, 1):
            with st.expander(f"#{i} - Sample {pred['id']} (Error: {pred['error']:+d})"):
                st.write(f"**Human Score:** {pred['human_score']}")
                st.write(f"**GPT-5 Score:** {pred['gpt5_score']}")
                st.write(f"**Error:** {pred['error']:+d}")
                st.write(f"**Text:** {pred['model_output'][:300]}...")

    with col2:
        st.subheader("✅ Best Predictions")
        best = evaluator.get_best_predictions(10)

        for i, pred in enumerate(best, 1):
            with st.expander(f"#{i} - Sample {pred['id']} (Error: {pred['error']:+d})"):
                st.write(f"**Human Score:** {pred['human_score']}")
                st.write(f"**GPT-5 Score:** {pred['gpt5_score']}")
                st.write(f"**Error:** {pred['error']:+d}")
                st.write(f"**Text:** {pred['model_output'][:300]}...")


def display_full_report(evaluator):
    """Display full text report."""
    st.header("📋 Full Statistical Report")

    # Generate report in text format
    import io
    from contextlib import redirect_stdout

    # Capture the printed report
    f = io.StringIO()
    with redirect_stdout(f):
        evaluator.print_report(detailed=True)
    report_text = f.getvalue()

    st.code(report_text, language=None)

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📥 Download Text Report",
            data=report_text,
            file_name="evaluation_report.txt",
            mime="text/plain"
        )

    with col2:
        # Generate JSON report
        metrics = evaluator.calculate_metrics()
        breakdown = evaluator.get_error_breakdown()

        json_report = {
            'metrics': {
                'exact_match_accuracy': float(metrics.exact_match_accuracy),
                'within_1_accuracy': float(metrics.within_1_accuracy),
                'mean_absolute_error': float(metrics.mean_absolute_error),
                'pearson_correlation': float(metrics.pearson_correlation),
            },
            'error_breakdown': {k: int(v) for k, v in breakdown.items()}
        }

        st.download_button(
            label="📥 Download JSON Report",
            data=json.dumps(json_report, indent=2),
            file_name="evaluation_report.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
