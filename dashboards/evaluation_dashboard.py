"""Streamlit dashboard for model evaluation and error analysis."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import torch
import numpy as np
import requests
import json
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    precision,
    recall,
    f1_score,
    exact_match,
    per_class_metrics,
    confusion_matrix_per_class,
    get_optimal_threshold
)
from evaluation.error_analysis import ErrorAnalyzer

# Page config
st.set_page_config(
    page_title="Model Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def _ensure_session_state() -> None:
    """
    Ensure required session state keys exist.

    Important: in Streamlit multipage apps, imported modules may not re-run on every script rerun.
    Initializing state only at import-time can lead to AttributeError on `st.session_state.<key>`.
    """
    if "use_api" not in st.session_state:
        st.session_state.use_api = False
    if "api_url" not in st.session_state:
        st.session_state.api_url = os.environ.get("API_URL", "http://localhost:8000")
    if "evaluation_data" not in st.session_state:
        st.session_state.evaluation_data = None


def load_predictions_from_file(file_path: str) -> Optional[Dict]:
    """Load predictions from file (placeholder for actual implementation)."""
    # This would load actual predictions from a saved file
    # For now, return None to indicate manual input needed
    return None


def main():
    """Main dashboard function."""
    _ensure_session_state()
    # Header
    st.markdown('<h1 class="main-header">📊 Model Evaluation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        use_api = st.checkbox(
            "Use API (instead of local processing)",
            value=st.session_state.use_api,
            help="Use FastAPI endpoint for evaluation"
        )
        st.session_state.use_api = use_api
        
        if use_api:
            api_url = st.text_input(
                "API URL",
                value=st.session_state.api_url,
                help="FastAPI endpoint URL"
            )
            st.session_state.api_url = api_url
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overall Metrics",
        "🏷️ Per-Class Metrics",
        "🔍 Error Analysis",
        "⚙️ Threshold Optimization"
    ])
    
    # Tab 1: Overall Metrics
    with tab1:
        st.header("Overall Classification Metrics")
        st.markdown("View overall precision, recall, F1 score, and exact match metrics.")
        
        # Input method
        input_method = st.radio(
            "Input method:",
            ["Upload predictions file", "Manual entry"],
            horizontal=True
        )
        
        target = None
        y_pred = None
        class_names = None
        
        if input_method == "Upload predictions file":
            uploaded_file = st.file_uploader(
                "Upload predictions file",
                type=["csv", "json", "pt"],
                help="File with predictions and targets"
            )
            if uploaded_file:
                # For demo, we'll use CSV format
                # Expected: columns: sample_id, class_0, class_1, ..., target_class_0, target_class_1, ...
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} samples")
                    st.dataframe(df.head())

                    # Infer prediction and target columns
                    pred_cols = [c for c in df.columns if c.startswith("class_")]
                    target_cols = [c for c in df.columns if c.startswith("target_class_")]

                    if not pred_cols or not target_cols:
                        st.error(
                            "Could not find prediction/target columns.\n\n"
                            "Expected columns like `class_0, class_1, ...` and "
                            "`target_class_0, target_class_1, ...`."
                        )
                    else:
                        # Ensure consistent ordering (numeric sort)
                        def get_class_num(col_name):
                            """Extract class number from column name."""
                            try:
                                if col_name.startswith('class_'):
                                    return int(col_name.split('_')[1])
                                elif col_name.startswith('target_class_'):
                                    return int(col_name.split('_')[2])
                            except (ValueError, IndexError):
                                return 0
                            return 0
                        
                        pred_cols = sorted(pred_cols, key=get_class_num)
                        target_cols = sorted(target_cols, key=get_class_num)
                        
                        # Verify column counts match
                        if len(pred_cols) != len(target_cols):
                            st.warning(
                                f"Warning: Number of prediction columns ({len(pred_cols)}) "
                                f"does not match target columns ({len(target_cols)})"
                            )

                        class_names = pred_cols

                        # Load predictions (may be probabilities or binary)
                        pred_probs = torch.tensor(df[pred_cols].values, dtype=torch.float32)
                        target = torch.tensor(df[target_cols].values, dtype=torch.float32)
                        
                        # Store probabilities for threshold optimization
                        st.session_state.pred_probs = pred_probs
                        st.session_state.target = target
                        st.session_state.class_names = class_names
                        
                        # Default: assume probabilities need thresholding
                        # User can adjust threshold in the UI
                        default_threshold = 0.5
                        y_pred = (pred_probs >= default_threshold).float()
                        st.session_state.y_pred = y_pred
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            st.info("""
            **Manual Entry Format:**
            - Enter predictions and targets as comma-separated values
            - Each row represents one sample
            - Format: predictions (0/1), targets (0/1)
            - Example: `1,0,1` for predictions and `1,0,0` for targets
            """)
            
            num_samples = st.number_input("Number of samples", min_value=1, max_value=100, value=5)
            num_classes = st.number_input("Number of classes", min_value=1, max_value=100, value=3)
            
            # Get class names
            class_names_input = st.text_input(
                "Class names (comma-separated)",
                value="class_0,class_1,class_2",
                help="Enter class names separated by commas"
            )
            class_names = [name.strip() for name in class_names_input.split(",")][:num_classes]
            
            # Manual entry for predictions and targets
            st.subheader("Enter Predictions and Targets")
            predictions_data = []
            targets_data = []
            
            for i in range(num_samples):
                with st.expander(f"Sample {i+1}"):
                    pred_input = st.text_input(
                        f"Predictions (0/1, comma-separated)",
                        value=",".join(["0"] * num_classes),
                        key=f"pred_{i}"
                    )
                    target_input = st.text_input(
                        f"Targets (0/1, comma-separated)",
                        value=",".join(["0"] * num_classes),
                        key=f"target_{i}"
                    )
                    
                    try:
                        preds = [int(x.strip()) for x in pred_input.split(",")][:num_classes]
                        targets = [int(x.strip()) for x in target_input.split(",")][:num_classes]
                        predictions_data.append(preds)
                        targets_data.append(targets)
                    except Exception as e:
                        st.error(f"Error parsing sample {i+1}: {e}")
            
            if predictions_data and targets_data:
                y_pred = torch.tensor(predictions_data, dtype=torch.float32)
                target = torch.tensor(targets_data, dtype=torch.float32)
        
        # Threshold input (if we have probabilities)
        threshold = 0.5
        if 'pred_probs' in st.session_state:
            threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get('threshold', 0.5),
                step=0.01,
                help="Threshold for converting probabilities to binary predictions"
            )
            # Convert probabilities to binary using threshold
            y_pred = (st.session_state.pred_probs >= threshold).float()
            target = st.session_state.target
            class_names = st.session_state.class_names
            # Store in session state for other tabs
            st.session_state.y_pred_binary = y_pred
            st.session_state.target_binary = target
            st.session_state.class_names_list = class_names
            st.session_state.threshold = threshold
        elif target is not None and y_pred is not None:
            # Already binary predictions (from manual entry)
            st.session_state.y_pred_binary = y_pred
            st.session_state.target_binary = target
            st.session_state.class_names_list = class_names if 'class_names' in locals() else None
            st.session_state.threshold = 0.5
        else:
            y_pred = None
            target = None
        
        # Use session state if available
        if 'y_pred_binary' in st.session_state:
            y_pred = st.session_state.y_pred_binary
            target = st.session_state.target_binary
        
        # Auto-calculate metrics when data is loaded (or on button click for manual entry)
        should_calculate = False
        if target is not None and y_pred is not None:
            if 'pred_probs' in st.session_state:
                # Auto-calculate for uploaded files
                should_calculate = True
            else:
                # Manual entry - require button click
                should_calculate = st.button("🔍 Calculate Metrics", type="primary")
        
        if should_calculate and target is not None and y_pred is not None:
            # Calculate overall metrics
            prec = precision(target, y_pred)
            rec = recall(target, y_pred)
            f1 = f1_score(target, y_pred)
            exact = exact_match(target, y_pred)
            
            # Display metrics
            st.subheader("Overall Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precision", f"{prec:.4f}")
            with col2:
                st.metric("Recall", f"{rec:.4f}")
            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
            with col4:
                st.metric("Exact Match", f"{exact:.4f}")
            
            # Metrics visualization
            metrics_df = pd.DataFrame({
                "Metric": ["Precision", "Recall", "F1 Score", "Exact Match"],
                "Score": [prec, rec, f1, exact]
            })
            
            fig_bar = px.bar(
                metrics_df,
                x="Metric",
                y="Score",
                title="Overall Classification Metrics",
                color="Score",
                color_continuous_scale="Blues",
                range_y=[0, 1]
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Tab 2: Per-Class Metrics
    with tab2:
        st.header("Per-Class Metrics")
        st.markdown("View precision, recall, and F1 score for each class.")
        
        # Get data from session state if available
        if 'y_pred_binary' in st.session_state:
            target = st.session_state.target_binary
            y_pred = st.session_state.y_pred_binary
            class_names = st.session_state.class_names_list
            threshold = st.session_state.get('threshold', 0.5)
            st.info(f"Using threshold: {threshold:.2f}")
        else:
            target = None
            y_pred = None
            class_names = None
        
        if target is not None and y_pred is not None and class_names is not None:
            if st.button("🔍 Calculate Per-Class Metrics", type="primary"):
                # Calculate per-class metrics
                per_class = per_class_metrics(target, y_pred, class_names)
                
                # Create DataFrame
                metrics_list = []
                for class_name, metrics in per_class.items():
                    metrics_list.append({
                        "Class": class_name,
                        "Precision": metrics["precision"],
                        "Recall": metrics["recall"],
                        "F1 Score": metrics["f1"],
                        "Support": metrics["support"],
                        "TP": metrics["tp"],
                        "FP": metrics["fp"],
                        "FN": metrics["fn"],
                        "TN": metrics["tn"]
                    })
                
                metrics_df = pd.DataFrame(metrics_list)
                
                st.subheader("Per-Class Metrics Table")
                st.dataframe(metrics_df, use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Precision by Class")
                    fig_prec = px.bar(
                        metrics_df,
                        x="Class",
                        y="Precision",
                        title="Precision per Class",
                        color="Precision",
                        color_continuous_scale="Greens",
                        range_y=[0, 1]
                    )
                    fig_prec.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_prec, use_container_width=True)
                
                with col2:
                    st.subheader("Recall by Class")
                    fig_rec = px.bar(
                        metrics_df,
                        x="Class",
                        y="Recall",
                        title="Recall per Class",
                        color="Recall",
                        color_continuous_scale="Blues",
                        range_y=[0, 1]
                    )
                    fig_rec.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_rec, use_container_width=True)
                
                # F1 Score comparison
                st.subheader("F1 Score by Class")
                fig_f1 = px.bar(
                    metrics_df,
                    x="Class",
                    y="F1 Score",
                    title="F1 Score per Class",
                    color="F1 Score",
                    color_continuous_scale="Purples",
                    range_y=[0, 1]
                )
                fig_f1.update_xaxes(tickangle=45)
                st.plotly_chart(fig_f1, use_container_width=True)
                
                # Confusion matrices
                st.subheader("Confusion Matrices")
                confusion_matrices = confusion_matrix_per_class(target, y_pred, class_names)
                
                # Display confusion matrices for top classes by support
                top_classes = metrics_df.nlargest(10, "Support")["Class"].tolist()
                
                for class_name in top_classes[:5]:  # Show top 5
                    with st.expander(f"Confusion Matrix: {class_name}"):
                        matrix = confusion_matrices[class_name].numpy()
                        
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=matrix,
                            x=["Predicted: 0", "Predicted: 1"],
                            y=["Actual: 0", "Actual: 1"],
                            colorscale="Blues",
                            text=matrix.astype(int),
                            texttemplate="%{text}",
                            textfont={"size": 20}
                        ))
                        fig_heatmap.update_layout(
                            title=f"Confusion Matrix: {class_name}",
                            xaxis_title="Predicted",
                            yaxis_title="Actual"
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.write(f"**True Negatives:** {matrix[0, 0]:.0f}")
                        st.write(f"**False Positives:** {matrix[0, 1]:.0f}")
                        st.write(f"**False Negatives:** {matrix[1, 0]:.0f}")
                        st.write(f"**True Positives:** {matrix[1, 1]:.0f}")
        else:
            st.info("Please enter predictions and targets in the 'Overall Metrics' tab first.")
    
    # Tab 3: Error Analysis
    with tab3:
        st.header("Error Analysis")
        st.markdown("Analyze false positives, false negatives, and common misclassification patterns.")
        
        # Get data from session state if available
        if 'y_pred_binary' in st.session_state:
            target = st.session_state.target_binary
            y_pred = st.session_state.y_pred_binary
            class_names = st.session_state.class_names_list
        else:
            target = None
            y_pred = None
            class_names = None
        
        if target is not None and y_pred is not None and class_names is not None:
            if st.button("🔍 Analyze Errors", type="primary"):
                analyzer = ErrorAnalyzer()
                
                # Get error summary
                error_summary = analyzer.get_error_summary(target, y_pred, class_names)
                
                # Display summary
                st.subheader("Error Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total FP", int(error_summary["total_false_positives"]))
                with col2:
                    st.metric("Total FN", int(error_summary["total_false_negatives"]))
                with col3:
                    st.metric("FP Rate", f"{error_summary['fp_rate']:.4f}")
                with col4:
                    st.metric("FN Rate", f"{error_summary['fn_rate']:.4f}")
                
                # Top error classes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top 10 Classes by False Positives")
                    fp_df = pd.DataFrame(error_summary["top_fp_classes"])
                    if not fp_df.empty:
                        fig_fp = px.bar(
                            fp_df,
                            x="class",
                            y="count",
                            title="False Positives by Class",
                            color="count",
                            color_continuous_scale="Reds"
                        )
                        fig_fp.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_fp, use_container_width=True)
                
                with col2:
                    st.subheader("Top 10 Classes by False Negatives")
                    fn_df = pd.DataFrame(error_summary["top_fn_classes"])
                    if not fn_df.empty:
                        fig_fn = px.bar(
                            fn_df,
                            x="class",
                            y="count",
                            title="False Negatives by Class",
                            color="count",
                            color_continuous_scale="Oranges"
                        )
                        fig_fn.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_fn, use_container_width=True)
                
                # Common misclassification patterns
                st.subheader("Common Misclassification Patterns")
                patterns = analyzer.find_common_misclassification_patterns(
                    target, y_pred, class_names, top_k=10
                )
                
                if patterns:
                    patterns_data = []
                    for (pred_classes, actual_classes), count in patterns:
                        patterns_data.append({
                            "Predicted": ", ".join(pred_classes) if pred_classes else "None",
                            "Actual": ", ".join(actual_classes) if actual_classes else "None",
                            "Count": count
                        })
                    
                    patterns_df = pd.DataFrame(patterns_data)
                    st.dataframe(patterns_df, use_container_width=True)
                else:
                    st.info("No misclassification patterns found.")
                
                # Confusion analysis
                st.subheader("Class Confusion Analysis")
                confusion_df = analyzer.analyze_class_confusion(target, y_pred, class_names)
                
                if not confusion_df.empty:
                    st.dataframe(confusion_df, use_container_width=True)
                    
                    # Confusion heatmap
                    if len(confusion_df) > 0:
                        fig_confusion = px.scatter(
                            confusion_df,
                            x="predicted",
                            y="actual",
                            size="count",
                            color="count",
                            title="Class Confusion Patterns",
                            labels={"count": "Confusion Count"},
                            color_continuous_scale="Reds"
                        )
                        fig_confusion.update_xaxes(tickangle=45)
                        fig_confusion.update_yaxes(tickangle=0)
                        st.plotly_chart(fig_confusion, use_container_width=True)
                else:
                    st.info("No class confusions found.")
        else:
            st.info("Please enter predictions and targets in the 'Overall Metrics' tab first.")
    
    # Tab 4: Threshold Optimization
    with tab4:
        st.header("Threshold Optimization")
        st.markdown("Find optimal threshold for classification based on different metrics.")
        
        # Check if we have predictions and targets from uploaded file
        if 'pred_probs' in st.session_state and 'target' in st.session_state:
            pred_probs = st.session_state.pred_probs
            target = st.session_state.target
            
            st.success(f"✅ Using predictions from uploaded file ({len(pred_probs)} samples)")
            
            # Threshold range
            col1, col2, col3 = st.columns(3)
            with col1:
                min_threshold = st.slider("Min Threshold", 0.0, 1.0, 0.0, 0.01)
            with col2:
                max_threshold = st.slider("Max Threshold", 0.0, 1.0, 1.0, 0.01)
            with col3:
                step = st.slider("Step", 0.01, 0.1, 0.05, 0.01)
            
            threshold_list = [round(x, 3) for x in np.arange(min_threshold, max_threshold + step, step)]
            
            # Metric selection
            metric = st.selectbox(
                "Optimize for metric:",
                ["precision", "recall", "f1"],
                help="Select metric to optimize"
            )
            
            st.info(f"Will test {len(threshold_list)} thresholds from {min_threshold} to {max_threshold}")
            
            if st.button("🔍 Optimize Threshold", type="primary"):
                with st.spinner("Testing thresholds..."):
                    # Convert to tensors if needed
                    if isinstance(pred_probs, np.ndarray):
                        pred_probs_tensor = torch.from_numpy(pred_probs).float()
                    else:
                        pred_probs_tensor = pred_probs.float()
                    
                    if isinstance(target, np.ndarray):
                        target_tensor = torch.from_numpy(target).float()
                    else:
                        target_tensor = target.float()
                    
                    # Test each threshold
                    results = []
                    metric_funcs = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1_score,
                    }
                    metric_func = metric_funcs[metric]
                    
                    progress_bar = st.progress(0)
                    for i, thresh in enumerate(threshold_list):
                        # Convert probabilities to binary using threshold
                        y_pred_binary = (pred_probs_tensor >= thresh).float()
                        
                        # Calculate metrics
                        prec = precision(target_tensor, y_pred_binary)
                        rec = recall(target_tensor, y_pred_binary)
                        f1 = f1_score(target_tensor, y_pred_binary)
                        exact = exact_match(target_tensor, y_pred_binary)
                        
                        results.append({
                            'threshold': thresh,
                            'precision': prec,
                            'recall': rec,
                            'f1': f1,
                            'exact_match': exact,
                        })
                        
                        progress_bar.progress((i + 1) / len(threshold_list))
                    
                    progress_bar.empty()
                    
                    # Find optimal threshold
                    results_df = pd.DataFrame(results)
                    optimal_idx = results_df[metric].idxmax()
                    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
                    optimal_score = results_df.loc[optimal_idx, metric]
                    
                    # Display results
                    st.subheader("Optimal Threshold")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"Optimal Threshold ({metric})", f"{optimal_threshold:.3f}")
                    with col2:
                        st.metric(f"Best {metric.capitalize()}", f"{optimal_score:.4f}")
                    
                    # Show metrics at optimal threshold
                    st.subheader(f"Metrics at Threshold {optimal_threshold:.3f}")
                    opt_row = results_df.loc[optimal_idx]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Precision", f"{opt_row['precision']:.4f}")
                    with col2:
                        st.metric("Recall", f"{opt_row['recall']:.4f}")
                    with col3:
                        st.metric("F1 Score", f"{opt_row['f1']:.4f}")
                    with col4:
                        st.metric("Exact Match", f"{opt_row['exact_match']:.4f}")
                    
                    # Visualization
                    st.subheader("Threshold vs Metrics")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results_df['threshold'],
                        y=results_df['precision'],
                        mode='lines',
                        name='Precision',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df['threshold'],
                        y=results_df['recall'],
                        mode='lines',
                        name='Recall',
                        line=dict(color='green')
                    ))
                    fig.add_trace(go.Scatter(
                        x=results_df['threshold'],
                        y=results_df['f1'],
                        mode='lines',
                        name='F1 Score',
                        line=dict(color='red')
                    ))
                    # Mark optimal threshold
                    fig.add_vline(
                        x=optimal_threshold,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text=f"Optimal ({optimal_threshold:.3f})"
                    )
                    fig.update_layout(
                        title="Metrics vs Threshold",
                        xaxis_title="Threshold",
                        yaxis_title="Score",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Export optimal threshold
                    st.subheader("Export Threshold")
                    if st.button("💾 Save to config/thresholds.json"):
                        import json
                        from pathlib import Path
                        
                        config_path = Path("config/thresholds.json")
                        config_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Load existing config or create new
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                        else:
                            config = {
                                "global_threshold": 0.5,
                                "per_class_thresholds": {},
                                "model_version": "best_model_v2.pt"
                            }
                        
                        config['global_threshold'] = float(optimal_threshold)
                        config['model_version'] = "best_model_v2.pt"
                        
                        with open(config_path, 'w') as f:
                            json.dump(config, f, indent=2)
                        
                        st.success(f"✅ Saved optimal threshold ({optimal_threshold:.3f}) to {config_path}")
        else:
            st.warning("⚠️ Please upload a predictions file in the 'Overall Metrics' tab first.")
            st.info("""
            **Threshold optimization requires:**
            - Model predictions (probabilities, not binary) - upload CSV file
            - Ground truth targets - included in uploaded CSV
            
            Upload your predictions file in the 'Overall Metrics' tab to enable this feature.
            """)


if __name__ == "__main__":
    main()



