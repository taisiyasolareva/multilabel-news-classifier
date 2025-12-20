"""Streamlit dashboard for model comparison and experiment tracking."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
from pathlib import Path
import sys
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.experiment_tracker import ExperimentTracker
from experiments.model_comparison import ModelComparison

# Page config
st.set_page_config(
    page_title="Model Comparison Dashboard",
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

# Initialize session state
if 'tracker' not in st.session_state:
    st.session_state.tracker = ExperimentTracker()
if 'comparison' not in st.session_state:
    st.session_state.comparison = ModelComparison(tracker=st.session_state.tracker)


def load_experiments() -> pd.DataFrame:
    """Load all experiments into a DataFrame."""
    experiments = st.session_state.tracker.list_experiments()
    if not experiments:
        return pd.DataFrame()
    
    # Flatten experiment data
    data = []
    for exp in experiments:
        row = {
            "experiment_id": exp.get("experiment_id"),
            "experiment_name": exp.get("experiment_name"),
            "model_name": exp.get("model_name"),
            "status": exp.get("status"),
            "start_time": exp.get("start_time"),
            "end_time": exp.get("end_time", ""),
        }
        
        # Add metrics
        metrics = exp.get("metrics", {})
        for key, value in metrics.items():
            row[key] = value
        
        data.append(row)
    
    return pd.DataFrame(data)


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<h1 class="main-header">📊 Model Comparison Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        results_dir = st.text_input(
            "Results Directory",
            value="experiments/results",
            help="Directory containing experiment results"
        )
        
        if st.button("🔄 Refresh Data"):
            st.session_state.tracker = ExperimentTracker(results_dir=results_dir)
            st.session_state.comparison = ModelComparison(tracker=st.session_state.tracker)
            st.rerun()
        
        st.markdown("---")
        st.header("📁 Data Source")
        
        data_source = st.radio(
            "Load data from:",
            ["Experiment Tracker", "Upload CSV"],
            help="Choose data source"
        )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🔍 Model Comparison",
        "📊 Metrics Analysis",
        "🏆 Best Model Selection"
    ])
    
    # Load data
    if data_source == "Experiment Tracker":
        df = load_experiments()
    else:
        uploaded_file = st.file_uploader("Upload comparison CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.DataFrame()
    
    # Tab 1: Overview
    with tab1:
        st.header("Experiments Overview")
        
        if df.empty:
            st.info("No experiments found. Start some experiments to see results here.")
        else:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Experiments", len(df))
            with col2:
                completed = len(df[df["status"] == "completed"])
                st.metric("Completed", completed)
            with col3:
                running = len(df[df["status"] == "running"])
                st.metric("Running", running)
            with col4:
                unique_models = df["model_name"].nunique() if "model_name" in df.columns else 0
                st.metric("Unique Models", unique_models)
            
            # Experiments table
            st.subheader("All Experiments")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=df["status"].unique() if "status" in df.columns else [],
                    default=df["status"].unique() if "status" in df.columns else []
                )
            with col2:
                model_filter = st.multiselect(
                    "Filter by Model",
                    options=df["model_name"].unique() if "model_name" in df.columns else [],
                    default=df["model_name"].unique() if "model_name" in df.columns else []
                )
            
            # Apply filters
            filtered_df = df.copy()
            if "status" in filtered_df.columns and status_filter:
                filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
            if "model_name" in filtered_df.columns and model_filter:
                filtered_df = filtered_df[filtered_df["model_name"].isin(model_filter)]
            
            st.dataframe(filtered_df, use_container_width=True)
            
            # Model distribution
            if "model_name" in df.columns:
                st.subheader("Model Distribution")
                model_counts = df["model_name"].value_counts()
                
                fig_pie = px.pie(
                    values=model_counts.values,
                    names=model_counts.index,
                    title="Experiments by Model"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # Tab 2: Model Comparison
    with tab2:
        st.header("Model Comparison")
        
        if df.empty:
            st.info("No experiments found. Start some experiments to see comparisons.")
        else:
            # Get metric columns
            metric_cols = [col for col in df.columns if any(
                m in col.lower() for m in ["precision", "recall", "f1", "exact_match", "accuracy"]
            )]
            
            if not metric_cols:
                st.warning("No metrics found in the data. Ensure experiments have logged metrics.")
            else:
                # Select metrics to compare
                selected_metrics = st.multiselect(
                    "Select Metrics to Compare",
                    options=metric_cols,
                    default=metric_cols[:4] if len(metric_cols) >= 4 else metric_cols
                )
                
                if selected_metrics:
                    # Filter completed experiments
                    completed_df = df[df["status"] == "completed"].copy()
                    
                    if completed_df.empty:
                        st.warning("No completed experiments found.")
                    else:
                        # Group by model and calculate averages
                        if "model_name" in completed_df.columns:
                            comparison_df = completed_df.groupby("model_name")[selected_metrics].mean().reset_index()
                            
                            st.subheader("Average Metrics by Model")
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Visualization
                            st.subheader("Metrics Comparison")
                            
                            # Bar chart
                            fig_bar = go.Figure()
                            
                            for metric in selected_metrics:
                                fig_bar.add_trace(go.Bar(
                                    name=metric,
                                    x=comparison_df["model_name"],
                                    y=comparison_df[metric]
                                ))
                            
                            fig_bar.update_layout(
                                barmode='group',
                                title="Metrics Comparison by Model",
                                xaxis_title="Model",
                                yaxis_title="Score",
                                height=500
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)
                            
                            # Heatmap
                            st.subheader("Metrics Heatmap")
                            heatmap_data = comparison_df.set_index("model_name")[selected_metrics]
                            
                            fig_heatmap = px.imshow(
                                heatmap_data.T,
                                labels=dict(x="Model", y="Metric", color="Score"),
                                title="Metrics Heatmap",
                                color_continuous_scale="RdYlGn",
                                aspect="auto"
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Tab 3: Metrics Analysis
    with tab3:
        st.header("Metrics Analysis")
        
        if df.empty:
            st.info("No experiments found.")
        else:
            # Get metric columns
            metric_cols = [col for col in df.columns if any(
                m in col.lower() for m in ["precision", "recall", "f1", "exact_match", "accuracy"]
            )]
            
            if not metric_cols:
                st.warning("No metrics found.")
            else:
                # Select metric to analyze
                selected_metric = st.selectbox(
                    "Select Metric",
                    options=metric_cols
                )
                
                if selected_metric:
                    # Filter completed experiments
                    completed_df = df[df["status"] == "completed"].copy()
                    
                    if completed_df.empty:
                        st.warning("No completed experiments found.")
                    else:
                        # Distribution
                        st.subheader(f"{selected_metric} Distribution")
                        
                        fig_dist = px.histogram(
                            completed_df,
                            x=selected_metric,
                            nbins=20,
                            title=f"Distribution of {selected_metric}",
                            labels={selected_metric: selected_metric}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        
                        # Box plot by model
                        if "model_name" in completed_df.columns:
                            st.subheader(f"{selected_metric} by Model")
                            
                            fig_box = px.box(
                                completed_df,
                                x="model_name",
                                y=selected_metric,
                                title=f"{selected_metric} Distribution by Model",
                                labels={"model_name": "Model"}
                            )
                            fig_box.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_box, use_container_width=True)
                            
                            # Scatter plot (if multiple metrics)
                            if len(metric_cols) >= 2:
                                st.subheader("Metrics Correlation")
                                
                                x_metric = st.selectbox(
                                    "X-axis Metric",
                                    options=metric_cols,
                                    index=0,
                                    key="x_metric"
                                )
                                y_metric = st.selectbox(
                                    "Y-axis Metric",
                                    options=metric_cols,
                                    index=1 if len(metric_cols) > 1 else 0,
                                    key="y_metric"
                                )
                                
                                if x_metric != y_metric:
                                    fig_scatter = px.scatter(
                                        completed_df,
                                        x=x_metric,
                                        y=y_metric,
                                        color="model_name" if "model_name" in completed_df.columns else None,
                                        size=selected_metric if selected_metric not in [x_metric, y_metric] else None,
                                        title=f"{x_metric} vs {y_metric}",
                                        labels={"model_name": "Model"}
                                    )
                                    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tab 4: Best Model Selection
    with tab4:
        st.header("Best Model Selection")
        
        if df.empty:
            st.info("No experiments found.")
        else:
            # Get metric columns
            metric_cols = [col for col in df.columns if any(
                m in col.lower() for m in ["precision", "recall", "f1", "exact_match", "accuracy"]
            )]
            
            if not metric_cols:
                st.warning("No metrics found.")
            else:
                # Select metric for best model selection
                selection_metric = st.selectbox(
                    "Select Metric for Best Model",
                    options=metric_cols,
                    help="Model with highest value in this metric will be selected"
                )
                
                # Filter completed experiments
                completed_df = df[df["status"] == "completed"].copy()
                
                if completed_df.empty:
                    st.warning("No completed experiments found.")
                else:
                    # Get best model
                    if selection_metric in completed_df.columns:
                        best_idx = completed_df[selection_metric].idxmax()
                        best_experiment = completed_df.loc[best_idx]
                        
                        st.subheader("🏆 Best Model")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Model Name", best_experiment.get("model_name", "Unknown"))
                            st.metric("Experiment ID", best_experiment.get("experiment_id", "Unknown"))
                            st.metric(f"Best {selection_metric}", f"{best_experiment[selection_metric]:.4f}")
                        
                        with col2:
                            # Show all metrics for best model
                            st.subheader("All Metrics")
                            best_metrics = {k: v for k, v in best_experiment.items() if k in metric_cols}
                            for metric, value in best_metrics.items():
                                if pd.notna(value):
                                    st.metric(metric, f"{value:.4f}")
                        
                        # Comparison with other models
                        st.subheader("Comparison with Other Models")
                        
                        if "model_name" in completed_df.columns:
                            comparison_data = completed_df.groupby("model_name")[selection_metric].mean().sort_values(ascending=False)
                            
                            fig_comparison = px.bar(
                                x=comparison_data.index,
                                y=comparison_data.values,
                                title=f"Average {selection_metric} by Model",
                                labels={"x": "Model", "y": selection_metric},
                                color=comparison_data.values,
                                color_continuous_scale="RdYlGn"
                            )
                            fig_comparison.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_comparison, use_container_width=True)
                            
                            # Highlight best model
                            st.success(f"✅ Best model: **{best_experiment.get('model_name', 'Unknown')}** with {selection_metric}={best_experiment[selection_metric]:.4f}")
                    else:
                        st.error(f"Metric {selection_metric} not found in data.")


if __name__ == "__main__":
    main()



