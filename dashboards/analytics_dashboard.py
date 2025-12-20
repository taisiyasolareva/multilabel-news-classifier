"""Streamlit dashboard for advanced analytics."""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional
import requests
import json

# Ensure project root is on sys.path so that `analysis` and other local packages can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.predictive_intervals import (
    calculate_predictive_interval,
    rank_by_predictive_interval,
    get_top_positive_by_interval,
    get_top_negative_by_interval
)
from analysis.category_analytics import CategoryAnalytics
from analysis.thread_analysis import ThreadAnalyzer

# Page config
st.set_page_config(
    page_title="Advanced Analytics Dashboard",
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
    if "analytics_data" not in st.session_state:
        st.session_state.analytics_data = None


def main():
    """Main dashboard function."""
    _ensure_session_state()
    # Header
    st.markdown('<h1 class="main-header">📊 Advanced Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        use_api = st.checkbox(
            "Use API (instead of local processing)",
            value=st.session_state.use_api,
            help="Use FastAPI endpoint for analytics"
        )
        st.session_state.use_api = use_api
        
        if use_api:
            api_url = st.text_input(
                "API URL",
                value=st.session_state.api_url,
                help="FastAPI endpoint URL"
            )
            st.session_state.api_url = api_url
            
            # Test API connection
            if st.button("Test API Connection"):
                try:
                    response = requests.get(
                        f"{api_url}/health",
                        timeout=5
                    )
                    if response.status_code == 200:
                        st.success("✅ API is healthy!")
                    else:
                        st.warning(f"API returned status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ API connection failed: {e}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Predictive Intervals",
        "🏷️ Category Analytics",
        "🧵 Thread Analysis"
    ])
    
    # Tab 1: Predictive Intervals
    with tab1:
        st.header("Predictive Intervals for Ranking")
        st.markdown("""
        Use predictive intervals to rank news articles or categories by positive sentiment,
        accounting for uncertainty when sample sizes are small.
        """)
        
        # Input method
        input_method = st.radio(
            "Input method:",
            ["Manual entry", "Upload CSV file"],
            horizontal=True
        )
        
        data = []
        
        if input_method == "Manual entry":
            st.subheader("Enter Sentiment Counts")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                num_items = st.number_input("Number of items", min_value=1, max_value=50, value=3)
            
            items = []
            for i in range(num_items):
                with st.expander(f"Item {i+1}"):
                    item_id = st.text_input(f"ID/Name {i+1}", value=f"item_{i+1}", key=f"id_{i}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        positive = st.number_input("Positive", min_value=0, value=10, key=f"pos_{i}")
                    with col2:
                        negative = st.number_input("Negative", min_value=0, value=5, key=f"neg_{i}")
                    with col3:
                        neutral = st.number_input("Neutral", min_value=0, value=0, key=f"neu_{i}")
                    
                    items.append({
                        "id": item_id,
                        "positive_count": positive,
                        "negative_count": negative,
                        "neutral_count": neutral
                    })
            
            data = items
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="CSV file with columns: id, positive_count, negative_count, neutral_count"
            )
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                required_cols = ["positive_count", "negative_count"]
                if all(col in df.columns for col in required_cols):
                    if "neutral_count" not in df.columns:
                        df["neutral_count"] = 0
                    if "id" not in df.columns:
                        df["id"] = range(len(df))
                    
                    data = df.to_dict('records')
                    st.success(f"Loaded {len(data)} items from CSV")
                else:
                    st.error(f"CSV must have columns: {required_cols}")
        
        if data:
            st.info(f"📝 {len(data)} items ready for analysis")
            
            # Confidence level
            confidence = st.slider(
                "Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Confidence level for predictive interval"
            )
            
            if st.button("🔍 Calculate Predictive Intervals", type="primary"):
                if st.session_state.use_api:
                    # Use API
                    try:
                        response = requests.post(
                            f"{st.session_state.api_url}/analytics/predictive-intervals",
                            json={
                                "data": data,
                                "confidence_level": confidence
                            },
                            timeout=30
                        )
                        response.raise_for_status()
                        results = response.json()
                        ranked_data = results.get("ranked_data", [])
                    except Exception as e:
                        st.error(f"API request failed: {e}")
                        ranked_data = []
                else:
                    # Local processing
                    ranked_data = rank_by_predictive_interval(
                        data,
                        confidence_level=confidence
                    )
                
                if ranked_data:
                    # Display results
                    st.subheader("Ranked Results")
                    
                    # Create DataFrame
                    df_results = pd.DataFrame(ranked_data)
                    
                    # Display table
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Predictive Intervals")
                        fig_bar = px.bar(
                            df_results.head(20),
                            x="id",
                            y="predictive_interval",
                            title="Top 20 Items by Predictive Interval",
                            labels={"predictive_interval": "Predictive Interval", "id": "Item"}
                        )
                        fig_bar.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        st.subheader("Positive Ratio vs Predictive Interval")
                        fig_scatter = px.scatter(
                            df_results,
                            x="positive_ratio",
                            y="predictive_interval",
                            size="total_comments",
                            hover_data=["id"],
                            title="Positive Ratio vs Predictive Interval",
                            labels={
                                "positive_ratio": "Positive Ratio",
                                "predictive_interval": "Predictive Interval",
                                "total_comments": "Total Comments"
                            }
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Top positive/negative
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Top 10 Most Positive")
                        top_positive = get_top_positive_by_interval(data, top_k=10, confidence_level=confidence)
                        for i, item in enumerate(top_positive, 1):
                            st.write(f"{i}. {item.get('id', 'Unknown')}: {item.get('predictive_interval', 0):.3f}")
                    
                    with col2:
                        st.subheader("Top 10 Most Negative")
                        top_negative = get_top_negative_by_interval(data, top_k=10, confidence_level=confidence)
                        for i, item in enumerate(top_negative, 1):
                            st.write(f"{i}. {item.get('id', 'Unknown')}: {item.get('predictive_interval', 0):.3f}")
    
    # Tab 2: Category Analytics
    with tab2:
        st.header("Category-Level Sentiment Analysis")
        st.markdown("Analyze sentiment distribution across different categories.")
        
        # Input method
        input_method = st.radio(
            "Input method:",
            ["Upload CSV file"],
            horizontal=True
        )
        
        df = None
        
        if input_method == "Upload CSV file":
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="CSV file with columns: category, text"
            )
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if "category" in df.columns and "text" in df.columns:
                    st.success(f"Loaded {len(df)} records from CSV")
                else:
                    st.error("CSV must have 'category' and 'text' columns")
        
        if df is not None and not df.empty:
            if st.button("🔍 Analyze Categories", type="primary"):
                if st.session_state.use_api:
                    # Use API
                    try:
                        data = df[["category", "text"]].to_dict('records')
                        response = requests.post(
                            f"{st.session_state.api_url}/analytics/category-sentiment",
                            json={"data": data},
                            timeout=60
                        )
                        response.raise_for_status()
                        results = response.json()
                        category_stats = results.get("category_stats", {})
                        stats_df = pd.DataFrame(list(category_stats.values()))
                    except Exception as e:
                        st.error(f"API request failed: {e}")
                        stats_df = pd.DataFrame()
                else:
                    # Local processing
                    with st.spinner("Analyzing category sentiment..."):
                        analytics = CategoryAnalytics()
                        stats_df = analytics.analyze_from_dataframe(df)
                
                if not stats_df.empty:
                    st.subheader("Category Statistics")
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Top Categories by Predictive Interval")
                        top_cats = stats_df.nlargest(10, "predictive_interval")
                        fig_bar = px.bar(
                            top_cats,
                            x="category",
                            y="predictive_interval",
                            title="Top 10 Categories",
                            labels={"predictive_interval": "Predictive Interval", "category": "Category"}
                        )
                        fig_bar.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        st.subheader("Sentiment Distribution")
                        fig_pie = px.pie(
                            stats_df,
                            values="total_comments",
                            names="category",
                            title="Comments per Category"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Positive/negative ratios
                    st.subheader("Sentiment Ratios by Category")
                    fig_stacked = go.Figure()
                    
                    top_cats = stats_df.nlargest(15, "total_comments")
                    fig_stacked.add_trace(go.Bar(
                        x=top_cats["category"],
                        y=top_cats["positive_ratio"],
                        name="Positive",
                        marker_color="#28a745"
                    ))
                    fig_stacked.add_trace(go.Bar(
                        x=top_cats["category"],
                        y=top_cats["negative_ratio"],
                        name="Negative",
                        marker_color="#dc3545"
                    ))
                    fig_stacked.add_trace(go.Bar(
                        x=top_cats["category"],
                        y=top_cats["neutral_ratio"],
                        name="Neutral",
                        marker_color="#6c757d"
                    ))
                    
                    fig_stacked.update_layout(
                        barmode='stack',
                        title="Sentiment Ratios by Category",
                        xaxis_title="Category",
                        yaxis_title="Ratio",
                        xaxis_tickangle=45
                    )
                    st.plotly_chart(fig_stacked, use_container_width=True)
    
    # Tab 3: Thread Analysis
    with tab3:
        st.header("Thread Length vs Sentiment Correlation")
        st.markdown("Analyze correlation between thread length (number of comments) and sentiment temperature.")
        
        # Input method
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="CSV file with columns: news_id, text"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "news_id" in df.columns and "text" in df.columns:
                st.success(f"Loaded {len(df)} comments from CSV")
                
                if st.button("🔍 Analyze Thread Correlation", type="primary"):
                    if st.session_state.use_api:
                        # Use API
                        try:
                            data = df[["news_id", "text"]].to_dict('records')
                            response = requests.post(
                                f"{st.session_state.api_url}/analytics/thread-analysis",
                                json={"data": data},
                                timeout=60
                            )
                            response.raise_for_status()
                            results = response.json()
                            thread_stats_df = pd.DataFrame(results.get("thread_stats", []))
                            correlation_results = results.get("correlation", {})
                        except Exception as e:
                            st.error(f"API request failed: {e}")
                            thread_stats_df = pd.DataFrame()
                            correlation_results = {}
                    else:
                        # Local processing
                        with st.spinner("Analyzing thread correlation..."):
                            analyzer = ThreadAnalyzer()
                            thread_stats_df, correlation_results = analyzer.analyze_from_dataframe(df)
                    
                    if not thread_stats_df.empty:
                        # Display correlation results
                        st.subheader("Correlation Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Correlation", f"{correlation_results.get('correlation', 0):.3f}")
                        with col2:
                            st.metric("P-value", f"{correlation_results.get('p_value', 1):.4f}")
                        with col3:
                            st.metric("R²", f"{correlation_results.get('r_squared', 0):.3f}")
                        with col4:
                            significant = correlation_results.get('significant', False)
                            st.metric("Significant", "✅ Yes" if significant else "❌ No")
                        
                        st.info(f"**Interpretation:** {correlation_results.get('interpretation', 'N/A')}")
                        
                        # Display thread statistics
                        st.subheader("Thread Statistics")
                        st.dataframe(thread_stats_df, use_container_width=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Thread Length vs Temperature")
                            # Check if statsmodels is available for trendline
                            try:
                                import statsmodels.api as sm
                                has_statsmodels = True
                            except ImportError:
                                has_statsmodels = False
                            
                            fig_scatter = px.scatter(
                                thread_stats_df,
                                x="thread_length",
                                y="temperature",
                                trendline="ols" if has_statsmodels else None,
                                title="Thread Length vs Temperature",
                                labels={
                                    "thread_length": "Thread Length (comments)",
                                    "temperature": "Temperature (negative probability)"
                                }
                            )
                            if not has_statsmodels:
                                st.warning("⚠️ Install `statsmodels` to show trendline: `pip install statsmodels`")
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        with col2:
                            st.subheader("Thread Length Distribution")
                            fig_hist = px.histogram(
                                thread_stats_df,
                                x="thread_length",
                                title="Distribution of Thread Lengths",
                                labels={"thread_length": "Thread Length", "count": "Frequency"}
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Regression line info
                        if "slope" in correlation_results:
                            st.subheader("Regression Analysis")
                            slope = correlation_results.get("slope", 0)
                            intercept = correlation_results.get("intercept", 0)
                            st.write(f"**Equation:** Temperature = {slope:.6f} × Thread Length + {intercept:.4f}")
            else:
                st.error("CSV must have 'news_id' and 'text' columns")


if __name__ == "__main__":
    main()



