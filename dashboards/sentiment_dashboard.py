"""Streamlit dashboard for sentiment analysis."""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import requests
import json

# Ensure project root is on sys.path so that `analysis` and other packages can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.sentiment_analyzer import SentimentAnalyzer

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="😊",
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
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    </style>
""", unsafe_allow_html=True)

def _ensure_session_state() -> None:
    """
    Ensure required session state keys exist.

    Important: in Streamlit multipage apps, imported modules may not re-run on every script rerun.
    Initializing state only at import-time can lead to AttributeError on `st.session_state.<key>`.
    """
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "results" not in st.session_state:
        st.session_state.results = []
    if "use_api" not in st.session_state:
        st.session_state.use_api = True

    # Prefer Streamlit secrets (Streamlit Cloud), then env vars (Docker/local), then localhost.
    try:
        secret_api_url = st.secrets.get("API_URL")  # type: ignore[attr-defined]
    except Exception:
        secret_api_url = None
    desired_api_url = secret_api_url or os.environ.get("API_URL") or "http://localhost:8000"

    if "api_url" not in st.session_state or st.session_state.api_url == "http://localhost:8000":
        st.session_state.api_url = desired_api_url


def initialize_analyzer():
    """Initialize sentiment analyzer."""
    if st.session_state.analyzer is None:
        with st.spinner("Loading sentiment model..."):
            try:
                st.session_state.analyzer = SentimentAnalyzer()
                st.success("Sentiment model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load sentiment model: {e}")
                st.session_state.analyzer = None


def analyze_with_api(text: str) -> Dict:
    """Analyze sentiment using API."""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/sentiment/analyze",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def analyze_batch_with_api(texts: List[str]) -> List[Dict]:
    """Analyze sentiment batch using API."""
    try:
        response = requests.post(
            f"{st.session_state.api_url}/sentiment/batch",
            json={"texts": texts},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return []


def get_sentiment_color(label: str) -> str:
    """Get color for sentiment label."""
    colors = {
        "POSITIVE": "#28a745",
        "NEGATIVE": "#dc3545",
        "NEUTRAL": "#6c757d"
    }
    return colors.get(label.upper(), "#6c757d")


def main():
    """Main dashboard function."""
    _ensure_session_state()
    # Header
    st.markdown('<h1 class="main-header">😊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        use_api = st.checkbox(
            "Use API (instead of local model)",
            value=st.session_state.use_api,
            help="Use FastAPI endpoint for sentiment analysis"
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
                        f"{api_url}/sentiment/health",
                        timeout=5
                    )
                    if response.status_code == 200:
                        st.success("✅ API is healthy!")
                        st.json(response.json())
                    else:
                        st.warning(f"API returned status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ API connection failed: {e}")
        else:
            if st.button("Load Sentiment Model"):
                initialize_analyzer()
            
            if st.session_state.analyzer:
                st.success("✅ Model loaded")
                st.info(f"Model: {st.session_state.analyzer.model_name}")
                st.info(f"Device: {st.session_state.analyzer.device}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis", "📈 Statistics"])
    
    # Tab 1: Single Analysis
    with tab1:
        st.header("Analyze Single Text")
        
        text_input = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Введите текст для анализа тональности...",
            help="Enter Russian text for sentiment analysis"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary")
        
        if analyze_button and text_input:
            if st.session_state.use_api:
                result = analyze_with_api(text_input)
                if result:
                    st.session_state.results.append(result)
            else:
                if st.session_state.analyzer is None:
                    initialize_analyzer()
                
                if st.session_state.analyzer:
                    with st.spinner("Analyzing sentiment..."):
                        result = st.session_state.analyzer.analyze(text_input)
                        st.session_state.results.append(result)
            
            if result:
                # Display result
                label = result.get("label", "NEUTRAL")
                score = result.get("score", 0.0)
                color = get_sentiment_color(label)
                
                st.markdown("### Result")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", label)
                
                with col2:
                    st.metric("Confidence", f"{score:.2%}")
                
                with col3:
                    # Progress bar for confidence
                    st.progress(score)
                
                # Visual indicator
                if label == "POSITIVE":
                    st.success(f"✅ Positive sentiment ({score:.2%} confidence)")
                elif label == "NEGATIVE":
                    st.error(f"❌ Negative sentiment ({score:.2%} confidence)")
                else:
                    st.info(f"⚪ Neutral sentiment ({score:.2%} confidence)")
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Analysis")
        
        # Input methods
        input_method = st.radio(
            "Input method:",
            ["Text area (one per line)", "Upload CSV file"],
            horizontal=True
        )
        
        texts = []
        
        if input_method == "Text area (one per line)":
            batch_text = st.text_area(
                "Enter texts (one per line):",
                height=200,
                help="Enter multiple texts, one per line"
            )
            if batch_text:
                texts = [line.strip() for line in batch_text.split("\n") if line.strip()]
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV file",
                type=["csv"],
                help="CSV file with 'text' column"
            )
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if "text" in df.columns:
                    texts = df["text"].dropna().tolist()
                    st.success(f"Loaded {len(texts)} texts from CSV")
                else:
                    st.error("CSV file must have a 'text' column")
        
        if texts:
            st.info(f"📝 {len(texts)} texts ready for analysis")
            
            if st.button("🔍 Analyze Batch", type="primary"):
                if st.session_state.use_api:
                    results = analyze_batch_with_api(texts)
                else:
                    if st.session_state.analyzer is None:
                        initialize_analyzer()
                    
                    if st.session_state.analyzer:
                        with st.spinner(f"Analyzing {len(texts)} texts..."):
                            results = st.session_state.analyzer.analyze_batch(texts)
                
                if results:
                    # Create results DataFrame
                    df_results = pd.DataFrame(results)
                    
                    # Display results table
                    st.subheader("Results")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Distribution
                    distribution = df_results["label"].value_counts().to_dict()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Distribution")
                        fig_pie = px.pie(
                            values=list(distribution.values()),
                            names=list(distribution.keys()),
                            title="Sentiment Distribution",
                            color_discrete_map={
                                "POSITIVE": "#28a745",
                                "NEGATIVE": "#dc3545",
                                "NEUTRAL": "#6c757d"
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.subheader("Statistics")
                        st.metric("Total Texts", len(results))
                        for label, count in distribution.items():
                            percentage = (count / len(results)) * 100
                            st.metric(
                                label,
                                f"{count} ({percentage:.1f}%)"
                            )
                    
                    # Score distribution
                    st.subheader("Score Distribution")
                    fig_hist = px.histogram(
                        df_results,
                        x="score",
                        color="label",
                        nbins=20,
                        title="Sentiment Score Distribution",
                        color_discrete_map={
                            "POSITIVE": "#28a745",
                            "NEGATIVE": "#dc3545",
                            "NEUTRAL": "#6c757d"
                        }
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Tab 3: Statistics
    with tab3:
        st.header("Analysis Statistics")
        
        if st.session_state.results:
            df_history = pd.DataFrame(st.session_state.results)
            
            # Overall statistics
            st.subheader("Overall Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Analyses", len(df_history))
            
            with col2:
                avg_score = df_history["score"].mean()
                st.metric("Average Score", f"{avg_score:.2%}")
            
            with col3:
                positive_pct = (df_history["label"] == "POSITIVE").sum() / len(df_history) * 100
                st.metric("Positive %", f"{positive_pct:.1f}%")
            
            with col4:
                negative_pct = (df_history["label"] == "NEGATIVE").sum() / len(df_history) * 100
                st.metric("Negative %", f"{negative_pct:.1f}%")
            
            # Distribution chart
            st.subheader("Sentiment Distribution Over Time")
            distribution = df_history["label"].value_counts()
            
            fig_bar = px.bar(
                x=distribution.index,
                y=distribution.values,
                title="Sentiment Distribution",
                labels={"x": "Sentiment", "y": "Count"},
                color=distribution.index,
                color_discrete_map={
                    "POSITIVE": "#28a745",
                    "NEGATIVE": "#dc3545",
                    "NEUTRAL": "#6c757d"
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Recent results
            st.subheader("Recent Results")
            st.dataframe(df_history.tail(10), use_container_width=True)
            
            # Clear history
            if st.button("🗑️ Clear History"):
                st.session_state.results = []
                st.rerun()
        else:
            st.info("No analysis results yet. Start analyzing texts in the other tabs!")


if __name__ == "__main__":
    main()



