"""
Streamlit multipage app entrypoint.

This repo exposes multiple dashboards as pages under `pages/`:
- Evaluation
- Analytics
- Model Comparison
- Sentiment

Run:
  streamlit run streamlit_app.py
"""

import os
import streamlit as st

st.set_page_config(
    page_title="Russian News Classification — Dashboards",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared defaults across pages
if "api_url" not in st.session_state:
    st.session_state.api_url = os.environ.get("API_URL", "http://localhost:8000")
if "use_api" not in st.session_state:
    st.session_state.use_api = True

st.title("Russian News Classification — Dashboards")

st.markdown(
    """
This is a **multipage Streamlit app**. Use the left sidebar to navigate between pages:

- **Evaluation**: upload predictions CSV / inspect metrics
- **Analytics**: predictive intervals, category analytics, thread analysis
- **Model comparison**: compare experiment results
- **Sentiment**: analyze text sentiment (local or via API)
"""
)

with st.sidebar:
    st.header("Global Settings")
    st.session_state.use_api = st.checkbox(
        "Use API by default",
        value=st.session_state.use_api,
        help="When enabled, pages will prefer calling the FastAPI backend.",
    )
    st.session_state.api_url = st.text_input(
        "API_URL",
        value=st.session_state.api_url,
        help="FastAPI base URL. Local: http://localhost:8000. Docker Compose: http://api:8000",
    )

st.info(
    f"Current API_URL: `{st.session_state.api_url}` — open API docs at `{st.session_state.api_url}/docs`"
)

