import streamlit as st
import requests
import os


def _get_api_url_default() -> str:
    try:
        secret_val = st.secrets.get("API_URL")  # type: ignore[attr-defined]
    except Exception:
        secret_val = None
    return secret_val or os.environ.get("API_URL") or st.session_state.get("api_url") or "http://localhost:8000"


st.set_page_config(
    page_title="Classifier",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏷️ News Tag Classifier")
st.caption("Paste a title and optional snippet, then call the FastAPI `/classify` endpoint.")

if "api_url" not in st.session_state:
    st.session_state.api_url = _get_api_url_default()

with st.sidebar:
    st.header("API")
    st.session_state.api_url = st.text_input("API_URL", value=st.session_state.api_url)
    st.markdown("Tip: in Streamlit Cloud, set `API_URL` in **Secrets** to your Render API base URL.")

col1, col2 = st.columns([2, 1])

with col1:
    title = st.text_area("Title", height=100, placeholder="Enter news title…")
    snippet = st.text_area("Snippet (optional)", height=140, placeholder="Enter snippet…")

with col2:
    st.subheader("Options")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=10)
    threshold = st.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="0.50 uses API defaults; if thresholds config is loaded, the API will apply it when threshold is 0.50.",
    )
    timeout_s = st.slider("Timeout (seconds)", min_value=5, max_value=120, value=30)

run = st.button("Classify", type="primary", use_container_width=True)

if run:
    if not title.strip():
        st.error("Please enter a title.")
    else:
        api_url = (st.session_state.api_url or "").rstrip("/")
        if not api_url:
            st.error("API_URL is empty. Set it in the sidebar or Streamlit Secrets.")
        else:
            payload = {
                "title": title.strip(),
                "snippet": snippet.strip() if snippet.strip() else None,
                "top_k": int(top_k),
                "threshold": float(threshold),
            }
            try:
                resp = requests.post(
                    f"{api_url}/classify",
                    json=payload,
                    timeout=int(timeout_s),
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                st.error(f"API request failed: {e}")
                st.stop()

            preds = data.get("predictions", []) or []
            st.success(
                f"Model: `{data.get('model_version')}` | Threshold used: `{data.get('threshold')}` | "
                f"Predictions: `{len(preds)}`"
            )

            if preds:
                for p in preds:
                    tag = p.get("tag", "")
                    score = float(p.get("score", 0.0))
                    st.write(f"**{tag}** — {score:.3f}")
            else:
                st.info("No predictions above threshold.")



