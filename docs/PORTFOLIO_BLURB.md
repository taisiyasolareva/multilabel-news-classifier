# multilabel-news-classifier — Portfolio Blurb

## One-line pitch

Multi-label news classification for Russian articles using transformer models (DistilBERT, RuBERT), deployed on HuggingFace Spaces with reproducible model zoo protocol and interactive dashboards.

## Key metrics

- **F1 0.4518** (DistilBERT + LoRA, threshold 0.15)
- **Protocol**: 10k train / 1k validation (frozen split)
- **141 labels**, snippet-aware input

## Live links

- [Streamlit Classifier](https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier)
- [API docs](https://solarevat-multilabel-news-classifier.hf.space/docs)
- [API health](https://solarevat-multilabel-news-classifier.hf.space/health)

## Technical highlights

- **Model zoo protocol**: Frozen 10k/1k split, threshold optimization, fair comparisons
- **Production**: FastAPI, monitoring, data drift detection, fp16 deployment
- **Full-stack**: API + Streamlit dashboards + CI/CD

## For portfolio website

**Short:** Multi-label news classification API using DistilBERT/RuBERT transformers on Russian news. Reproducible model zoo, HuggingFace Spaces deployment, interactive dashboards.

**Medium:** Built a production-ready multi-label news classification system using DistilBERT and RuBERT, fine-tuned on Russian articles. Implemented a reproducible model zoo protocol (frozen 10k/1k split) with threshold optimization. Deployed on HuggingFace Spaces (fp16, 16GB) with F1 0.4518. Streamlit dashboards for evaluation, analytics, and sentiment.

**Tags:** NLP · Transformers · FastAPI · PyTorch · MLOps · Multi-label Classification
