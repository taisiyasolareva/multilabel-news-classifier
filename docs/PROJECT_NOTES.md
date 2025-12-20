# Project Notes (Consolidated from legacy docs)

This file preserves the **useful** information from the historical “*_COMPLETE.md / *_PLAN.md” docs that were produced while building the project, without keeping dozens of duplicative markdown files in the repo root.

---

## What’s implemented (high signal)

- **API**: FastAPI model serving + analytics + sentiment + monitoring endpoints (see `api/`).
- **Dashboards**: Streamlit dashboards for Evaluation, Analytics, Model Comparison, Sentiment (see `dashboards/`).
- **Training/Eval**:
  - Training script supports modern transformer fine-tuning + LoRA (see `scripts/train_model.py`).
  - Evaluation supports metrics + threshold optimization and emits artifacts for dashboards (see `scripts/evaluate.py` + `experiments/results/*.json`).
  - Fair comparison protocol exists: `experiments/model_zoo/protocol_10k_1k`.
- **CI/CD**: GitHub Actions workflows exist in `.github/workflows/` (CI, lint, security, release, CD, model-deploy).

---

## Repo conventions that matter (so reviewers don’t get lost)

### Model zoo artifacts (canonical)
- **Protocol**: `experiments/model_zoo/protocol_10k_1k` (splits + `tag_to_idx.json`).
- **Predictions**: `experiments/predictions/<model_id>_val_preds.csv`.
- **Metrics**: `experiments/results/<model_id>.json` (includes optimized threshold).
- **Served model + threshold**: `config/thresholds.json`.

### Dashboards
- Dashboards can run from **uploaded CSV/JSON artifacts** (no training required).
- Dashboards can optionally call the API (“Use API”) — useful for analytics/sentiment and for remote deployments.

---

## GitHub publishing checklist (the practical bits)

### Hygiene before first push
- Confirm `.gitignore` prevents committing:
  - `venv/`, `wandb/`, `logs/`, `.env`, datasets, and large model checkpoints.
- Decide model distribution strategy:
  - **Recommended**: publish checkpoints via **GitHub Releases** (or W&B Artifacts), and provide a small download script.

### CI/CD notes
- CI should be “always green” on forks:
  - CD deploy steps should be **conditional** on secrets like `STAGING_API_URL` / `PRODUCTION_API_URL`.
- Optional integrations:
  - Codecov is optional; CI should still pass if it’s not configured.

---

## Known “portfolio polish” gaps (things to do next)

- **Publish to GitHub** (currently local-only).
- **Hosted demo**:
  - API on Render/Fly/Railway
  - Streamlit dashboards on Streamlit Cloud / HuggingFace Spaces
- **Model artifacts**: make downloadable by reviewers (release assets or W&B artifacts).
- **Screenshots/GIFs**: add to `docs/screenshots/` and link in README.
- **License + data provenance**: add a real `LICENSE` and clarify dataset source/constraints.

---

## Future ideas backlog (optional)

- Add a strict **title-only Distil** variant for clean ablation vs `use_snippet=True`.
- Add a small `models/REGISTRY.md` (or `config/models.yaml`) that maps model_id → checkpoint URL → threshold → W&B run.


