# Portfolio Reviewer Polish Plan (API + Model Zoo + Dashboards)

This document is a practical checklist + plan to make this project **easy to evaluate** for someone who clicks it from a portfolio website. The goal is that a reviewer can:

- understand the problem + approach in **< 2 minutes**
- run a live demo in **< 5 minutes**
- verify model quality + comparisons without reading the whole codebase
- trust reproducibility (frozen protocol, metrics, thresholds, W&B)

---

## 1) Define the “Reviewer Path” (the top-of-funnel experience)

### 1.1 What a reviewer should be able to do quickly
- **Open README and immediately understand**: the task, data, labels, outputs, and what “success” looks like.
- **Try a demo**:
  - a hosted API endpoint and/or Streamlit app
  - a few example requests and expected responses
- **See proof of rigor**:
  - frozen protocol + fair comparisons
  - optimized threshold metrics
  - W&B runs linked to checkpoints

### 1.2 Create 3 “levels” of engagement
- **Level A (1–2 minutes)**: screenshots + architecture diagram + 1–2 demo GIFs, plus a quick benchmark table.
- **Level B (5 minutes)**: run locally via Docker Compose (one command).
- **Level C (deep dive)**: rerun training/evaluation with frozen protocol and reproduce the benchmark table.

---

## 2) Repo structure + navigation polish

### 2.1 Clean repo root (reduce clutter)
- Add a short `README.md` top section:
  - What it is
  - What it does
  - How to run demo (local + hosted)
  - Results summary table
  - Links (W&B runs, demo URLs)
- Add `docs/` folder:
  - `docs/architecture.md`
  - `docs/screenshots/` (PNG)
  - `docs/demo.md` (curl + UI walkthrough)
  - `docs/results.md` (model zoo + thresholding explanation)

### 2.2 Add a “Quick Links” block at top of README
- **Hosted API**: `/docs`, `/health`
- **Hosted dashboards** (if hosted)
- **W&B project** link
- **Best model checkpoint details** (name + threshold + version)

---

## 3) Make it runnable in ONE command (best reviewer experience)

### 3.1 Provide a Dockerized demo (recommended)
Create:
- `Dockerfile.api` (FastAPI)
- `Dockerfile.streamlit` (dashboards)
- `docker-compose.yml`
- `.dockerignore`

Target experience:

```bash
docker compose up --build
```

Helpful commands (copy/paste):

```bash
# Stop everything
docker compose down

# Tail logs (API is the most useful one to debug)
docker compose logs -f api

# If you changed Dockerfiles / imports and things look “cached”
docker compose build --no-cache api
```

Then reviewer opens:
- `http://localhost:8000/docs` (API docs)
- `http://localhost:8000/health` (API health)
- `http://localhost:8501` (Streamlit multipage app: Evaluation + Analytics + Model Comparison + Sentiment)

Notes:
- On macOS, PyTorch in Docker typically runs on **CPU** (MPS is available on host Python, not inside standard Docker Desktop Linux containers).

### 3.2 Environment variables documented and safe-by-default
Ensure `config/env.example` contains:

- Required (API):
  - `MODEL_PATH=...`
  - `THRESHOLDS_PATH=...`

- Optional:
  - `WANDB_API_KEY=...` (training/experiment logging only)
  - `API_URL=http://localhost:8000` (dashboards “Use API” mode)

Add Makefile targets (reviewer-friendly):
- `make demo` (docker compose up --build)
- `make down`
- `make logs`


---

## 4) Hosted demo options (pick at least one)

### Option A (simple): host only FastAPI
- Deploy FastAPI on **Render / Fly.io / Railway**
- Upload model artifact to the platform (or fetch from a release asset)
- Provide `/docs` and `/health` publicly

### Option B (best “wow”): host API + Streamlit
- Host API (Render/Fly/Railway)
- Host Streamlit separately (Streamlit Community Cloud) pointing to the API

#### Option B execution checklist (recommended for portfolio)

**Goal:** a reviewer clicks a link and can immediately try:
- API docs (`/docs`) + health (`/health`)
- Streamlit dashboards hitting the hosted API (the “Use API” mode)

##### B.1 Pick a hosting combo (simple defaults)
- **API hosting**: Render (easiest) or Railway (also easy). Fly.io is great but slightly more setup.
- **Dashboards hosting**: Streamlit Community Cloud.

##### B.2 Decide how the hosted API gets the model weights
Do **not** commit `.pt` weights into git history. Pick one:
- **GitHub Release asset** (recommended): upload `models/distilmbert_lora_10k_v1.pt` to a release.
- **W&B Artifacts**: host model as an artifact and download at startup (requires W&B token).

##### B.3 Add a “download model” step for deployment
Have a predictable path on the server/container:
- Download the model to something like `/app/models/distilmbert_lora_10k_v1.pt`
- Set env vars:
  - `MODEL_PATH=models/distilmbert_lora_10k_v1.pt`
  - `THRESHOLDS_PATH=config/thresholds.json`
  - `TOKENIZER_NAME=distilbert-base-multilingual-cased` (optional; usually inferred)

##### B.4 Deploy the FastAPI service (Render example)
- Create a new **Web Service** from your repo.
- **Start command**:
  - `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Build command** (one approach):
  - `pip install -r requirements.txt && python scripts/download_model.py --model-id distilmbert_lora_10k_v1`
  - (If you don’t have `download_model.py`, add it or use `curl`/`wget` in the build step.)
- Set environment variables:
  - `MODEL_PATH=models/distilmbert_lora_10k_v1.pt`
  - `THRESHOLDS_PATH=config/thresholds.json`
  - `TOKENIZER_NAME=distilbert-base-multilingual-cased`
- Verify:
  - `https://<api-host>/health`
  - `https://<api-host>/docs`

##### B.5 Deploy Streamlit dashboards (Streamlit Cloud)
- Create **one multipage** Streamlit app:
  - entrypoint: `streamlit_app.py`
  - pages: `pages/` (Evaluation, Analytics, Model Comparison, Sentiment)
- Set Streamlit Cloud **secrets** (or env vars):
  - `API_URL=https://<api-host>`
- Ensure `requirements-streamlit.txt` is used by Streamlit Cloud (or point it to the right requirements file).
- Verify each dashboard works in “Use API” mode (and still works with sample artifacts if API is slow).

##### B.6 Production niceties (don’t skip)
- **CORS**: restrict `allow_origins` in production to your Streamlit domains (or document why it’s open).
- **Cold start**: document expected latency on first request.
- **Rate limits**: optional; document intended usage for reviewers.

##### B.7 Final deliverables (mark Option B “Done”)
- Add URLs to `README.md`:
  - API `/docs` and `/health`
  - 4 dashboard URLs
- Add the same links to your portfolio site.
- Add a short “Model artifact source” note (Release asset / W&B artifact).

### Option C (single platform): host everything in one container
- One Docker image exposing both API and Streamlit (or two services in Compose on Fly)

**Deliverable**: Put the final URLs into README and your portfolio site.

---

## 5) Model artifacts & reproducibility story (critical for ML reviewers)

### 5.1 Decide how reviewers get models
Pick one:
- **Small model committed?** (usually no; too large)
- **Download from release assets**: GitHub Releases `models/*.pt`
- **Download from W&B artifacts**
- **Download from S3/GCS** (overkill unless you already have it)

### 5.2 Provide a “model registry” file
Add `models/REGISTRY.md` or `config/models.yaml` with entries:
- `model_id`
- `checkpoint_url` (or relative path)
- `trained_on_protocol` (e.g. `protocol_10k_1k`)
- `use_snippet`
- `global_threshold`
- `wandb_run_url`

### 5.3 Make reproducibility explicit
Add a section (README + docs) explaining:
- frozen protocol directory (`experiments/model_zoo/protocol_10k_1k`)
- label mapping stability (`tag_to_idx.json`)
- evaluation method (optimized threshold + metrics)
- what “fair comparison” means in this repo

---

## 6) Results presentation (what a reviewer will judge)

### 6.1 Create a single “Results Table” in README
Include:
- model_id
- backbone
- snippet on/off
- opt_threshold
- opt_f1 (and optionally micro_f1)
- link to W&B run

### 6.2 Add plots/screenshots
Add:
- W&B metrics screenshot (train_loss/val_loss curves)
- Streamlit model comparison screenshot
- Example `/classify` outputs (top-5 tags)

---

## 7) FastAPI polish (DX + trust)

### 7.1 Ensure `/health` is reviewer-grade
Must include:
- model_version / model_path
- device
- thresholds_path + sha256
- global_threshold

### 7.2 Add “Example Requests” to docs
In `docs/demo.md` include:
- curl example with `text` alias
- curl example with `title` + `snippet`
- batch classify example

### 7.3 Add minimal rate limiting / CORS note (if hosted)
Document:
- intended usage
- any limits
- CORS policy (especially if called from hosted Streamlit)

---

## 8) Streamlit dashboards polish

### 8.1 Clear entrypoint
Add `dashboards/README.md`:
- what each dashboard is for
- what files it expects (predictions CSV, results JSONs)
- “Use API” meaning (local vs remote inference)

### 8.2 Provide sample data bundles
Add `experiments/sample_outputs/` containing:
- a small predictions CSV
- a metrics JSON
- any analytics CSVs needed

So reviewers can open dashboards even without running training.

---

## 9) Code quality & engineering signals (portfolio strength)

### 9.1 Add CI (basic)
GitHub Actions:
- `python -m compileall`
- unit tests (even small)
- lint/format if you use it

### 9.1.1 CI/CD audit + re-enable (you already have workflows)
This repo already contains `.github/workflows/` and related templates. The polish work is:
- **Audit what exists**: CI, lint, security, release, model deploy, CD.
- **Fix broken/default workflows** so they pass on a fresh public repo:
  - remove placeholder deployment URLs / steps that would fail
  - make “deploy” conditional on secrets being configured
  - ensure lint workflow YAML is valid
- **Decide how far to go** for portfolio:
  - keep CI/lint/security always-on
  - keep “CD deploy” either manual or conditional (so it doesn’t fail for reviewers/forks)
- **Wire optional integrations**:
  - Codecov (optional; keep CI passing even if not configured)
  - container registry (GHCR) publishing (works with default `GITHUB_TOKEN`)

### 9.2 Add tests that matter for this project
High-signal tests:
- model loading + inference shape checks (BERT + DistilBERT)
- API schema accepts `title` and `text`
- thresholds reload works and updates effective threshold

### 9.3 Logging and error messages
Reviewers love:
- clean logs
- actionable errors
- no giant stack traces on expected user errors

---

## 10) Data + licensing + ethics (don’t skip for a portfolio project)

### 10.1 Data provenance
Document:
- data source
- license / usage
- whether raw data is included or not

### 10.2 Add a LICENSE + citations
- Choose MIT/Apache-2.0 (common)
- Include citations for HuggingFace models used

### 10.3 Security baseline (hosted demo)
- Don’t commit API keys
- Don’t allow arbitrary file reads
- Validate input sizes (already partially done)

---

## 11) “Portfolio integration” deliverables

### 11.1 Add a short “Project summary” card
Create `docs/portfolio_blurb.md` with:
- 3–5 bullet points
- 1–2 metrics
- links to demo + W&B
- 1 architecture image

### 11.2 Add a short demo video/GIF
- 30–60 seconds: show API `/docs`, call `/classify`, open Streamlit model comparison, show W&B curve.

---

## 12) Concrete execution checklist (copy/paste)

### Phase 1 (1–2 hours): reviewer experience
- [ ] Add README “Quick Start” + links
- [ ] Add `docs/demo.md` with copy/paste curl commands
- [ ] Add screenshots (API docs + Streamlit + W&B)
- [ ] Add `experiments/sample_outputs/` bundle

### Phase 2 (2–4 hours): one-command run
- [ ] Add Dockerfiles + docker-compose
- [ ] Add `config/env.example` + `Makefile` targets
- [ ] Confirm fresh clone works

### Phase 2.5 (1–3 hours): markdown clutter audit + prune
- [ ] Inventory root `.md` files
- [ ] Read them to extract:
  - unfinished ideas / planned features
  - important debugging notes
  - future tasks worth keeping
- [ ] Consolidate extracted value into ONE place (e.g. `docs/PROJECT_NOTES.md`)
- [ ] Delete redundant `*_COMPLETE.md`, old guides, and progress logs after consolidation

### Phase 3 (2–6 hours): hosted demo
- [ ] Deploy API
- [ ] Deploy Streamlit
- [ ] Add URLs to README + portfolio website

### Phase 4 (2–6 hours): engineering credibility
- [ ] CI pipeline
- [ ] High-signal tests (API schema, thresholds reload, model forward compatibility)
- [ ] License + data provenance section


### Phase 5 (30–120 min): publish the production-grade GitHub repo (FINAL)
- [ ] Create a clean public repo from a final snapshot (so reviewers do not see messy intermediate history)
- [ ] Update README badges/links to match the real repo
- [ ] Publish model artifacts (GitHub Release assets or W&B Artifacts) and link them in README
- [ ] (If hosted) add demo URLs to README + your portfolio website

---

## 13) Definition of Done (DoD)
- A reviewer can run the demo without training anything.
- Results table is visible and backed by artifacts (W&B + metrics JSON).
- API responds correctly and reports correct model/threshold/version.
- Repo has clean docs + clear structure + licensing/data notes.

---

## 14) Git/GitHub (publish only when production-grade) — FINAL STEP

Your portfolio repo should look **production-grade**. That means publishing is the *last packaging step*, not something you do while the repo is still messy.

### 14.1 Recommended portfolio approach (clean history)
Use a clean public history: create a brand-new public repo from the final snapshot so reviewers **cannot** browse messy intermediate commits, file deletions, or old notebooks.

### 14.1.1 Use the snapshot generator (recommended)
This repo includes a script that creates a **reviewer-safe public snapshot** folder:
- `python scripts/make_public_snapshot.py --output-dir public_release --force`

It intentionally excludes:
- local virtual envs (`venv/`, `.venv/`)
- raw data (e.g. `data/**/*.tsv`, `data/**/*.csv`)
- training artifacts / checkpoints (`*.ckpt`, `checkpoints/`)
- model weights (`*.pt`) — publish these as **Release assets** instead
- logs (`wandb/`, `monitoring/predictions/`, `logs/`)

This is the simplest way to guarantee reviewers cannot “find stuff in history”, because the public repo is created from scratch from the snapshot.

### 14.2 Pre-publish checklist
- Confirm `.gitignore` blocks secrets and large artifacts:
  - `venv/`, `wandb/`, `logs/`, `.env`
  - raw datasets
  - `*.pt`, `*.ckpt`
- Ensure dashboards can run using lightweight artifacts:
  - `experiments/sample_outputs/`
- Ensure CI/CD won’t fail on forks:
  - CI/lint/security run normally
  - any deploy steps are conditional on secrets

### 14.3 Model artifacts (don’t commit checkpoints)
Pick one:
- GitHub Releases assets (simple + reviewer-friendly)
- W&B Artifacts (good for ML audiences)
- Git LFS (works, but reviewer friction)

### 14.4 Publish steps (snapshot → public repo)
- Generate a clean snapshot:
  - `python scripts/make_public_snapshot.py --output-dir public_release --force`
- Create a brand-new git repo from the snapshot (do **not** publish from the original folder):
  - `cd public_release`
  - `git init`
  - `git add .`
  - `git commit -m "Initial public release"`
- Create GitHub repo (public) and push:
  - `git branch -M main`
  - `git remote add origin <your_repo_git_url>`
  - `git push -u origin main`
- Update README:
  - real badges/links
  - demo URLs (if hosted)

