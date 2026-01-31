# Pre-Push Release Gate Report — `public_release`

**Repo:** `/Users/tay/Development/portfolio projects/nlp/public_release`  
**Expected public demo link:** `https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier`  
**Date:** 2025-01-31

---

## 1) Repo discovery + git state

| Item | Value |
|------|--------|
| **Current branch** | `main` |
| **Tracking** | `origin/main` — **ahead by 2 commits** |
| **Remotes** | `origin` → `https://github.com/taisiyasolareva/multilabel-news-classifier.git` (fetch/push) |
| | `hf` → `https://huggingface.co/spaces/solarevat/multilabel-news-classifier` (fetch/push) |

### `git status` (summary)

- **Staged:** none  
- **Modified (not staged):** 11 files  
- **Untracked:** 1 file  

### Modified files

- `.github/workflows/ci.yml`
- `LICENSE`
- `README.md`
- `README_SPACES.md`
- `pages/0_Classifier.py`
- `railway.json`
- `railway.toml`
- `scripts/convert_checkpoint_fp16.py`
- `utils/data_processing.py`
- `utils/russian_text_utils.py`
- `utils/text_processing.py`

### Untracked

- `docs/PORTFOLIO_BLURB.md`

### Last 5 commits

1. `b44d065` — fix(spaces): use uvicorn directly in Dockerfile for HuggingFace Spaces  
2. `e5bb1eb` — fix(spaces): add HuggingFace Spaces configuration to README  
3. `1f901bc` — feat(deploy): add HuggingFace Spaces support  
4. `1b82a45` — feat(deploy): add Railway.app support  
5. `c432b47` — fix(api): add pandas to requirements-api.txt (needed by monitoring)  

**Local vs origin:** `main` is **ahead 2** of `origin/main` (0 behind).

---

## 2) “What will be committed” (if `git add -A`)

| Category | Paths |
|----------|--------|
| **Docs** | `README.md`, `README_SPACES.md`, `docs/PORTFOLIO_BLURB.md` |
| **Source** | `pages/0_Classifier.py`, `utils/data_processing.py`, `utils/russian_text_utils.py`, `utils/text_processing.py`, `scripts/convert_checkpoint_fp16.py` |
| **Configs** | `railway.json`, `railway.toml` |
| **CI** | `.github/workflows/ci.yml` |
| **Other** | `LICENSE` |

**Suspicious:** None. No results/experiments or data files would be added by `git add -A` beyond what is already tracked.

### Tracked files that are gitignored

These are **already tracked**; `.gitignore` does not untrack them. Do **not** use `git rm` (per your constraints). For future cleanup you could consider `git rm --cached` in a separate change.

| Tracked path | Gitignore rule |
|-------------|----------------|
| `experiments/model_zoo/distilmbert_v1_eval.csv` | `experiments/model_zoo/*.csv` (line 127) |
| `experiments/model_zoo/rubert_base_v1_eval.csv` | same |
| `experiments/model_zoo/rubert_snippet_ablation_v1_eval.csv` | same |
| `experiments/model_zoo_summary.csv` | (implied by model_zoo/*.csv pattern — actually `experiments/model_zoo_summary.csv` is explicitly in .gitignore line 128) |
| `experiments/analytics_sentiment_counts.csv` | line 129 |

So: **tracked but gitignored** — `experiments/model_zoo/*.csv`, `experiments/model_zoo_summary.csv`, `experiments/analytics_sentiment_counts.csv`. They will remain in history if you push; no destructive fix requested.

---

## 3) Secrets + sensitive info scan (STRICT)

### Absolute user paths (BLOCKER — must fix before push)

| File | Snippet | Fix |
|------|---------|-----|
| `experiments/results/rubert_snippet_ablation_v1.json` | `"checkpoint_path": "/Users/tay/Development/portfolio projects/nlp/models/rubert_snippet_ablation_v1.pt"` | Replace with `"checkpoint_path": "models/rubert_snippet_ablation_v1.pt"` |
| `experiments/results/distilmbert_v1.json` | `"checkpoint_path": "/Users/tay/Development/portfolio projects/nlp/models/distilmbert_v1.pt"` | Replace with `"checkpoint_path": "models/distilmbert_v1.pt"` |

### .env / keys

- **`.env` / `.env.*`:** Not tracked. `config/env.example` is present (no leading dot) — OK.  
- **`README.md` line 676:** `WANDB_API_KEY=your_key_here` — example value in “Create `.env` file” block; **OK** (placeholder, not a real key).  
- **`config/logging/wandb.yaml`:** Comment about `WANDB_API_KEY` env var — OK.  
- **`scripts/train_model.py`:** Help text mentioning `WANDB_API_KEY` — OK.  
- **`api/main.py` line 55:** Docstring example `CORS_ALLOW_ORIGINS=https://my-app.streamlit.app,...` — OK (example only).

### Dataset / internal hostnames

- No dataset content leaks or internal-only hostnames/URLs found in scanned files.

---

## 4) Large files / artifact / dataset leak scan

- **Large tracked files:** None over ~500 KB in the repo.  
- **Experiment artifacts:**  
  - `experiments/results/*.json` — ~4 KB each; OK.  
  - `experiments/sample_outputs/distilmbert_lora_10k_v1_val_preds_sample_50.csv` — ~104 KB; OK.  
- **`.gitignore` coverage:**  
  - Model files (`*.pt`, `*.pth`, etc.), `checkpoints/`, `data/raw/`, `data/processed/`, `experiments/dashboard_eval_predictions.csv`, `experiments/predictions/*.csv`, etc. are ignored.  
- **Tracked despite ignore:** See §2 (e.g. `experiments/model_zoo/*.csv`, `experiments/model_zoo_summary.csv`, `experiments/analytics_sentiment_counts.csv`). No change requested (no destructive commands).

**Safe to push (size/artifact):** Yes, once path leaks in §3 are fixed.  
**Must exclude/untrack:** No additional exclusions required for this gate (path fixes only).

---

## 5) Documentation release gate (public-facing)

Public-facing set: `README.md`, `README_SPACES.md`, `docs/DEMO.md`, `docs/ARCHITECTURE.md`, `docs/RESULTS.md`, `docs/PORTFOLIO_BLURB.md`.

### Required Streamlit demo link

- **Required:** `https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier`  
- **Status:** **Not present** in any of the above docs.  
- README and README_SPACES mention HuggingFace Spaces **API** URLs (e.g. `solarevat-multilabel-news-classifier.hf.space`), not the Streamlit Community Cloud Classifier app.  
- **Action:** Add this link to the public docs (see punchlist).

### Placeholders

- Grep for `<PASTE_...>`, `[TBD]`, `[fill]`, “paste URLs here”, “placeholder implementation”: **none** in repo. OK.

### Docs vs runtime

- **DEMO.md vs docker-compose:**  
  - **DEMO.md** (Option A): “Evaluation dashboard: `http://localhost:8502`”, “Analytics … `http://localhost:8503`”, “Model comparison … `http://localhost:8504`”, “Sentiment … `http://localhost:8504`”.  
  - **docker-compose.yml:** Single Streamlit service, one port **8501**, multipage app (`streamlit run streamlit_app.py`).  
  - **Mismatch:** DEMO.md describes four separate apps on 8502–8504; reality is **one** multipage app on **8501**.  
- **Suggested DEMO.md fix:** Describe one app at `http://localhost:8501` with sidebar pages (Classifier, Evaluation, Analytics, Model Comparison, Sentiment), and remove references to 8502–8504.

### Broken / internal-only links

- **README.md line 1217:** “Portfolio polish plan”: `PORTFOLIO_REVIEWER_POLISH_PLAN.md` — **file does not exist** in repo.  
- **Action:** Remove this bullet or replace with an existing doc (e.g. `docs/PORTFOLIO_BLURB.md`).

---

## 6) Reproducibility & install sanity

- **Dependency files:** `requirements.txt`, `requirements-api.txt`, `requirements-streamlit.txt`, `requirements-test.txt` exist and are consistent with layout (api, streamlit, tests).  
- **Install/run:** README clone path says `russian-news-classification`; repo is `multilabel-news-classifier` — minor naming inconsistency only.  
- **Recommended pre-push (do not run unless asked):**  
  - `python -m compileall api analysis dashboards evaluation experiments models monitoring pages scripts tests streamlit_app.py app.py`  
  - `pytest tests/ -v`  
  - `docker compose build`  

CI (`.github/workflows/ci.yml`) runs compile; no pytest in the snippet seen — optional to add later.

---

## 7) Code quality quick checks

- **WIP/TODO in `pages/`:** None found. OK.  
- **Internal-only docs linked from README:** `docs/PROJECT_NOTES.md` is linked as “Project notes (dev history + remaining polish gaps)” — acceptable as “internal-ish”; no TODO loop in **public** docs.  
- **“Public release snapshot” / autogenerated:** No misleading “public_release snapshot” or “autogenerated” language in the **public** doc set. `scripts/make_public_snapshot.py` and `PUBLIC_RELEASE_README.md` (gitignored) refer to snapshot; that’s fine.

---

## 8) Verdict + punchlist

### Verdict: **GO** (all punchlist items applied)

~~Blockers (all addressed):~~  
1. Absolute paths in experiment JSONs → fixed.  
2. Required Streamlit demo link → added to README, README_SPACES, DEMO, PORTFOLIO_BLURB, ARCHITECTURE, RESULTS.  
3. DEMO.md vs docker-compose → DEMO.md updated to single app on 8501.  
4. README broken link → replaced with `docs/PORTFOLIO_BLURB.md`.

### Ordered punchlist

| # | File | What to change | Replacement / action |
|---|------|----------------|------------------------|
| 1 | `experiments/results/rubert_snippet_ablation_v1.json` | Remove absolute path | Line 3: `"checkpoint_path": "models/rubert_snippet_ablation_v1.pt"` |
| 2 | `experiments/results/distilmbert_v1.json` | Remove absolute path | Line 3: `"checkpoint_path": "models/distilmbert_v1.pt"` |
| 3 | `README.md` | Add required Streamlit demo link | In “Hosted Demo (HuggingFace Spaces)” or “Level A/B” section add: “**Streamlit Classifier (live):** https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier” |
| 4 | `README_SPACES.md` | Add Streamlit app link | After API endpoints, add: “**Streamlit Classifier app:** https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier” |
| 5 | `docs/DEMO.md` | Align with single multipage app on 8501 | Replace Option A with: “Then open: API docs: `http://localhost:8000/docs`, API health: `http://localhost:8000/health`, **Streamlit multipage app:** `http://localhost:8501` (Classifier, Evaluation, Analytics, Model Comparison, Sentiment in sidebar).” Remove 8502, 8503, 8504. |
| 6 | `docs/PORTFOLIO_BLURB.md` | Add required demo link | In “Live Demo Links” add: “**Streamlit Classifier:** https://multilabel-news-classifier-flqhorz4ntfkpmtn3jwdvy.streamlit.app/Classifier” |
| 7 | `docs/ARCHITECTURE.md` | Optional: add demo link | If you want it in architecture doc, add the same link in a “Live demo” line. |
| 8 | `docs/RESULTS.md` | Optional: add demo link | Same as above if desired. |
| 9 | `README.md` | Fix broken doc link | Remove the bullet “**Portfolio polish plan**: `PORTFOLIO_REVIEWER_POLISH_PLAN.md`” or change to “**Portfolio summary**: `docs/PORTFOLIO_BLURB.md`”. |

Optional (non-blocking):

- **Dockerfile.streamlit:** Add `COPY pages/ pages/` so a standalone `docker build -f Dockerfile.streamlit .` includes the multipage app (docker-compose already mounts `.` so it works today).

---

## 9) Safe staging + commit + push plan (do not execute)

### Option A — Explicit `git add` (preferred)

After applying the punchlist fixes:

```bash
git add \
  .github/workflows/ci.yml \
  LICENSE \
  README.md \
  README_SPACES.md \
  docs/DEMO.md \
  docs/PORTFOLIO_BLURB.md \
  docs/ARCHITECTURE.md \
  docs/RESULTS.md \
  experiments/results/rubert_snippet_ablation_v1.json \
  experiments/results/distilmbert_v1.json \
  pages/0_Classifier.py \
  railway.json \
  railway.toml \
  scripts/convert_checkpoint_fp16.py \
  utils/data_processing.py \
  utils/russian_text_utils.py \
  utils/text_processing.py
```

Then:

```bash
git status   # verify only intended paths are staged
git commit -m "chore(release): initial public snapshot"
git push origin main
```

(Include `docs/ARCHITECTURE.md` and `docs/RESULTS.md` in `git add` only if you changed them.)

### Option B — `git add -A` with verification

```bash
git add -A
git status   # required: confirm no unintended files (e.g. no .env, no large data)
git diff --cached --stat  # optional: review what will be committed
git commit -m "chore(release): initial public snapshot"
git push origin main
```

### Commit message

Use exactly: **`chore(release): initial public snapshot`**

### Push

- **Command:** `git push origin main`  
- Branch is already tracking `origin/main`; no need to set upstream.

---

**End of report.**
