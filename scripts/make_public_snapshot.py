"""
Create a clean, reviewer-safe public snapshot of this project.

Why:
- You currently have large/private artifacts locally (models/, raw data, checkpoints/, venv/)
- Even if you delete them later, they can be recovered from git history *if committed*
- This script creates a fresh, allowlist-based folder you can publish as a brand-new repo

Usage:
  python scripts/make_public_snapshot.py --output-dir public_release --force

Then publish ONLY the snapshot:
  cd public_release
  git init
  git add .
  git commit -m "Initial public release"
  git remote add origin <your github repo>
  git push -u origin main
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


INCLUDE_DIRS = [
    "api",
    "analysis",
    "config",
    "dashboards",
    "docs",
    "evaluation",
    "experiments",
    "models",  # python code only; weights excluded by patterns below
    "monitoring",  # python code only; prediction logs excluded by patterns below
    "nginx",
    "pages",
    "scripts",
    "tests",
]

# Top-level files needed for the hosted demo / one-command demo / reviewer docs
INCLUDE_FILES = [
    "README.md",
    "PORTFOLIO_REVIEWER_POLISH_PLAN.md",
    "docker-compose.yml",
    "docker-compose.prod.yml",
    "Dockerfile.api",
    "Dockerfile.streamlit",
    ".dockerignore",
    ".gitignore",
    "Makefile",
    "render.yaml",
    "requirements.txt",
    "requirements-streamlit.txt",
    "requirements-test.txt",
    "pytest.ini",
]


EXCLUDE_GLOBS = [
    # Python/IDE cruft
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
    "**/.idea/**",
    "**/.vscode/**",
    "**/.DS_Store",
    # Local envs / caches
    "venv/**",
    ".venv/**",
    # Secrets (never publish)
    ".env",
    ".env.*",
    "**/*.pem",
    "**/*.key",
    # Model artifacts (publish weights as GitHub Release assets instead)
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "checkpoints/**",
    # Data (publish only tiny safe samples if you want)
    "data/**/*.csv",
    "data/**/*.tsv",
    "data/**/*.parquet",
    "data/news_data/**",
    # Monitoring logs
    "monitoring/predictions/**",
    # Experiment CSVs that contain raw text (keep private unless you have explicit rights to publish)
    "experiments/analytics_category_data.csv",
    "experiments/analytics_thread_data.csv",
    # Training logs
    "wandb/**",
    "logs/**",
    "tensorboard_logs/**",
    "outputs/**",
    "multirun/**",
]

# Allowlist exceptions inside excluded areas
ALLOWLIST_EXCEPTIONS = [
    # keep tiny sample CSVs for demos/tests
    "data/tiny_*.csv",
    # keep curated, non-text sample outputs
    "experiments/sample_outputs/**",
]


def _matches_any(path_str: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, pat) for pat in patterns)


def _should_exclude(rel_path: Path) -> bool:
    rel_str = rel_path.as_posix()

    # allowlist exceptions win
    if _matches_any(rel_str, ALLOWLIST_EXCEPTIONS):
        return False

    return _matches_any(rel_str, EXCLUDE_GLOBS)


def copy_tree(src_dir: Path, dst_dir: Path) -> None:
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(REPO_ROOT)

        # prune excluded dirs early
        pruned_dirs = []
        for d in list(dirs):
            rel_d = (rel_root / d)
            if _should_exclude(rel_d) or _matches_any(rel_d.as_posix() + "/", EXCLUDE_GLOBS):
                dirs.remove(d)
                pruned_dirs.append(d)

        # ensure destination directory exists
        dest_root = dst_dir / rel_root
        dest_root.mkdir(parents=True, exist_ok=True)

        for f in files:
            rel_f = rel_root / f
            if _should_exclude(rel_f):
                continue

            src_f = root_path / f
            dst_f = dest_root / f
            dst_f.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_f, dst_f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "public_release"),
        help="Destination folder for the public snapshot",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete output-dir if it exists",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"Output dir already exists: {out_dir} (use --force)")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy allowlisted directories
    for d in INCLUDE_DIRS:
        src = REPO_ROOT / d
        if not src.exists():
            continue
        copy_tree(src, out_dir)

    # Copy allowlisted top-level files
    for f in INCLUDE_FILES:
        src = REPO_ROOT / f
        if not src.exists():
            continue
        dst = out_dir / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Add a tiny README to explain publishing flow
    (out_dir / "PUBLIC_RELEASE_README.md").write_text(
        "\n".join(
            [
                "# Public release snapshot",
                "",
                "This folder was generated by `scripts/make_public_snapshot.py`.",
                "",
                "It intentionally excludes:",
                "- local datasets",
                "- local virtual environments",
                "- model weights / checkpoints (publish weights as release assets instead)",
                "- logs (wandb/, monitoring/predictions/, etc.)",
                "",
                "## Publish",
                "",
                "```bash",
                "git init",
                "git add .",
                "git commit -m \"Initial public release\"",
                "```",
                "",
                "Then create a new GitHub repo and push this folder.",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"✅ Public snapshot created: {out_dir}")
    print("Next: `cd public_release` and publish that folder as a brand-new repo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


