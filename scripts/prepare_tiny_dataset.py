import argparse
from pathlib import Path

import pandas as pd


def prepare_tiny_dataset(
    input_path: Path,
    output_dir: Path,
    n_total: int = 110,
    n_train: int = 100,
) -> None:
    """
    Create tiny train/val splits from a full labeled dataset.

    Expected input columns (can be adapted as needed):
      - id: unique identifier (will be created if missing)
      - title: news title (Russian)
      - snippet: optional text snippet (can be empty)
      - tags: comma-separated labels, e.g. "политика,экономика"
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect separator based on file extension (CSV vs TSV)
    if input_path.suffix.lower() == ".tsv":
        df = pd.read_csv(input_path, sep="\t")
    else:
        df = pd.read_csv(input_path)

    # Ensure required columns exist or can be constructed
    if "title" not in df.columns:
        raise ValueError("Input dataset must contain a 'title' column.")

    if "tags" not in df.columns:
        raise ValueError("Input dataset must contain a 'tags' column (comma-separated labels).")

    if "snippet" not in df.columns:
        df["snippet"] = ""

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    # Take the first n_total rows (or fewer if dataset is small)
    df_tiny = df.head(n_total).copy()

    tiny_all_path = output_dir / "tiny_all.csv"
    df_tiny.to_csv(tiny_all_path, index=False)

    # Split into train / val
    df_train = df_tiny.head(n_train).copy()
    df_val = df_tiny.tail(len(df_tiny) - len(df_train)).copy()

    tiny_train_path = output_dir / "tiny_train.csv"
    tiny_val_path = output_dir / "tiny_val.csv"

    df_train.to_csv(tiny_train_path, index=False)
    df_val.to_csv(tiny_val_path, index=False)

    print(f"Saved tiny_all to: {tiny_all_path}")
    print(f"Saved tiny_train (n={len(df_train)}) to: {tiny_train_path}")
    print(f"Saved tiny_val   (n={len(df_val)}) to: {tiny_val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tiny train/val splits for quick finetuning.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the full labeled dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save tiny_all.csv, tiny_train.csv, tiny_val.csv (default: data).",
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=110,
        help="Total number of samples to keep for tiny dataset (default: 110).",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=100,
        help="Number of samples to use for training (default: 100, rest go to val).",
    )

    args = parser.parse_args()

    prepare_tiny_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        n_total=args.n_total,
        n_train=args.n_train,
    )


if __name__ == "__main__":
    main()


