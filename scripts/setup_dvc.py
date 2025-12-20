"""Script to set up DVC for data versioning."""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_dvc(remote_name: str = "default", remote_url: Optional[str] = None) -> None:
    """
    Set up DVC repository.
    
    Args:
        remote_name: Remote storage name
        remote_url: Remote storage URL (S3, GCS, Azure, etc.)
    """
    logger.info("Setting up DVC...")
    
    # Initialize DVC if not already initialized
    if not Path(".dvc").exists():
        logger.info("Initializing DVC repository...")
        subprocess.run(["dvc", "init"], check=True)
        logger.info("DVC repository initialized")
    else:
        logger.info("DVC repository already initialized")
    
    # Add remote if provided
    if remote_url:
        logger.info(f"Adding remote: {remote_name} -> {remote_url}")
        try:
            subprocess.run(
                ["dvc", "remote", "add", remote_name, remote_url],
                check=True
            )
            logger.info(f"Remote '{remote_name}' added successfully")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to add remote: {e}")
            logger.info("You can add remote manually with: dvc remote add <name> <url>")
    
    # Add data files to DVC
    data_files = [
        "data/raw/ria_news.tsv",
        "data/raw/vk_news.tsv",
        "data/raw/vk_comments.tsv",
    ]
    
    logger.info("Adding data files to DVC...")
    for data_file in data_files:
        file_path = Path(data_file)
        if file_path.exists():
            try:
                subprocess.run(["dvc", "add", data_file], check=True)
                logger.info(f"Added to DVC: {data_file}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to add {data_file}: {e}")
        else:
            logger.warning(f"Data file not found: {data_file}")
    
    logger.info("DVC setup complete!")
    logger.info("Next steps:")
    logger.info("  1. Commit .dvc files: git add *.dvc .dvcignore")
    logger.info("  2. Push data to remote: dvc push")
    logger.info("  3. Run pipeline: dvc repro")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Set up DVC for data versioning")
    parser.add_argument(
        "--remote-name",
        type=str,
        default="default",
        help="Remote storage name"
    )
    parser.add_argument(
        "--remote-url",
        type=str,
        default=None,
        help="Remote storage URL (S3, GCS, Azure, etc.)"
    )
    
    args = parser.parse_args()
    
    setup_dvc(
        remote_name=args.remote_name,
        remote_url=args.remote_url,
    )

