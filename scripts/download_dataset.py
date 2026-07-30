#!/usr/bin/env python3
"""Download and extract the MovieLens 32M dataset into the project data/raw folder."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable


DEFAULT_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the MovieLens 32M dataset into the project's raw data folder."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Dataset URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Directory where the dataset files will be extracted (default: project data/raw).",
    )
    parser.add_argument(
        "--zip-name",
        default="ml-32m.zip",
        help="Name of the downloaded zip file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download the archive even if it already exists.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} into {output_dir}")

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        if not members:
            raise RuntimeError("The downloaded archive is empty.")

        for member in members:
            if member.endswith("/"):
                continue

            parts = Path(member).parts
            if len(parts) > 1:
                relative_path = Path(*parts[1:])
            else:
                relative_path = Path(parts[0])

            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"Refusing to extract unsafe path: {member}")

            target_path = output_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member) as source, open(target_path, "wb") as destination:
                shutil.copyfileobj(source, destination)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    zip_path = output_dir / args.zip_name

    if zip_path.exists() and not args.force:
        print(f"Archive already exists at {zip_path}; skipping download.")
    else:
        download_file(args.url, zip_path)

    if not zip_path.exists():
        print("The dataset archive was not found after download.", file=sys.stderr)
        return 1

    extract_zip(zip_path, output_dir)
    print("Dataset ready in", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
