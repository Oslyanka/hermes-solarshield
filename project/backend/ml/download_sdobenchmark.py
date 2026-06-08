from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def download_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        "fhnw-i4ds/sdobenchmark",
        "-p",
        str(output_dir),
        "--unzip",
    ]
    print("Executando:", " ".join(command))
    subprocess.run(command, check=True)
    print(f"SDOBenchmark baixado em {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa o dataset SDOBenchmark via Kaggle API.")
    parser.add_argument("--output-dir", default="data/sdobenchmark", type=Path)
    args = parser.parse_args()
    download_dataset(args.output_dir)

