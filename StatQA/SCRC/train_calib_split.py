#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


TARGET_SPLITS = ("probe-train", "conformal-calib")


def read_train_rows(train_csv: Path):
    with train_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError(f"No header found in training CSV: {train_csv}")
        rows = list(reader)
        return reader.fieldnames, rows


def read_manifest_indices(manifest_csv: Path):
    split_to_indices = {split: [] for split in TARGET_SPLITS}

    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"split", "source_index"}
        if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
            raise RuntimeError(
                f"Manifest must contain columns {sorted(required_cols)}: {manifest_csv}"
            )

        for row in reader:
            split_name = row["split"]
            if split_name not in split_to_indices:
                continue
            try:
                idx = int(row["source_index"])
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid source_index in manifest row: {row}"
                ) from exc
            split_to_indices[split_name].append(idx)

    for split_name in TARGET_SPLITS:
        if not split_to_indices[split_name]:
            raise RuntimeError(f"No rows found in manifest for split: {split_name}")

    return split_to_indices


def write_split_csv(output_csv: Path, fieldnames, rows, indices):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in indices:
            writer.writerow(rows[idx])


def validate_indices(split_to_indices, train_size: int):
    for split_name, indices in split_to_indices.items():
        for idx in indices:
            if idx < 0 or idx >= train_size:
                raise RuntimeError(
                    f"Out-of-range source_index for {split_name}: {idx} (train size: {train_size})"
                )


def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    training_dir = (
        repo_root
        / "Data"
        / "Integrated Dataset"
        / "Dataset with Prompt"
        / "Training Set"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Split D_train for methods-only.csv into probe-train and conformal-calib CSVs "
            "according to step1 split manifest."
        )
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=training_dir / "D_train for methods-only.csv",
        help="Path to D_train for methods-only.csv",
    )
    parser.add_argument(
        "--manifest_csv",
        type=Path,
        default=script_dir / "outputs" / "step1_full" / "llama3_8b_split_manifest.csv",
        help="Path to split manifest CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=training_dir,
        help="Directory to write split CSV files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {args.manifest_csv}")

    fieldnames, train_rows = read_train_rows(args.train_csv)
    split_to_indices = read_manifest_indices(args.manifest_csv)
    validate_indices(split_to_indices, train_size=len(train_rows))

    output_probe = args.output_dir / "D_train for probe-train.csv"
    output_calib = args.output_dir / "D_train for conformal-calib.csv"

    write_split_csv(output_probe, fieldnames, train_rows, split_to_indices["probe-train"])
    write_split_csv(
        output_calib, fieldnames, train_rows, split_to_indices["conformal-calib"]
    )

    print(f"train_rows={len(train_rows)}")
    print(f"probe-train_rows={len(split_to_indices['probe-train'])}")
    print(f"conformal-calib_rows={len(split_to_indices['conformal-calib'])}")
    print(f"written={output_probe}")
    print(f"written={output_calib}")


if __name__ == "__main__":
    main()