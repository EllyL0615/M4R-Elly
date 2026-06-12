#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 0 (upstream): generate the SCRC "methods-only" prompts.

This is the self-contained version of the original StatQA prompt-generation code
(`SCRC/0_prompt_multilabel_zero_shot.py` for the training set, and
`Construction/prompt_organization.py --trick_name methods-only` for the test set),
collapsed into one script with NO dependency on the StatQA root modules
(`prompt_organization`, `prompt_wording`, `utils`, `path`). The SCRC-only prompt
constants and helpers are inlined below.

Two output prompt files are produced (both written to NMSCRC/data/prompts/):
  - train : `D_train for methods-only.csv`  (reuses column-info blocks already present
            in `D_train for zero-shot.csv` — Priority 1 — falling back to metadata files)
  - test  : `mini-StatQA for methods-only.csv` (built fresh from column metadata)

Inputs live in the StatQA repo (the sibling of this NMSCRC folder). Their root is
resolved from $STATQA_ROOT, defaulting to the sibling StatQA directory, so no absolute
path is hard-coded.

Downstream: these prompts feed `step1_hidden_states.py`, whose hidden-state outputs are
then pooled by `notebooks/pool.ipynb` into the `{model}_pool_*` inputs the nmscrc package
consumes.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Path roots (env-overridable, no absolute literals — mirrors nmscrc/paths.py) #
# --------------------------------------------------------------------------- #
NMSCRC_ROOT = Path(__file__).resolve().parents[1]                 # .../StatQA/NMSCRC
STATQA_ROOT = Path(os.environ.get("STATQA_ROOT", NMSCRC_ROOT.parent))  # .../StatQA
DEFAULT_OUT_DIR = NMSCRC_ROOT / "data" / "prompts"

# Input files inside the StatQA repo (relative to STATQA_ROOT).
TRAIN_ZERO_SHOT_CSV = STATQA_ROOT / "Data/Integrated Dataset/Dataset with Prompt/Training Set/D_train for zero-shot.csv"
TEST_SOURCE_CSV     = STATQA_ROOT / "Data/Integrated Dataset/Balanced Benchmark/mini-StatQA.csv"
METADATA_DIR        = STATQA_ROOT / "Data/Metadata/Column Metadata"


# --------------------------------------------------------------------------- #
# SCRC methods-only prompt constants (verbatim from prompt_wording.py)         #
# --------------------------------------------------------------------------- #
PROMPT_TASK_DESCRIPTION_SCRC = (
    "You need to select all applicable methods from provided Classification List "
    "based on the given Statistical Question and the properties of the Relevant Columns."
)
PROMPT_INSTRUCTION_SCRC = (
    "You should only reply with one answer in JSON format containing one key: 'methods'. "
    "The value of 'methods' is a list containing all methods you think applicable. "
    'The template is: {"methods": ["<...>", "<...>", "<...>"]}. '
    "Ensure your methods selection is only limited to the classification list provided."
)
PROMPT_RESPONSE_SCRC_PREFIX = 'The answer of applicable methods in JSON format is: {"methods": ["'

# Shared 27-method classification list (verbatim from prompt_wording.PROMPT_CLASSIFICATION).
PROMPT_CLASSIFICATION = (
    "Correlation Analysis: Pearson Correlation Coefficient, Spearman Correlation Coefficient, "
    "Kendall Correlation Coefficient, Partial Correlation Coefficient;\n"
    "Distribution Compliance Test: Anderson-Darling Test, Shapiro-Wilk Test of Normality, "
    "Kolmogorov-Smirnov Test for Normality, Lilliefors Test, Kolmogorov-Smirnov Test, "
    "Kolmogorov-Smirnov Test for Uniform distribution, Kolmogorov-Smirnov Test for Gamma distribution, "
    "Kolmogorov-Smirnov Test for Exponential distribution;\n"
    "Contingency Table Test: Chi-square Independence Test, Fisher Exact Test, Mantel-Haenszel Test;\n"
    "Descriptive Statistics: Mean, Median, Mode, Range, Quartile, Standard Deviation, Skewness, Kurtosis;\n"
    "Variance Test: Mood Variance Test, Levene Test, Bartlett Test, F-Test for Variance."
)

# Output column order (identical for train and test paths).
OUTPUT_COLUMNS = ["dataset", "refined_question", "relevant_column", "task", "difficulty", "results", "prompt"]


# --------------------------------------------------------------------------- #
# Shared helpers (inlined from utils.py / prompt_organization.py)              #
# --------------------------------------------------------------------------- #
def get_metadata(dataset_name: str) -> pd.DataFrame:
    """Column metadata for one dataset (from utils.get_metadata)."""
    metadata_file_path = METADATA_DIR / f"{dataset_name}_col_meta.csv"
    return pd.read_csv(metadata_file_path)


def format_column_metadata_row(meta_row, metadata_headers) -> str:
    """Format one metadata row into prompt text (from prompt_organization.format_column_metadata_row)."""
    col_meta_str = ""
    for header in metadata_headers:
        # Value of key "dataset" and "column_description" will be not provided
        if header != "dataset" and header != "column_description":
            value = meta_row[header]
            # Normalize only data_type values; do not rewrite the whole line,
            # otherwise column_header strings containing "cate" can be corrupted.
            if header == "data_type":
                value_str = str(value)
                if value_str == "cate":
                    value = "categorical"
                elif value_str == "quant":
                    value = "quantitative"
            col_meta_str += f"{header}: {value}; "
    # replace semicolon at the end of the row with a period mark
    col_meta_str = col_meta_str[:-2] + "."
    return col_meta_str


def parse_relevant_column_headers(relevant_column_cell: str) -> list:
    """Extract column_header strings from a `relevant_column` JSON cell."""
    try:
        items = json.loads(relevant_column_cell)
    except Exception:
        items = json.loads(str(relevant_column_cell).replace("'", '"'))

    headers = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                header = str(item.get("column_header", "")).strip()
                if header:
                    headers.append(header)
    return headers


# --------------------------------------------------------------------------- #
# TEST path: build methods-only prompts fresh from metadata                    #
# (from Construction/prompt_organization.py, methods-only branch)             #
# --------------------------------------------------------------------------- #
def extract_scrc_relevant_columns(df: pd.DataFrame, row_index: int) -> list:
    """Relevant columns for SCRC: ground_truth.columns (priority 1) else relevant_column[*]."""
    if row_index >= len(df):
        return []

    # Priority 1: ground_truth columns
    if "ground_truth" in df.columns:
        ground_truth_content = df.at[row_index, "ground_truth"]
        if pd.notna(ground_truth_content):
            try:
                ground_truth_json = json.loads(ground_truth_content)
            except Exception:
                try:
                    ground_truth_json = json.loads(ground_truth_content.replace("'", '"'))
                except Exception:
                    ground_truth_json = None

            if isinstance(ground_truth_json, dict):
                columns = ground_truth_json.get("columns", [])
                if isinstance(columns, list):
                    cleaned_columns = [str(col).strip() for col in columns if str(col).strip()]
                    if cleaned_columns:
                        return cleaned_columns

    # Priority 2: relevant_column field
    relevant_column_content = df.at[row_index, "relevant_column"]
    try:
        relevant_column_json = json.loads(relevant_column_content)
    except Exception:
        try:
            relevant_column_json = json.loads(relevant_column_content.replace("'", '"'))
        except Exception:
            return []

    extracted_columns = []
    if isinstance(relevant_column_json, list):
        for item in relevant_column_json:
            if isinstance(item, dict):
                column_header = str(item.get("column_header", "")).strip()
                if column_header:
                    extracted_columns.append(column_header)
    return extracted_columns


def build_methods_only_prompt(refined_question: str, relevant_meta_info_list: list) -> str:
    """Assemble the SCRC methods-only prompt body (shared by train/test paths)."""
    return (
        "### Task Description: " + PROMPT_TASK_DESCRIPTION_SCRC
        + "\n### Instruction: " + PROMPT_INSTRUCTION_SCRC
        + "\n### Classification List: \n" + PROMPT_CLASSIFICATION
        + "\n### Relevant Columns:\n" + "\n".join(relevant_meta_info_list)
        + "\n### Statistical Question: " + str(refined_question)
        + "\n### Response: " + PROMPT_RESPONSE_SCRC_PREFIX
    )


def generate_test_prompts(source_csv: Path, output_csv: Path) -> None:
    """Test set: build methods-only prompts from the Balanced Benchmark source + metadata."""
    df = pd.read_csv(source_csv)
    df["prompt"] = ""

    meta_cache = {}
    for row_index, row in df.iterrows():
        try:
            dataset_name = str(df["dataset"].iloc[row_index])
            if dataset_name not in meta_cache:
                meta_cache[dataset_name] = get_metadata(dataset_name)
            curr_dataset_meta_df = meta_cache[dataset_name]

            refined_question = df.at[row_index, "refined_question"]
            relevant_column_headers = extract_scrc_relevant_columns(df, row_index)

            relevant_meta_info_list = []
            for column_header in relevant_column_headers:
                matched_rows = curr_dataset_meta_df[
                    curr_dataset_meta_df["column_header"].astype(str).str.strip()
                    == str(column_header).strip()
                ]
                if not matched_rows.empty:
                    relevant_meta_info_list.append(
                        format_column_metadata_row(
                            meta_row=matched_rows.iloc[0],
                            metadata_headers=curr_dataset_meta_df.columns,
                        )
                    )
                else:
                    relevant_meta_info_list.append(
                        f"column_header: {column_header}; data_type: unknown; "
                        f"num_of_rows: N/A; is_normality: N/A."
                    )

            if not relevant_meta_info_list:
                raise ValueError(f"No relevant columns found for row {row_index}.")

            df.at[row_index, "prompt"] = build_methods_only_prompt(refined_question, relevant_meta_info_list)
        except Exception as exc:
            print(f"[!] Error organizing test prompt for row {row_index}: {exc}")
            df.at[row_index, "prompt"] = "Error!"

    df = df[OUTPUT_COLUMNS]
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[+] Test prompts saved: {output_csv}  ({len(df)} rows)")


# --------------------------------------------------------------------------- #
# TRAIN path: reuse column-info blocks already in the zero-shot prompts        #
# (from SCRC/0_prompt_multilabel_zero_shot.py)                                 #
# --------------------------------------------------------------------------- #
def extract_column_info_map_from_prompt(prompt_text: str) -> dict:
    """Pull the `### Column Information:` block out of an existing zero-shot prompt."""
    text = "" if pd.isna(prompt_text) else str(prompt_text)
    start_tag = "### Column Information:"
    end_tag = "### Statistical Question:"

    start_idx = text.find(start_tag)
    if start_idx == -1:
        return {}
    end_idx = text.find(end_tag, start_idx + len(start_tag))
    if end_idx == -1:
        return {}

    block = text[start_idx + len(start_tag):end_idx]
    info_map = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("column_header:"):
            continue
        matched = re.match(r"column_header:\s*(.*?);\s*data_type:", line)
        if matched:
            header = matched.group(1).strip()
        else:
            header = line.split(";", 1)[0].replace("column_header:", "").strip()
        if header:
            info_map[header] = line if line.endswith(".") else line + "."
    return info_map


def build_metadata_line_from_file(dataset_name: str, column_header: str):
    """Fallback: build one column-info line from the metadata file."""
    try:
        metadata_df = get_metadata(dataset_name=dataset_name)
    except Exception:
        return None
    if "column_header" not in metadata_df.columns:
        return None

    matched = metadata_df[
        metadata_df["column_header"].astype(str).str.strip() == str(column_header).strip()
    ]
    if matched.empty:
        return None

    row = matched.iloc[0]
    data_type_value = str(row.get("data_type", "unknown"))
    if data_type_value == "cate":
        data_type_value = "categorical"
    elif data_type_value == "quant":
        data_type_value = "quantitative"

    num_of_rows = row.get("num_of_rows", "N/A")
    is_normality = row.get("is_normality", "N/A")
    return (
        f"column_header: {column_header}; "
        f"data_type: {data_type_value}; "
        f"num_of_rows: {num_of_rows}; "
        f"is_normality: {is_normality}."
    )


def generate_train_prompts(input_csv: Path, output_csv: Path, report_csv: Path) -> None:
    """Training set: rebuild methods-only prompts by reusing zero-shot column-info blocks."""
    df = pd.read_csv(input_csv)

    output_prompts = []
    report_rows = []

    for row_idx, row in df.iterrows():
        dataset_name = str(row.get("dataset", ""))
        relevant_headers = parse_relevant_column_headers(row["relevant_column"])
        prompt_info_map = extract_column_info_map_from_prompt(row.get("prompt", ""))

        relevant_lines = []
        for header in relevant_headers:
            if header in prompt_info_map:
                relevant_lines.append(prompt_info_map[header])
                continue

            metadata_line = build_metadata_line_from_file(dataset_name, header)
            if metadata_line is not None:
                relevant_lines.append(metadata_line)
                report_rows.append({
                    "row_index": row_idx, "dataset": dataset_name, "column_header": header,
                    "fallback_type": "metadata_file",
                    "reason": "missing_or_mismatched_in_zero_shot_prompt_block",
                })
            else:
                relevant_lines.append(
                    f"column_header: {header}; data_type: unknown; num_of_rows: N/A; is_normality: N/A."
                )
                report_rows.append({
                    "row_index": row_idx, "dataset": dataset_name, "column_header": header,
                    "fallback_type": "unknown", "reason": "not_found_in_prompt_and_metadata",
                })

        output_prompts.append(build_methods_only_prompt(row["refined_question"], relevant_lines))

    out_df = df.copy()
    out_df["prompt"] = output_prompts
    out_df = out_df[OUTPUT_COLUMNS]
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_csv, index=False, encoding="utf-8")

    metadata_count = 0 if report_df.empty else int((report_df["fallback_type"] == "metadata_file").sum())
    unknown_count = 0 if report_df.empty else int((report_df["fallback_type"] == "unknown").sum())
    print(f"[+] Train prompts saved: {output_csv}  ({len(out_df)} rows)")
    print(f"[i] Report: {report_csv}  (metadata fallbacks: {metadata_count}, unknown: {unknown_count})")


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SCRC methods-only prompts (train and/or test) into NMSCRC/data/prompts."
    )
    parser.add_argument(
        "--set", dest="which_set", choices=["train", "test", "both"], default="both",
        help="Which prompt set to generate (default: both).",
    )
    parser.add_argument(
        "--out_dir", type=str, default=str(DEFAULT_OUT_DIR),
        help="Output directory for the prompt CSVs (default: NMSCRC/data/prompts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[i] STATQA_ROOT : {STATQA_ROOT}")
    print(f"[i] out_dir     : {out_dir}")

    if args.which_set in ("train", "both"):
        generate_train_prompts(
            input_csv=TRAIN_ZERO_SHOT_CSV,
            output_csv=out_dir / "D_train for methods-only.csv",
            report_csv=out_dir / "D_train for methods-only.report.csv",
        )
    if args.which_set in ("test", "both"):
        generate_test_prompts(
            source_csv=TEST_SOURCE_CSV,
            output_csv=out_dir / "mini-StatQA for methods-only.csv",
        )

    print("[+] Step 0 prompt generation finished.")


if __name__ == "__main__":
    main()
