#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate methods-only linear-probe prompts for training set by reusing
per-row Column Information blocks from `D_train for zero-shot.csv`.

This script is designed for reproducibility of the one-off conversion.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd


main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

from prompt_wording import (  # noqa: E402
    PROMPT_CLASSIFICATION,
    PROMPT_INSTRUCTION_SCRC,
    PROMPT_RESPONSE_SCRC_PREFIX,
    PROMPT_TASK_DESCRIPTION_SCRC,
)
import utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate D_train for linear-probe from D_train for zero-shot.'
    )
    parser.add_argument(
        '--input_csv',
        default='Data/Integrated Dataset/Dataset with Prompt/Training Set/D_train for zero-shot.csv',
        type=str,
        help='Input CSV path (zero-shot training prompt file).',
    )
    parser.add_argument(
        '--output_csv',
        default='Data/Integrated Dataset/Dataset with Prompt/Training Set/D_train for linear-probe.csv',
        type=str,
        help='Output CSV path (linear-probe training prompt file).',
    )
    parser.add_argument(
        '--report_csv',
        default='SCRC/D_train for linear-probe.report.csv',
        type=str,
        help='Diagnostic report path for fallback cases.',
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(main_folder_path, path_value)


def parse_relevant_column_headers(relevant_column_cell: str) -> list:
    try:
        items = json.loads(relevant_column_cell)
    except Exception:
        items = json.loads(str(relevant_column_cell).replace("'", '"'))

    headers = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                header = str(item.get('column_header', '')).strip()
                if header:
                    headers.append(header)
    return headers


def extract_column_info_map_from_prompt(prompt_text: str) -> dict:
    text = '' if pd.isna(prompt_text) else str(prompt_text)
    start_tag = '### Column Information:'
    end_tag = '### Statistical Question:'

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
        if not line.startswith('column_header:'):
            continue

        matched = re.match(r'column_header:\s*(.*?);\s*data_type:', line)
        if matched:
            header = matched.group(1).strip()
        else:
            header = line.split(';', 1)[0].replace('column_header:', '').strip()

        if header:
            info_map[header] = line if line.endswith('.') else line + '.'

    return info_map


def build_metadata_line_from_file(dataset_name: str, column_header: str) -> str | None:
    try:
        metadata_df = utils.get_metadata(dataset_name=dataset_name)
    except Exception:
        return None

    if 'column_header' not in metadata_df.columns:
        return None

    matched = metadata_df[
        metadata_df['column_header'].astype(str).str.strip() == str(column_header).strip()
    ]
    if matched.empty:
        return None

    row = matched.iloc[0]
    data_type_value = str(row.get('data_type', 'unknown'))
    if data_type_value == 'cate':
        data_type_value = 'categorical'
    elif data_type_value == 'quant':
        data_type_value = 'quantitative'

    num_of_rows = row.get('num_of_rows', 'N/A')
    is_normality = row.get('is_normality', 'N/A')
    return (
        f'column_header: {column_header}; '
        f'data_type: {data_type_value}; '
        f'num_of_rows: {num_of_rows}; '
        f'is_normality: {is_normality}.'
    )


def build_linear_probe_prompt(refined_question: str, relevant_lines: list) -> str:
    return (
        '### Task Description: ' + PROMPT_TASK_DESCRIPTION_SCRC
        + '\n### Instruction: ' + PROMPT_INSTRUCTION_SCRC
        + '\n### Classification List: \n' + PROMPT_CLASSIFICATION
        + '\n### Relevant Columns:\n' + '\n'.join(relevant_lines)
        + '\n### Statistical Question: ' + str(refined_question)
        + '\n### Response: ' + PROMPT_RESPONSE_SCRC_PREFIX
    )


def main() -> None:
    args = parse_args()
    input_csv = resolve_repo_path(args.input_csv)
    output_csv = resolve_repo_path(args.output_csv)
    report_csv = resolve_repo_path(args.report_csv)

    # Keep metadata and dataset path behavior stable regardless of launch directory.
    os.chdir(main_folder_path)
    df = pd.read_csv(input_csv)

    output_prompts = []
    report_rows = []

    for row_idx, row in df.iterrows():
        dataset_name = str(row.get('dataset', ''))
        relevant_headers = parse_relevant_column_headers(row['relevant_column'])
        prompt_info_map = extract_column_info_map_from_prompt(row.get('prompt', ''))

        relevant_lines = []
        for header in relevant_headers:
            if header in prompt_info_map:
                relevant_lines.append(prompt_info_map[header])
                continue

            metadata_line = build_metadata_line_from_file(dataset_name, header)
            if metadata_line is not None:
                relevant_lines.append(metadata_line)
                report_rows.append({
                    'row_index': row_idx,
                    'dataset': dataset_name,
                    'column_header': header,
                    'fallback_type': 'metadata_file',
                    'reason': 'missing_or_mismatched_in_zero_shot_prompt_block',
                })
            else:
                unknown_line = (
                    f'column_header: {header}; '
                    f'data_type: unknown; '
                    f'num_of_rows: N/A; '
                    f'is_normality: N/A.'
                )
                relevant_lines.append(unknown_line)
                report_rows.append({
                    'row_index': row_idx,
                    'dataset': dataset_name,
                    'column_header': header,
                    'fallback_type': 'unknown',
                    'reason': 'not_found_in_prompt_and_metadata',
                })

        output_prompts.append(
            build_linear_probe_prompt(
                refined_question=row['refined_question'],
                relevant_lines=relevant_lines,
            )
        )

    out_df = df.copy()
    out_df['prompt'] = output_prompts
    out_df = out_df[['dataset', 'refined_question', 'relevant_column', 'task', 'difficulty', 'results', 'prompt']]

    output_dir = os.path.dirname(output_csv)
    report_dir = os.path.dirname(report_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    out_df.to_csv(output_csv, index=False, encoding='utf-8')
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_csv, index=False, encoding='utf-8')

    unknown_count = 0 if report_df.empty else int((report_df['fallback_type'] == 'unknown').sum())
    metadata_count = 0 if report_df.empty else int((report_df['fallback_type'] == 'metadata_file').sum())

    print(f'[+] Generated: {output_csv}')
    print(f'[+] Report: {report_csv}')
    print(f'[i] Total rows: {len(out_df)}')
    print(f'[i] Metadata fallback rows: {metadata_count}')
    print(f'[i] Unknown fallback rows: {unknown_count}')


if __name__ == '__main__':
    main()
