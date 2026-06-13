#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute XLRS accuracy broken down by annotated dataset categories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize XLRS predictions by dataset category."
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        required=True,
        help="JSONL file produced by eval_xlrs_lite_to_json.py.",
    )
    parser.add_argument(
        "--category-json",
        type=Path,
        required=True,
        help="JSON file that contains category annotations (e.g., xlrs_category_sample.json).",
    )
    parser.add_argument(
        "--category-field",
        default="category",
        help="Field name in the category JSON to report (default: category).",
    )
    parser.add_argument(
        "--secondary-field",
        default=None,
        help="Optional secondary field (e.g., l2_category) to also tabulate.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=0,
        help="Only print categories with at least this many samples.",
    )
    return parser.parse_args()


def _extract_human_question(entry: Dict[str, Any]) -> str:
    convs = entry.get("conversations") or []
    for msg in convs:
        if msg.get("from") == "human":
            return str(msg.get("value", "") or "").strip()
    return ""


def load_category_pools(
    path: Path,
    field_names: List[str],
) -> Tuple[Dict[int, deque[Dict[str, str]]], Dict[str, deque[Dict[str, str]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")
    by_id: Dict[int, deque[Dict[str, str]]] = defaultdict(deque)
    by_question: Dict[str, deque[Dict[str, str]]] = defaultdict(deque)
    for idx, item in enumerate(data):
        entry = {field: str(item.get(field, "") or "").strip() for field in field_names}
        question = _extract_human_question(item)
        sid = idx
        if "id" in item:
            try:
                sid = int(item.get("id", idx))
            except (TypeError, ValueError):
                sid = idx
        by_id[sid].append(entry)
        if question:
            by_question[question].append(entry)
    return by_id, by_question


def normalize_choice(value: Any) -> Any:
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return ""
        upper = trimmed.upper()
        if len(upper) == 1 and upper in {"A", "B", "C", "D"}:
            return upper
        return upper
    return value


def _extract_single_choice_letter(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().upper()
    letters = []
    for ch in cleaned:
        if ch in {"A", "B", "C", "D"} and ch not in letters:
            letters.append(ch)
    if len(letters) == 1:
        return letters[0]
    return None


def determine_correctness(result: Dict[str, Any]) -> bool:
    if "correct" in result:
        val = result["correct"]
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            normalized = val.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
    gt = normalize_choice(result.get("ground_truth"))
    pred = normalize_choice(result.get("model_output"))
    if gt == "" or pred == "":
        return gt == pred
    return gt == pred


def _pop_category(
    question: str,
    by_question: Dict[str, deque[Dict[str, str]]],
) -> Dict[str, str] | None:
    """Only use the question text to locate the category; IDs are ignored by design."""
    if not question:
        return None
    qdeque = by_question.get(question)
    if qdeque and len(qdeque):
        return qdeque.popleft()
    return None


def tally(
    results_path: Path,
    by_question: Dict[str, deque[Dict[str, str]]],
    primary_field: str,
    secondary_field: str | None,
) -> Tuple[Dict[str, Dict[str, int]], int, Dict[str, int]]:
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    secondary_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    missing = 0
    single_only = {"correct": 0, "total": 0}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            question = str(data.get("question", "") or "").strip()
            category_meta = _pop_category(question, by_question)
            category_value = (
                category_meta.get(primary_field) if category_meta else ""
            ) or "unknown"
            secondary_value = ""
            if secondary_field:
                secondary_value = (
                    category_meta.get(secondary_field) if category_meta else ""
                ) or "unknown"
            if category_meta is None:
                missing += 1
            correct = determine_correctness(data)
            stats[category_value]["total"] += 1
            if correct:
                stats[category_value]["correct"] += 1
            gt_single = _extract_single_choice_letter(data.get("ground_truth"))
            if gt_single:
                single_only["total"] += 1
                if correct:
                    single_only["correct"] += 1
            if secondary_field:
                secondary_stats[secondary_value]["total"] += 1
                if correct:
                    secondary_stats[secondary_value]["correct"] += 1
    summary = {"primary": dict(stats)}
    if secondary_field:
        summary["secondary"] = dict(secondary_stats)
    return summary, missing, single_only


def format_stats(
    stats: Dict[str, Dict[str, int]],
    min_samples: int,
) -> List[str]:
    lines = []
    for category, counts in sorted(stats.items(), key=lambda item: (-item[1]["total"], item[0])):
        total = counts["total"]
        if total < min_samples:
            continue
        correct = counts["correct"]
        acc = correct / total if total else 0.0
        lines.append(f"{category}: {correct}/{total} ({acc:.2%})")
    return lines


def compute_overall_accuracy(stats: Dict[str, Dict[str, int]]) -> Tuple[int, int, float]:
    total = sum(entry.get("total", 0) for entry in stats.values())
    correct = sum(entry.get("correct", 0) for entry in stats.values())
    acc = correct / total if total else 0.0
    return correct, total, acc


def main() -> None:
    args = parse_args()
    _by_id, by_question = load_category_pools(
        args.category_json,
        [args.category_field] + ([args.secondary_field] if args.secondary_field else []),
    )
    total_samples = sum(len(q) for q in by_question.values())
    summary, missing, single_only = tally(
        args.results_jsonl,
        by_question,
        args.category_field,
        args.secondary_field,
    )
    print(f"Loaded {total_samples} annotated samples, missing categories for {missing} predictions.")
    print("Primary category accuracy:")
    for line in format_stats(summary["primary"], args.min_samples):
        print("  " + line)
    overall_correct, overall_total, overall_acc = compute_overall_accuracy(summary["primary"])
    print(f"Overall accuracy: {overall_correct}/{overall_total} ({overall_acc:.2%})")
    single_correct = single_only.get("correct", 0)
    single_total = single_only.get("total", 0)
    single_acc = single_correct / single_total if single_total else 0.0
    print(f"Overall accuracy (single-choice only): {single_correct}/{single_total} ({single_acc:.2%})")
    if args.secondary_field:
        print(f"\nSecondary ({args.secondary_field}) category accuracy:")
        for line in format_stats(summary["secondary"], args.min_samples):
            print("  " + line)


if __name__ == "__main__":
    main()
