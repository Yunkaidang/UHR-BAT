#!/usr/bin/env python3
"""Convert RL-MIND/UHR-BAT-SFT-10K metadata to LongVA/LLaVA JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return value is None


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [stripped]
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if _is_missing(value):
        return []
    return [value]


def _normalize_image_path(path: Any, strip_images_prefix: bool) -> str:
    rel = str(path).replace("\\", "/").strip()
    while rel.startswith("./"):
        rel = rel[2:]
    if strip_images_prefix and rel.startswith("images/"):
        rel = rel[len("images/") :]
    return rel


def _normalize_conversations(value: Any, prompt: Any, question: Any, answer: Any) -> List[dict]:
    cleaned = []
    for item in _as_list(value):
        if isinstance(item, dict) and "from" in item and "value" in item:
            cleaned.append({"from": str(item["from"]), "value": str(item["value"])})
    if cleaned:
        return cleaned

    user_text = str(prompt if not _is_missing(prompt) else question).strip()
    if user_text and not user_text.startswith("<image>"):
        user_text = "<image>\n" + user_text
    return [
        {"from": "human", "value": user_text},
        {"from": "gpt", "value": "" if _is_missing(answer) else str(answer).strip()},
    ]


def convert_rows(rows: Iterable[dict], strip_images_prefix: bool) -> List[dict]:
    output = []
    for idx, row in enumerate(rows):
        image_paths = _as_list(row.get("image_paths")) or _as_list(row.get("file_name"))
        image_paths = [_normalize_image_path(path, strip_images_prefix) for path in image_paths if str(path).strip()]
        if not image_paths:
            raise ValueError(f"row {idx} has no image path")

        item = {
            "id": row.get("id", idx),
            "image": image_paths[0] if len(image_paths) == 1 else image_paths,
            "conversations": _normalize_conversations(
                row.get("conversations"),
                row.get("prompt"),
                row.get("question"),
                row.get("answer"),
            ),
        }
        if row.get("source_dataset") is not None and not _is_missing(row.get("source_dataset")):
            item["source_dataset"] = str(row["source_dataset"])
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Path to train/metadata.parquet.")
    parser.add_argument("--output", required=True, help="Output LongVA/LLaVA-style JSON path.")
    parser.add_argument(
        "--keep-images-prefix",
        action="store_true",
        help="Keep a leading images/ prefix. By default paths are relative to train/images.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional row limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(Path(args.metadata).expanduser())
    if args.limit and args.limit > 0:
        df = df.head(args.limit)
    records = convert_rows(df.to_dict(orient="records"), strip_images_prefix=not args.keep_images_prefix)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} samples to {output_path}")


if __name__ == "__main__":
    main()
