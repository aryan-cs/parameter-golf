#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a raw text corpus locally for repeatable tokenizer experiments.")
    parser.add_argument("--hf-dataset", default="", help="Optional Hugging Face dataset id.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--hf-split", default="train", help="Dataset split for Hugging Face loading.")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for Hugging Face datasets.")
    parser.add_argument("--input-file", action="append", default=[], help="Optional local input files instead of HF.")
    parser.add_argument("--train-docs", type=int, default=15000, help="Number of training documents to cache.")
    parser.add_argument("--val-docs", type=int, default=1000, help="Number of validation documents to cache.")
    parser.add_argument("--output-dir", default="./data/raw/default", help="Output directory for train/val JSONL files.")
    return parser.parse_args()


def iter_hf_text(args: argparse.Namespace):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("datasets is not installed. Run: uv sync") from exc

    dataset = load_dataset(
        args.hf_dataset,
        name=args.hf_config,
        split=args.hf_split,
        streaming=args.stream,
    )
    for doc in dataset:
        text = doc.get("text")
        if text:
            yield text


def iter_local_text(args: argparse.Namespace):
    if not args.input_file:
        raise FileNotFoundError("Provide --hf-dataset or at least one --input-file")
    for raw_path in args.input_file:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        if len(blocks) > 1:
            for block in blocks:
                yield block
        else:
            for line in text.splitlines():
                line = line.strip()
                if line:
                    yield line


def write_jsonl(path: Path, documents) -> tuple[int, int]:
    doc_count = 0
    total_bytes = 0
    with path.open("w", encoding="utf-8") as handle:
        for text in documents:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            doc_count += 1
            total_bytes += len(text.encode("utf-8"))
    return doc_count, total_bytes


def main() -> int:
    args = parse_args()
    if args.train_docs <= 0:
        raise ValueError("--train-docs must be positive")
    if args.val_docs < 0:
        raise ValueError("--val-docs cannot be negative")

    if args.hf_dataset:
        source = f"hf:{args.hf_dataset}"
        documents = iter_hf_text(args)
    else:
        source = "local-files"
        documents = iter_local_text(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = output_dir / "val.jsonl"
    train_path = output_dir / "train.jsonl"

    total_needed = args.val_docs + args.train_docs
    cached_docs = list(itertools.islice(documents, total_needed))
    val_docs = cached_docs[: args.val_docs]
    train_docs = cached_docs[args.val_docs : args.val_docs + args.train_docs]

    val_count, val_bytes = write_jsonl(val_path, val_docs)
    train_count, train_bytes = write_jsonl(train_path, train_docs)
    manifest = {
        "source": source,
        "output_dir": str(output_dir),
        "train_docs": train_count,
        "val_docs": val_count,
        "train_total_bytes": train_bytes,
        "val_total_bytes": val_bytes,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
