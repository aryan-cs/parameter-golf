#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import zlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ByteLevel BPE tokenizer for Parameter Golf experiments.")
    parser.add_argument("--input-dir", default="", help="Directory containing local text files.")
    parser.add_argument("--glob", default="*.txt", help="Glob used under --input-dir.")
    parser.add_argument("--input-file", action="append", default=[], help="Individual text file(s) to include.")
    parser.add_argument("--hf-dataset", default="", help="Optional Hugging Face dataset id.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--hf-split", default="train", help="Dataset split for Hugging Face loading.")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for Hugging Face datasets.")
    parser.add_argument("--max-docs", type=int, default=50000, help="Maximum documents to feed into tokenizer training.")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Tokenizer vocabulary size.")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token pair frequency.")
    parser.add_argument("--output-dir", default="./data/tokenizers", help="Where vocab and merges files are written.")
    parser.add_argument("--prefix", default="fineweb_32k_bpe", help="Output file prefix.")
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"], help="Special token(s) to reserve.")
    return parser.parse_args()


def iter_local_text(args: argparse.Namespace):
    files = [Path(path) for path in args.input_file]
    if args.input_dir:
        files.extend(sorted(Path(args.input_dir).glob(args.glob)))
    if not files:
        raise FileNotFoundError("No local text files matched. Provide --input-file or --input-dir.")
    docs_seen = 0
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for chunk in filter(None, (part.strip() for part in text.splitlines())):
            yield chunk
            docs_seen += 1
            if docs_seen >= args.max_docs:
                return


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
    for doc in itertools.islice(dataset, args.max_docs):
        text = doc.get("text")
        if text:
            yield text


def main() -> int:
    args = parse_args()
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ModuleNotFoundError:
        print("tokenizers is not installed. Run: uv sync")
        return 1

    if args.hf_dataset:
        iterator = iter_hf_text(args)
        source = f"hf:{args.hf_dataset}"
    else:
        iterator = iter_local_text(args)
        source = "local-files"

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iterator,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_token,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(output_dir), args.prefix)

    vocab_path = output_dir / f"{args.prefix}-vocab.json"
    merges_path = output_dir / f"{args.prefix}-merges.txt"
    combined = vocab_path.read_bytes() + merges_path.read_bytes()
    compressed = zlib.compress(combined, level=9)

    print(f"source={source}")
    print(f"vocab_path={vocab_path}")
    print(f"merges_path={merges_path}")
    print(f"raw_size_bytes={len(combined)}")
    print(f"compressed_size_bytes={len(compressed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
