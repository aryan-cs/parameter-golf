#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from array import array
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack text documents into local token shards for training.")
    parser.add_argument("--input-dir", default="", help="Directory containing local text files.")
    parser.add_argument("--glob", default="*.txt", help="Glob used under --input-dir.")
    parser.add_argument("--input-file", action="append", default=[], help="Individual text file(s) to include.")
    parser.add_argument("--hf-dataset", default="", help="Optional Hugging Face dataset id.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config.")
    parser.add_argument("--hf-split", default="train", help="Dataset split for Hugging Face loading.")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode for Hugging Face datasets.")
    parser.add_argument("--train-docs", type=int, default=10000, help="Number of documents to write into the train split.")
    parser.add_argument("--val-docs", type=int, default=1000, help="Number of documents to write into the val split.")
    parser.add_argument("--tokenizer-prefix", required=True, help="Prefix for <prefix>-vocab.json and <prefix>-merges.txt.")
    parser.add_argument("--output-dir", default="./data/tokens/default", help="Where train/ and val/ shards are written.")
    parser.add_argument("--token-dtype", default="auto", choices=["auto", "uint16", "uint32"], help="Token dtype for saved shards.")
    parser.add_argument("--shard-size-tokens", type=int, default=2_000_000, help="Flush a shard after this many buffered tokens.")
    return parser.parse_args()


def split_local_documents(text: str):
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if len(blocks) > 1:
        for block in blocks:
            yield block
        return
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield line


def iter_local_text(args: argparse.Namespace):
    files = [Path(path) for path in args.input_file]
    if args.input_dir:
        files.extend(sorted(Path(args.input_dir).glob(args.glob)))
    if not files:
        raise FileNotFoundError("No local text files matched. Provide --input-file or --input-dir.")
    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        yield from split_local_documents(text)


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


class TokenizerAdapter:
    def __init__(self, prefix: str) -> None:
        try:
            from tokenizers import ByteLevelBPETokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("tokenizers is not installed. Run: uv sync") from exc

        vocab_path = f"{prefix}-vocab.json"
        merges_path = f"{prefix}-merges.txt"
        if not Path(vocab_path).exists() or not Path(merges_path).exists():
            raise FileNotFoundError(f"Expected tokenizer files {vocab_path} and {merges_path}")

        self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)
        eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            raise ValueError("Tokenizer is missing the <|endoftext|> special token")
        self.eos_id = eos_id
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        ids.append(self.eos_id)
        return ids


def dtype_and_typecode(token_dtype: str, vocab_size: int):
    if token_dtype == "auto":
        token_dtype = "uint16" if vocab_size <= 65535 else "uint32"
    mapping = {
        "uint16": ("uint16", "H"),
        "uint32": ("uint32", "I"),
    }
    return mapping[token_dtype]


class SplitWriter:
    def __init__(self, split: str, output_dir: Path, dtype_name: str, typecode: str, shard_size_tokens: int, tokenizer_prefix: str, vocab_size: int, source: str) -> None:
        import numpy as np

        self.np = np
        self.split = split
        self.root = output_dir / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype_name = dtype_name
        self.typecode = typecode
        self.shard_size_tokens = shard_size_tokens
        self.tokenizer_prefix = tokenizer_prefix
        self.vocab_size = vocab_size
        self.source = source

        self.buffer = array(typecode)
        self.shard_count = 0
        self.docs = 0
        self.total_bytes = 0
        self.total_tokens = 0

    def add_document(self, token_ids: list[int], byte_count: int) -> None:
        self.buffer.extend(token_ids)
        self.docs += 1
        self.total_bytes += byte_count
        self.total_tokens += len(token_ids)
        if len(self.buffer) >= self.shard_size_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        array_path = self.root / f"{self.split}_{self.shard_count:05d}.npy"
        tokens = self.np.asarray(self.buffer, dtype=self.dtype_name)
        self.np.save(array_path, tokens)
        self.buffer = array(self.typecode)
        self.shard_count += 1

    def finish(self) -> dict:
        self.flush()
        metadata = {
            "split": self.split,
            "source": self.source,
            "tokenizer_prefix": self.tokenizer_prefix,
            "vocab_size": self.vocab_size,
            "token_dtype": self.dtype_name,
            "docs": self.docs,
            "shards": self.shard_count,
            "total_bytes": self.total_bytes,
            "total_tokens": self.total_tokens,
            "avg_bytes_per_token": (self.total_bytes / self.total_tokens) if self.total_tokens else None,
        }
        (self.root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata


def main() -> int:
    args = parse_args()
    if args.train_docs <= 0:
        raise ValueError("--train-docs must be positive")
    if args.val_docs < 0:
        raise ValueError("--val-docs cannot be negative")

    tokenizer = TokenizerAdapter(args.tokenizer_prefix)
    dtype_name, typecode = dtype_and_typecode(args.token_dtype, tokenizer.vocab_size)

    if args.hf_dataset:
        documents = iter_hf_text(args)
        source = f"hf:{args.hf_dataset}"
    else:
        documents = iter_local_text(args)
        source = "local-files"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_writer = SplitWriter("train", output_dir, dtype_name, typecode, args.shard_size_tokens, args.tokenizer_prefix, tokenizer.vocab_size, source)
    val_writer = SplitWriter("val", output_dir, dtype_name, typecode, args.shard_size_tokens, args.tokenizer_prefix, tokenizer.vocab_size, source)

    total_needed = args.val_docs + args.train_docs
    for index, text in enumerate(itertools.islice(documents, total_needed)):
        token_ids = tokenizer.encode(text)
        byte_count = len(text.encode("utf-8"))
        if index < args.val_docs:
            val_writer.add_document(token_ids, byte_count)
        else:
            train_writer.add_document(token_ids, byte_count)

    train_meta = train_writer.finish()
    val_meta = val_writer.finish()

    summary = {
        "source": source,
        "output_dir": str(output_dir),
        "tokenizer_prefix": args.tokenizer_prefix,
        "train": train_meta,
        "val": val_meta,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
