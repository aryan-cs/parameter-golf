#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one match for snippet, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch PR755 retokenize_corpus.py to support resumable shard generation."
    )
    parser.add_argument("retokenize_corpus", type=Path)
    args = parser.parse_args()

    path = args.retokenize_corpus
    text = path.read_text()

    if "def shard_is_complete(path: Path) -> bool:" not in text:
        text = replace_once(
            text,
            "def decode_shard_to_text(shard_path: Path, sp: spm.SentencePieceProcessor) -> str:\n",
            "def shard_is_complete(path: Path) -> bool:\n"
            "    if not path.is_file():\n"
            "        return False\n"
            "    try:\n"
            "        header = np.fromfile(path, dtype=\"<i4\", count=256)\n"
            "        if header.size != 256 or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:\n"
            "            return False\n"
            "        num_tokens = int(header[2])\n"
            "        header_bytes = HEADER_INTS * np.dtype(\"<i4\").itemsize\n"
            "        token_bytes = np.dtype(\"<u2\").itemsize\n"
            "        expected_size = header_bytes + num_tokens * token_bytes\n"
            "        return path.stat().st_size == expected_size\n"
            "    except Exception:\n"
            "        return False\n\n\n"
            "def decode_shard_to_text(shard_path: Path, sp: spm.SentencePieceProcessor) -> str:\n",
        )

    if "skipped = 0" not in text:
        text = replace_once(
            text,
            "    total_base_tokens = 0\n    total_gravity_tokens = 0\n\n    for shard_path in tqdm(train_shards + val_shards, desc=\"Re-tokenizing\"):\n",
            "    total_base_tokens = 0\n    total_gravity_tokens = 0\n    skipped = 0\n\n    for shard_path in tqdm(train_shards + val_shards, desc=\"Re-tokenizing\"):\n",
        )

    if "Skipping existing complete shard" not in text:
        text = replace_once(
            text,
            "    for shard_path in tqdm(train_shards + val_shards, desc=\"Re-tokenizing\"):\n        print(f\"\\n  {shard_path.name}:\")\n\n        # Decode to text\n",
            "    for shard_path in tqdm(train_shards + val_shards, desc=\"Re-tokenizing\"):\n"
            "        print(f\"\\n  {shard_path.name}:\")\n"
            "        output_path = output_dir / shard_path.name\n"
            "        if shard_is_complete(output_path):\n"
            "            print(f\"    Skipping existing complete shard: {output_path.name}\")\n"
            "            skipped += 1\n"
            "            continue\n\n"
            "        # Decode to text\n",
        )

    if "Skipped existing complete shards" not in text:
        text = replace_once(
            text,
            "    print(f\"  (>1.0 means gravity produces longer sequences)\")\n    print(f\"  Output directory: {output_dir}\")\n",
            "    print(f\"  (>1.0 means gravity produces longer sequences)\")\n"
            "    print(f\"  Skipped existing complete shards: {skipped}\")\n"
            "    print(f\"  Output directory: {output_dir}\")\n",
        )

    path.write_text(text)


if __name__ == "__main__":
    main()
