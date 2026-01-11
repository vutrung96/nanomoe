#!/usr/bin/env python3
"""
Preprocess HuggingFace datasets for training.

Downloads a HuggingFace dataset, tokenizes it in parallel, and saves to binary format
suitable for efficient training data loading.

Usage:
    python data/preprocess_data.py \
        --dataset HuggingFaceFW/fineweb-edu \
        --config sample-10BT \
        --output-prefix data/fineweb-edu \
        --tokenizer gpt2 \
        --num-workers 8
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Optional
import multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess HuggingFace datasets for training")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g., HuggingFaceFW/fineweb-edu)")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset configuration/subset name (e.g., sample-10BT)")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process (default: train)")
    parser.add_argument("--text-key", type=str, default="text",
                        help="Key for text field in dataset (default: text)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large datasets")

    # Tokenizer arguments
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Tokenizer name or path (default: gpt2)")
    parser.add_argument("--append-eod", action="store_true", default=True,
                        help="Append end-of-document token after each document")
    parser.add_argument("--no-append-eod", action="store_false", dest="append_eod",
                        help="Do not append end-of-document token")

    # Output arguments
    parser.add_argument("--output-prefix", type=str, required=True,
                        help="Output file prefix (will create .bin and .idx files)")

    # Processing arguments
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of workers for parallel processing (default: CPU count)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for tokenization (default: 1000)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--log-interval", type=int, default=10000,
                        help="Log progress every N samples")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to cache downloaded datasets")

    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    return args


def write_index_file(idx_path: str, dtype: np.dtype, doc_indices: np.ndarray):
    """Write index file with document offsets."""
    with open(idx_path, 'wb') as f:
        # Magic number and version
        f.write(b'MMIDX\x00\x01')
        # Dtype code (uint16=1, uint32=2)
        dtype_code = 1 if dtype == np.uint16 else 2
        f.write(np.array([dtype_code], dtype=np.int8).tobytes())
        # Number of documents
        f.write(np.array([len(doc_indices) - 1], dtype=np.int64).tobytes())
        # Document indices (offsets)
        f.write(doc_indices.astype(np.int64).tobytes())


def main():
    args = get_args()

    # Import here to avoid slow startup when just checking --help
    print("Loading dependencies...")
    from datasets import load_dataset, DownloadConfig
    from transformers import AutoTokenizer
    from tqdm import tqdm

    # Create output directory
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Determine optimal dtype based on vocab size
    vocab_size = tokenizer.vocab_size
    if vocab_size <= 65535:
        dtype = np.uint16
    else:
        dtype = np.uint32
    print(f"Vocab size: {vocab_size}, using dtype: {dtype}")

    # Get EOD token
    eod_token_id = tokenizer.eos_token_id
    if eod_token_id is None:
        eod_token_id = tokenizer.pad_token_id
    if eod_token_id is None:
        print("Warning: No EOS or PAD token found, using vocab_size - 1")
        eod_token_id = vocab_size - 1
    print(f"EOD token ID: {eod_token_id}")

    # Load dataset
    print(f"Loading dataset: {args.dataset}" + (f" ({args.config})" if args.config else ""))

    download_config = DownloadConfig(
        num_proc=args.num_workers,
    )

    load_kwargs = {
        "path": args.dataset,
        "split": args.split,
    }

    if args.config:
        load_kwargs["name"] = args.config

    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    if args.streaming:
        load_kwargs["streaming"] = True
        print("Using streaming mode")
    else:
        load_kwargs["num_proc"] = args.num_workers
        print(f"Using {args.num_workers} workers for download/processing")

    dataset = load_dataset(**load_kwargs)

    # Limit samples if requested
    if args.max_samples and not args.streaming:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Setup output files
    bin_path = f"{args.output_prefix}.bin"
    idx_path = f"{args.output_prefix}.idx"

    print(f"Output files: {bin_path}, {idx_path}")

    # Process and tokenize
    print("Tokenizing and writing data...")

    total_tokens = 0
    doc_indices = [0]  # Start offset

    # Open binary file for writing
    bin_file = open(bin_path, 'wb')

    start_time = time.time()
    sample_count = 0

    def tokenize_batch(batch):
        """Tokenize a batch of texts."""
        texts = batch[args.text_key]
        results = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {"input_ids": results["input_ids"]}

    if args.streaming:
        # Streaming mode - process one at a time
        iterator = iter(dataset)
        if args.max_samples:
            from itertools import islice
            iterator = islice(iterator, args.max_samples)

        for sample in tqdm(iterator, desc="Processing", total=args.max_samples):
            text = sample[args.text_key]
            tokens = tokenizer.encode(text, add_special_tokens=False)

            if args.append_eod:
                tokens.append(eod_token_id)

            if tokens:
                token_array = np.array(tokens, dtype=dtype)
                bin_file.write(token_array.tobytes())
                total_tokens += len(tokens)
                doc_indices.append(total_tokens)

            sample_count += 1

            if sample_count % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Processed {sample_count} samples, {total_tokens:,} tokens "
                      f"({sample_count/elapsed:.1f} samples/s)")
    else:
        # Non-streaming mode - use batched processing
        dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_workers,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
        )

        for sample in tqdm(dataset, desc="Writing"):
            tokens = sample["input_ids"]

            if args.append_eod:
                tokens = tokens + [eod_token_id]

            if tokens:
                token_array = np.array(tokens, dtype=dtype)
                bin_file.write(token_array.tobytes())
                total_tokens += len(tokens)
                doc_indices.append(total_tokens)

            sample_count += 1

            if sample_count % args.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Processed {sample_count} samples, {total_tokens:,} tokens "
                      f"({sample_count/elapsed:.1f} samples/s)")

    bin_file.close()

    # Write index file
    print("Writing index file...")
    doc_indices = np.array(doc_indices, dtype=np.int64)
    write_index_file(idx_path, dtype, doc_indices)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"  Total samples: {sample_count:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens/sample: {total_tokens / max(sample_count, 1):.1f}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"  Throughput: {sample_count / elapsed:.1f} samples/s")
    print(f"  Binary file: {bin_path} ({os.path.getsize(bin_path) / 1e9:.2f} GB)")
    print(f"  Index file: {idx_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
