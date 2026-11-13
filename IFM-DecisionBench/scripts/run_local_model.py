#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run an IFM-DecisionBench JSONL file on a local Transformers-compatible
causal language model, including AWQ models.

Key controls:
- only sends the JSONL `prompt` field;
- single-turn conversation;
- no system prompt by default;
- greedy deterministic decoding;
- disables Qwen3 thinking when supported;
- resumes by benchmark_id;
- preserves all source metadata in the output JSONL.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--id-field", default="benchmark_id")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only for a local model repository that requires custom code.",
    )
    parser.add_argument(
        "--disable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Qwen3-style chat templates, request non-thinking responses.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no}: {exc}"
                ) from exc
    return rows


def load_completed_ids(output_path: Path, id_field: str) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Ignore an incomplete final line after interruption.
                continue
            value = row.get(id_field)
            if value is not None:
                completed.add(str(value))
    return completed


def build_chat_text(
    tokenizer: AutoTokenizer,
    prompt: str,
    system_prompt: str,
    disable_thinking: bool,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    # Qwen3 and Qwen3-derived models support enable_thinking.
    # Older/Qwen2.5 templates may reject it, so fall back cleanly.
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            pass

    return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not detected.")

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    model_path = Path(args.model_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local model directory not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_path)
    print(f"Loaded {len(rows)} benchmark requests.")

    for field in (args.id_field, args.prompt_field):
        missing = [
            index
            for index, row in enumerate(rows)
            if row.get(field) in (None, "")
        ]
        if missing:
            raise KeyError(
                f"Field {field!r} is missing or empty in "
                f"{len(missing)} rows; first indexes: {missing[:5]}"
            )

    ids = [str(row[args.id_field]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark IDs detected in input JSONL.")

    completed_ids = load_completed_ids(output_path, args.id_field)
    pending = [
        row for row in rows
        if str(row[args.id_field]) not in completed_ids
    ]

    print(f"Model: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Input rows: {len(rows)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Pending: {len(pending)}")
    print(f"Prompt field: {args.prompt_field}")
    print(f"Disable thinking: {args.disable_thinking}")
    print(f"Output: {output_path}")

    if not pending:
        print("Nothing to do.")
        return

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    # Remove sampling defaults embedded in some model repositories.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    eos_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_ids.extend(tokenizer.eos_token_id)
        else:
            eos_ids.append(int(tokenizer.eos_token_id))

    for token in ("<|eot_id|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            isinstance(token_id, int)
            and token_id >= 0
            and token_id != tokenizer.unk_token_id
            and token_id not in eos_ids
        ):
            eos_ids.append(token_id)

    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if eos_ids:
        generation_kwargs["eos_token_id"] = eos_ids

    all_started = time.time()

    with output_path.open("a", encoding="utf-8") as output_file:
        for start in tqdm(
            range(0, len(pending), args.batch_size),
            desc="Generating",
        ):
            batch = pending[start : start + args.batch_size]

            chat_texts = [
                build_chat_text(
                    tokenizer=tokenizer,
                    prompt=str(row[args.prompt_field]),
                    system_prompt=args.system_prompt,
                    disable_thinking=args.disable_thinking,
                )
                for row in batch
            ]

            encoded = tokenizer(
                chat_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_tokens,
                add_special_tokens=False,
            )
            encoded = {
                key: value.to(model.device)
                for key, value in encoded.items()
            }

            batch_started = time.time()
            with torch.inference_mode():
                outputs = model.generate(**encoded, **generation_kwargs)
            batch_seconds = time.time() - batch_started

            input_width = encoded["input_ids"].shape[1]
            generated_ids = outputs[:, input_width:]
            responses = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )

            input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            output_lengths = [
                int((tokens != tokenizer.pad_token_id).sum().item())
                for tokens in generated_ids
            ]

            for row, response, in_tokens, out_tokens in zip(
                batch,
                responses,
                input_lengths,
                output_lengths,
            ):
                clean_response = response.strip()
                result = dict(row)
                result.update(
                    {
                        "model_name": args.model_name,
                        "model_path": args.model_path,
                        "evaluation_setting": {
                            "conversation_mode": "single_turn",
                            "system_prompt": args.system_prompt or None,
                            "thinking_enabled": False if args.disable_thinking else None,
                        },
                        "generation_config": {
                            "do_sample": False,
                            "num_beams": 1,
                            "max_input_tokens": args.max_input_tokens,
                            "max_new_tokens": args.max_new_tokens,
                            "repetition_penalty": args.repetition_penalty,
                            "dtype": args.dtype,
                            "seed": args.seed,
                            "batch_size": args.batch_size,
                        },
                        "input_tokens": int(in_tokens),
                        "generated_tokens": int(out_tokens),
                        "hit_max_new_tokens": bool(
                            out_tokens >= args.max_new_tokens
                        ),
                        "model_answer": clean_response,
                        "model_response": clean_response,
                        "status": "success" if clean_response else "empty_response",
                        "batch_generation_seconds": round(batch_seconds, 4),
                    }
                )
                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

    elapsed = time.time() - all_started
    print(f"Finished. Total seconds: {elapsed:.2f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
