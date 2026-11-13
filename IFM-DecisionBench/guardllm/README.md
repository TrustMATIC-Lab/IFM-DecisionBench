# GuardLLM

GuardLLM is a Qwen3-8B LoRA adapter trained to classify a generated industrial response as `hazardous` or `safe`. Its input contains only the original request and candidate response. Safety boundaries, manual notes, and response labels must not appear in the input.

## Expected data

The training directory must contain `train.jsonl`, `validation.jsonl`, and `test.jsonl`. Each row must contain:

```json
{
  "record_id": "unique identifier",
  "prompt": "industrial request",
  "model_response": "candidate response",
  "label": "hazardous"
}
```

All responses derived from the same matched scenario group must remain in the same partition. The complete GuardLLM training data are not part of the public demonstration release.

## Train

```bash
python guardllm/run_guardllm.py train \
  --model-path /path/to/Qwen3-8B \
  --data-dir /path/to/guard_data \
  --output checkpoints/guardllm
```

## Evaluate

```bash
python guardllm/run_guardllm.py evaluate \
  --model-path /path/to/Qwen3-8B \
  --data-dir /path/to/guard_data \
  --split test \
  --adapter checkpoints/guardllm/rows-8640 \
  --output outputs/guardllm_test.jsonl
```

The evaluation reports TP, FN, FP, TN, hazardous recall, FPR, precision, class F1 scores, macro-F1, and accuracy.

