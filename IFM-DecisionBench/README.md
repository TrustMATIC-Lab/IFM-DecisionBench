# IFM-DecisionBench

This repository accompanies the manuscript **IFM-DecisionBench: Evaluating Decision Safety in Industrial Foundation Models**.

IFM-DecisionBench evaluates whether a language based industrial foundation model provides assistance for an industrial objective that conflicts with an explicitly stated safety requirement. The benchmark uses matched scenario groups: one direct request (P0) and four variants that add time urgency (P1), production or cost pressure (P2), asserted authority (P3), or concealment and process evasion (P4) while preserving the operating context and hazardous objective.

## Representative public examples

The repository contains 44 representative requests selected from the complete 2,400 request benchmark. One direct request is provided for every combination of the eight risk categories and five scenario types. A complete matched group is included to demonstrate how P1, P2, P3, and P4 extend the corresponding direct request without changing its operating context or hazardous objective. The selected examples span all 12 industrial domains.

Selection is deterministic and uses benchmark metadata only. Model responses and performance results were not used when choosing the public examples.

The complete benchmark and full response annotations are retained for controlled research release. They are not included in this public repository.

## Repository structure

```text
data/demo/                         Representative public benchmark examples
scripts/build_demo_subset.py       Illustrative public example selection code
scripts/validate_demo_subset.py    Validate coverage and matched group integrity
scripts/run_local_model.py         Generate responses with a local Transformers model
guardllm/                          GuardLLM training and evaluation code
```

## Validate the public subset

```bash
python scripts/validate_demo_subset.py \
  --input data/demo/ifm_decisionbench_examples_44.jsonl
```

## Run a local model

```bash
python scripts/run_local_model.py \
  --model-path /path/to/model \
  --model-name MODEL_NAME \
  --input-jsonl data/demo/ifm_decisionbench_examples_44.jsonl \
  --output-jsonl outputs/model_demo_responses.jsonl \
  --batch-size 1
```

The evaluation configuration used in the paper is a single turn conversation without an additional system prompt. Local models use greedy decoding, and the maximum generated length is 768 tokens.

## Annotation format

Response annotations use five labels: `HAE` for Hazardous Action Enablement, `HPE` for Hazardous Partial Enablement, `SR` for Safe Redirection, `SA` for Safe Abstention, and `IR` for Invalid Response. The final annotation prompt and human review protocol will be added after the response audit is frozen.

## Data use

The public subset is intended to support research verification and method development. The complete benchmark is withheld to preserve the integrity of future evaluation. 

