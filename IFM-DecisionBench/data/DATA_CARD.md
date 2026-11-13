# IFM-DecisionBench representative examples

## Scope

This file contains 44 representative requests selected from IFM-DecisionBench. It demonstrates the benchmark structure and supports code verification.

## Coverage

The examples contain one direct request for every combination of the eight risk categories and five scenario types. One selected scenario additionally includes P1, P2, P3, and P4 to demonstrate the matched pressure design. The examples cover all 12 industrial domains.

## Selection rule

For risk category `Rr` and scenario type `Ss`, the domain used for the direct request example is

```text
1 + (((r - 1) * 5 + (s - 1)) mod 12)
```

The first selected scenario is also used to illustrate the four pressure variants. Selection depends only on benchmark metadata and does not use model responses, response labels, or performance results.

## Public fields

- `benchmark_id`
- `matched_group_id`
- `domain_id`
- `domain_name`
- `risk_id`
- `risk_name`
- `scenario_id`
- `scenario_name`
- `pressure_id`
- `pressure_name`
- `prompt`

Internal review notes, target responses, model outputs, and full benchmark records are excluded.


