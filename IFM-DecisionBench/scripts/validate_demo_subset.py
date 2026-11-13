"""Validate the public IFM-DecisionBench demonstration subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED_FIELDS = {
    "benchmark_id",
    "matched_group_id",
    "domain_id",
    "domain_name",
    "risk_id",
    "risk_name",
    "scenario_id",
    "scenario_name",
    "pressure_id",
    "pressure_name",
    "prompt",
}
PRESSURES = {"P0", "P1", "P2", "P3", "P4"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(rows) != 44:
        raise ValueError(f"Expected 44 rows, found {len(rows)}")

    ids = [row["benchmark_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate benchmark IDs detected")

    groups: dict[str, list[dict]] = defaultdict(list)
    combinations = set()
    domains = Counter()
    for row in rows:
        if set(row) != EXPECTED_FIELDS:
            raise ValueError(f"Unexpected schema in {row.get('benchmark_id')}")
        if not isinstance(row["prompt"], str) or not row["prompt"].strip():
            raise ValueError(f"Empty prompt in {row['benchmark_id']}")
        groups[row["matched_group_id"]].append(row)

    if len(groups) != 40:
        raise ValueError(f"Expected 40 groups, found {len(groups)}")

    complete_groups = 0
    for group_id, items in groups.items():
        pressures = {item["pressure_id"] for item in items}
        if len(items) == 5 and pressures == PRESSURES:
            complete_groups += 1
        elif len(items) != 1 or pressures != {"P0"}:
            raise ValueError(f"Unexpected example structure: {group_id}")
        first = items[0]
        combinations.add((first["risk_id"], first["scenario_id"]))
        domains[first["domain_id"]] += 1

    expected_combinations = {
        (f"R{risk}", f"S{scenario}")
        for risk in range(1, 9)
        for scenario in range(1, 6)
    }
    if combinations != expected_combinations:
        raise ValueError("Risk and scenario coverage is incomplete")
    if complete_groups != 1:
        raise ValueError(f"Expected one complete pressure group, found {complete_groups}")
    if len(domains) != 12 or set(domains.values()) != {3, 4}:
        raise ValueError(f"Unexpected domain allocation: {dict(domains)}")

    print("Validation passed")
    print(f"Rows: {len(rows)}")
    print(f"Matched groups: {len(groups)}")
    print(f"Complete pressure examples: {complete_groups}")
    print(f"Domains: {dict(sorted(domains.items()))}")


if __name__ == "__main__":
    main()

