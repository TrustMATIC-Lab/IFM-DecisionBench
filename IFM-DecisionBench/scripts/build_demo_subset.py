
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PRESSURES = ["P0", "P1", "P2", "P3", "P4"]
PUBLIC_FIELDS = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
    return rows


def numeric_suffix(value: str, prefix: str) -> int:
    if not value.startswith(prefix):
        raise ValueError(f"Expected {prefix} identifier, found {value!r}")
    return int(value[len(prefix) :])


def public_row(row: dict) -> dict:
    domain_id = str(row["domain_id"])
    risk_number = numeric_suffix(str(row["risk_id"]), "C")
    risk_id = f"R{risk_number}"
    scenario_id = str(row["scenario_id"])
    pressure_id = str(row["pressure_id"])
    matched_group_id = f"{domain_id}-{risk_id}-{scenario_id}"
    result = {
        "benchmark_id": f"{matched_group_id}-{pressure_id}",
        "matched_group_id": matched_group_id,
        "domain_id": domain_id,
        "domain_name": row["domain_en"],
        "risk_id": risk_id,
        "risk_name": row["risk_name_en"],
        "scenario_id": scenario_id,
        "scenario_name": row["scenario_name_en"],
        "pressure_id": pressure_id,
        "pressure_name": row["pressure_name_en"],
        "prompt": row["prompt"],
    }
    if tuple(result) != PUBLIC_FIELDS:
        raise AssertionError("Unexpected public schema")
    return result


def main() -> None:
    args = parse_args()
    source = load_jsonl(args.input)
    if len(source) != 2400:
        raise ValueError(f"Expected 2,400 source rows, found {len(source)}")

    selected = []
    for row in source:
        risk_number = numeric_suffix(str(row["risk_id"]), "C")
        scenario_number = numeric_suffix(str(row["scenario_id"]), "S")
        selected_domain = 1 + (((risk_number - 1) * 5 + (scenario_number - 1)) % 12)
        selected_group = str(row["domain_id"]) == f"D{selected_domain:02d}"
        pressure_id = str(row["pressure_id"])
        direct_example = selected_group and pressure_id == "P0"
        pressure_example = selected_group and risk_number == 1 and scenario_number == 1
        if direct_example or pressure_example:
            selected.append(public_row(row))

    selected.sort(
        key=lambda row: (
            numeric_suffix(row["risk_id"], "R"),
            numeric_suffix(row["scenario_id"], "S"),
            numeric_suffix(row["pressure_id"], "P"),
        )
    )

    if len(selected) != 44:
        raise ValueError(f"Expected 44 selected rows, found {len(selected)}")
    groups = Counter(row["matched_group_id"] for row in selected)
    if len(groups) != 40 or sorted(groups.values()) != ([1] * 39 + [5]):
        raise ValueError("Unexpected representative example structure")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()

