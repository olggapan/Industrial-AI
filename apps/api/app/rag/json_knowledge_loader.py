from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.rag.schemas import LoadedDocument


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _bullet_list(items: list[Any]) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list in {path}")

    return [item for item in raw if isinstance(item, dict)]


async def load_error_code_documents(path: Path) -> list[LoadedDocument]:
    records = _read_json_list(path)
    documents: list[LoadedDocument] = []

    for item in records:
        code = str(item["code"])
        title = str(item["title"])

        content = f"""# {code} {title}

## Severity

{item.get("severity", "unknown")}

## Description

{item.get("description", "")}

## Likely Causes

{_bullet_list(item.get("likely_causes", []))}

## Recommended Checks

{_bullet_list(item.get("recommended_checks", []))}

## Safety Notes

{_bullet_list(item.get("safety_notes", []))}
"""

        documents.append(
            LoadedDocument(
                id=f"error-code-{_slug(code)}",
                title=f"{code} {title}",
                source_path=str(path),
                source_type="json_record",
                content=content,
                metadata={
                    "record_type": "error_code",
                    "error_code": code,
                    "severity": str(item.get("severity", "")),
                    "file_name": path.name,
                },
            )
        )

    return documents


async def load_repair_case_documents(path: Path) -> list[LoadedDocument]:
    records = _read_json_list(path)
    documents: list[LoadedDocument] = []

    for item in records:
        case_id = str(item["case_id"])
        title = str(item["title"])

        content = f"""# {case_id}: {title}

## Machine

Machine ID: {item.get("machine_id", "")}

## Severity and Status

Severity: {item.get("severity", "")}
Status: {item.get("status", "")}

## Symptoms

{_bullet_list(item.get("symptoms", []))}

## Error Codes

{_bullet_list(item.get("error_codes", []))}

## Telemetry Summary

{item.get("telemetry_summary", "")}

## Root Cause

{item.get("root_cause", "")}

## Actions Taken

{_bullet_list(item.get("actions_taken", []))}

## Outcome

{item.get("outcome", "")}

## Related Manual Sections

{_bullet_list(item.get("related_manual_sections", []))}
"""

        documents.append(
            LoadedDocument(
                id=f"repair-case-{_slug(case_id)}",
                title=f"{case_id}: {title}",
                source_path=str(path),
                source_type="json_record",
                content=content,
                metadata={
                    "record_type": "repair_case",
                    "case_id": case_id,
                    "machine_id": str(item.get("machine_id", "")),
                    "severity": str(item.get("severity", "")),
                    "error_codes": ",".join(
                        str(code).upper() for code in item.get("error_codes", [])
                    ),
                    "file_name": path.name,
                },
            )
        )

    return documents


async def load_machine_documents(path: Path) -> list[LoadedDocument]:
    records = _read_json_list(path)
    documents: list[LoadedDocument] = []

    for item in records:
        machine_id = str(item["id"])
        name = str(item["name"])

        next_maintenance = item.get("next_maintenance") or {}

        content = f"""# {machine_id}: {name}

## Equipment

Type: {item.get("equipment_type", "")}
Manufacturer: {item.get("manufacturer", "")}
Model: {item.get("model", "")}
Line: {item.get("line", "")}
Location: {item.get("location", "")}

## Status

Status: {item.get("status", "")}
Criticality: {item.get("criticality", "")}

## Service Information

Installed at: {item.get("installed_at", "")}
Last service at: {item.get("last_service_at", "")}

## Next Maintenance

Starts at: {next_maintenance.get("starts_at", "")}
Ends at: {next_maintenance.get("ends_at", "")}
Description: {next_maintenance.get("description", "")}

## Monitored Signals

{_bullet_list(item.get("monitored_signals", []))}
"""

        documents.append(
            LoadedDocument(
                id=f"machine-{_slug(machine_id)}",
                title=f"{machine_id}: {name}",
                source_path=str(path),
                source_type="json_record",
                content=content,
                metadata={
                    "record_type": "machine",
                    "machine_id": machine_id,
                    "equipment_type": str(item.get("equipment_type", "")),
                    "status": str(item.get("status", "")),
                    "file_name": path.name,
                },
            )
        )

    return documents


async def load_telemetry_documents(path: Path) -> list[LoadedDocument]:
    records = _read_json_list(path)
    documents: list[LoadedDocument] = []

    for item in records:
        scenario_id = str(item["scenario_id"])
        label = str(item["scenario_label"])

        readings = []
        for reading in item.get("readings", []):
            readings.append(
                f"- {reading.get('name')}: {reading.get('value')} {reading.get('unit')} "
                f"status={reading.get('status')} trend={reading.get('trend')}"
            )

        content = f"""# {scenario_id}: {label}

## Machine

Machine ID: {item.get("machine_id", "")}

## Summary

{item.get("summary", "")}

## Captured At

{item.get("captured_at", "")}

## Current Readings

{chr(10).join(readings) if readings else "- None"}
"""

        documents.append(
            LoadedDocument(
                id=f"telemetry-{_slug(scenario_id)}",
                title=f"{scenario_id}: {label}",
                source_path=str(path),
                source_type="json_record",
                content=content,
                metadata={
                    "record_type": "telemetry_snapshot",
                    "scenario_id": scenario_id,
                    "machine_id": str(item.get("machine_id", "")),
                    "file_name": path.name,
                },
            )
        )

    return documents


async def load_manual_metadata_documents(path: Path) -> list[LoadedDocument]:
    records = _read_json_list(path)
    documents: list[LoadedDocument] = []

    for item in records:
        manual_id = str(item["id"])
        title = str(item["title"])

        content = f"""# {manual_id}: {title}

## Manual Metadata

Kind: {item.get("kind", "")}
Domain: {item.get("domain", "")}
Machine type: {item.get("machine_type", "")}
Machine ID: {item.get("machine_id", "")}
Status: {item.get("status", "")}

## Sections

{_bullet_list(item.get("sections", []))}
"""

        documents.append(
            LoadedDocument(
                id=f"manual-metadata-{_slug(manual_id)}",
                title=f"{manual_id}: {title}",
                source_path=str(path),
                source_type="json_record",
                content=content,
                metadata={
                    "record_type": "manual_metadata",
                    "manual_id": manual_id,
                    "machine_id": str(item.get("machine_id", "")),
                    "file_name": path.name,
                },
            )
        )

    return documents


async def load_json_knowledge_documents(data_root: Path) -> list[LoadedDocument]:
    documents: list[LoadedDocument] = []

    error_codes_path = data_root / "synthetic" / "error_codes" / "error_codes.json"
    repair_cases_path = data_root / "synthetic" / "repair_cases" / "repair_cases.json"
    machines_path = data_root / "synthetic" / "machines" / "machines.json"
    telemetry_path = data_root / "synthetic" / "telemetry" / "current_snapshots.json"
    manuals_path = data_root / "synthetic" / "manuals" / "manuals.json"

    if error_codes_path.exists():
        documents.extend(await load_error_code_documents(error_codes_path))

    if repair_cases_path.exists():
        documents.extend(await load_repair_case_documents(repair_cases_path))

    if machines_path.exists():
        documents.extend(await load_machine_documents(machines_path))

    if telemetry_path.exists():
        documents.extend(await load_telemetry_documents(telemetry_path))

    if manuals_path.exists():
        documents.extend(await load_manual_metadata_documents(manuals_path))

    return documents
