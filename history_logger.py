"""JSONL persistence for prediction snapshots and approved corrections."""

import json
import os


def read_records(path):
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as log_file:
            for line in log_file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("prediction_date"):
                    records.append(record)
    except OSError:
        return []
    return records


def write_records(path, records):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as log_file:
        for record in sorted(records, key=lambda item: item.get("prediction_date", "")):
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def upsert_record(path, record):
    records = [
        item for item in read_records(path)
        if item.get("prediction_date") != record.get("prediction_date")
    ]
    records.append(record)
    try:
        write_records(path, records)
    except OSError:
        pass
    return records


def update_record(path, prediction_date, updates):
    records = read_records(path)
    changed = False
    for record in records:
        if record.get("prediction_date") == prediction_date:
            record.update(updates)
            changed = True
    if changed:
        try:
            write_records(path, records)
        except OSError:
            pass
    return records


def read_adjustments(path):
    try:
        with open(path, "r", encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
        return settings if isinstance(settings, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_adjustments(path, settings):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as settings_file:
        json.dump(settings, settings_file, ensure_ascii=False, indent=2)