"""JSONL persistence for prediction snapshots and approved corrections."""

import json
import os

import fcntl


def _append_record(path, record):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as log_file:
        fcntl.flock(log_file.fileno(), fcntl.LOCK_EX)
        try:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())
        finally:
            fcntl.flock(log_file.fileno(), fcntl.LOCK_UN)


def read_records(path):
    if not os.path.exists(path):
        return []
    records_by_date = {}
    try:
        with open(path, "r", encoding="utf-8") as log_file:
            for line in log_file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("prediction_date"):
                    records_by_date[record["prediction_date"]] = record
    except OSError:
        return []
    return sorted(records_by_date.values(), key=lambda item: item.get("prediction_date", ""))


def write_records(path, records):
    for record in sorted(records, key=lambda item: item.get("prediction_date", "")):
        _append_record(path, record)


def upsert_record(path, record):
    try:
        _append_record(path, record)
    except OSError:
        pass
    return read_records(path)


def update_record(path, prediction_date, updates):
    records = read_records(path)
    current = next(
        (record for record in records if record.get("prediction_date") == prediction_date),
        None,
    )
    if current is not None:
        updated = dict(current)
        updated.update(updates)
        try:
            _append_record(path, updated)
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