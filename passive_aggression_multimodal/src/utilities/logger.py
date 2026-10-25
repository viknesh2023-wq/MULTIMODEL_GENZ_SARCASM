"""Structured logging utilities for dataset collection and verification logs."""
import datetime
import json
import logging
import os
from typing import Any, Dict, Optional


def setup_console_logger(name: str = "multimodal_pa") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_event_jsonl(log_filepath: str, event_type: str, dataset_name: str, status: str, details: Optional[Dict[str, Any]] = None):
    """Appends an event as a single JSON line to log_filepath."""
    os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": event_type,
        "dataset_name": dataset_name,
        "status": status,
        "details": details or {}
    }
    with open(log_filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
