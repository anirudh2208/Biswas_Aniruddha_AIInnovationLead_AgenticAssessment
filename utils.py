"""
utils.py — Retry logic, data loading, and shared helpers.
"""

import csv
import json
import time
from pathlib import Path
from config import MAX_RETRIES, RETRY_DELAY


def call_with_retry(chain, inputs: dict, step_name: str) -> dict | str:
    """Call a LangChain chain with retry + exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = chain.invoke(inputs)
            return result.content if hasattr(result, "content") else result
        except Exception as e:
            print(f"    [WARN] Attempt {attempt}/{MAX_RETRIES} failed ({type(e).__name__}: {e})")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise RuntimeError(f"{step_name} failed after {MAX_RETRIES} retries: {e}") from e


def load_inquiries(filepath: Path) -> list[dict]:
    with open(filepath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_lease_clauses(filepath: Path) -> dict:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)
