"""
Country column encoder.

``country`` is a weak-but-valuable signal: a business in France essentially
never matches a record in India. The problem statement explicitly warns
against hard-coding the label set, so countries are encoded as an *open* set -
codes are assigned in first-encounter order and unknown labels simply get a
fresh code.

The country is the last column of every source file, so it is extracted with a
plain string split. This avoids running the CSV parser over ~1 GB of text
purely to read four bytes per row.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np

from src.data.loader import fld

# Reserved code for a missing/blank country label.
UNKNOWN = 0

BLOCK_ROWS = 2_000_000


def scan_countries(tsv_path: str) -> Tuple[Dict[str, int], int]:
    """First pass: assign a stable code to every distinct country label."""
    codes: Dict[str, int] = {}
    n_rows = 0
    with open(tsv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        fh.readline()  # header
        for line in fh:
            n_rows += 1
            if line.count("\t") >= 3:
                label = line.rstrip("\r\n").rsplit("\t", 1)[-1].strip()
            else:
                label = ""
            if label and label not in codes:
                codes[label] = len(codes) + 1
    return codes, n_rows


def encode_countries(tsv_path: str, out_prefix: str, codes: Dict[str, int],
                     n_rows: int) -> np.ndarray:
    """Second pass: write the country code of every row as a uint8 array."""
    arr = np.lib.format.open_memmap(
        fld(out_prefix, "country"), mode="w+", dtype=np.uint8, shape=(n_rows,))
    # Decode table for the hot loop: a dict lookup is faster than a branchy
    # .get() with a default for the overwhelmingly common single hit.
    table = {k: v for k, v in codes.items()}

    row = 0
    with open(tsv_path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        fh.readline()  # header
        for line in fh:
            if row >= n_rows:
                break
            if line.count("\t") >= 3:
                label = line.rstrip("\r\n").rsplit("\t", 1)[-1].strip()
                arr[row] = table.get(label, UNKNOWN)
            else:
                arr[row] = UNKNOWN
            row += 1
    arr.flush()
    return arr


def encode_split(tsv_path: str, out_prefix: str, verbose: bool = True) -> Dict[str, int]:
    """Full two-pass country encode for one source file."""
    codes, n_rows = scan_countries(tsv_path)
    if verbose:
        top = sorted(codes.items(), key=lambda kv: -kv[1])
        pretty = ", ".join(f"{k}={v}" for k, v in top[:8])
        print(f"  {os.path.basename(tsv_path)}: {n_rows:,} rows, "
              f"{len(codes)} distinct countries [{pretty}]", flush=True)
    encode_countries(tsv_path, out_prefix, codes, n_rows)
    with open(fld(out_prefix, "country_map").replace(".npy", ".json"),
              "w", encoding="utf-8") as fh:
        json.dump(codes, fh, indent=2, sort_keys=True)
    return codes


class CountryCodes:
    """Lookup helper for a split's country codes."""

    def __init__(self, split_prefixes: List[str]):
        self.tables: List[np.ndarray] = []
        mapping: Dict[str, int] = {}
        for pref in split_prefixes:
            path = fld(pref, "country")
            if os.path.exists(path):
                self.tables.append(np.load(path, mmap_mode="r"))
            jpath = fld(pref, "country_map").replace(".npy", ".json")
            if os.path.exists(jpath):
                with open(jpath, encoding="utf-8") as fh:
                    mapping.update(json.load(fh))
        self.by_label = mapping

    def name(self, code: int) -> str:
        for label, val in self.by_label.items():
            if val == code:
                return label
        return "<unknown>"
