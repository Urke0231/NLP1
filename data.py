"""Ucitavanje anotiranog skupa podataka."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# Kanonske oznake sentimenta. Ako u podacima koristite druge nazive
# (npr. "pozitivan"/"negativan"/"neutralan"), samo dopunite ovu mapu.
LABEL_MAP = {
    "positive": "positive", "pozitivan": "positive", "poz": "positive", "+1": "positive",
    "negative": "negative", "negativan": "negative", "neg": "negative", "-1": "negative",
    "neutral": "neutral", "neutralan": "neutral", "neu": "neutral", "0": "neutral",
    "mixed": "mixed", "mesovit": "mixed",
}

LABEL_ORDER = ["negative", "neutral", "positive", "mixed"]


def load_dataset(path: str | Path,
                 drop_duplicates: bool = True) -> pd.DataFrame:
    """Ucitava JSON (lista objekata) ili JSONL i vraca DataFrame.

    Ocekivana polja: tekst, sentiment, duzina, uneo.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        records = json.loads(text)
    else:  # JSONL
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    df = pd.DataFrame(records)

    required = {"tekst", "sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Nedostaju obavezna polja u podacima: {missing}")

    df["tekst"] = df["tekst"].astype(str).str.strip()
    df["sentiment"] = (
        df["sentiment"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    )
    if df["sentiment"].isna().any():
        bad = df.loc[df["sentiment"].isna()]
        raise ValueError(
            f"Nepoznate oznake sentimenta u {len(bad)} unosa. "
            f"Dopunite LABEL_MAP u data.py."
        )

    if "duzina" in df.columns:
        df["duzina"] = df["duzina"].astype(str).str.strip().str.lower()

    n_before = len(df)
    if drop_duplicates:
        df = df.drop_duplicates(subset=["tekst"]).reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"Uklonjeno {n_before - n_after} dupliranih tekstova.")

    df["broj_reci"] = df["tekst"].str.split().str.len()
    df["broj_karaktera"] = df["tekst"].str.len()
    return df


def xy(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    return df["tekst"].tolist(), df["sentiment"].tolist()


def present_labels(df: pd.DataFrame) -> List[str]:
    present = set(df["sentiment"])
    return [l for l in LABEL_ORDER if l in present]
