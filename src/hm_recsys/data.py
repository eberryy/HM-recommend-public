from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    history: pd.DataFrame
    validation: pd.DataFrame
    cutoff: pd.Timestamp
    data_end: pd.Timestamp


def _customer_sample_mask(customer_ids: pd.Series, sample_rate: float) -> pd.Series:
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    if sample_rate == 1:
        return pd.Series(True, index=customer_ids.index)
    hashes = pd.util.hash_pandas_object(customer_ids, index=False).to_numpy()
    threshold = int(sample_rate * 10_000)
    return pd.Series(hashes % 10_000 < threshold, index=customer_ids.index)


def infer_data_end(path: str | Path, chunksize: int = 2_000_000) -> pd.Timestamp:
    """Stream one column to find the latest transaction date."""
    latest: pd.Timestamp | None = None
    for chunk in pd.read_csv(path, usecols=["t_dat"], chunksize=chunksize):
        chunk_latest = pd.to_datetime(chunk["t_dat"], errors="raise").max()
        latest = chunk_latest if latest is None else max(latest, chunk_latest)
    if latest is None:
        raise ValueError("transactions file is empty")
    return latest


def load_temporal_split(
    path: str | Path,
    *,
    cutoff: str | None = None,
    history_weeks: int = 12,
    sample_rate: float = 0.01,
    chunksize: int = 2_000_000,
) -> TemporalSplit:
    """Load only recent history and the final seven-day holdout without leakage."""
    if history_weeks <= 0:
        raise ValueError("history_weeks must be positive")
    data_end = infer_data_end(path, chunksize) if cutoff is None else None
    cutoff_ts = (
        data_end - pd.Timedelta(days=6) if cutoff is None else pd.Timestamp(cutoff)
    )
    history_start = cutoff_ts - pd.Timedelta(weeks=history_weeks)
    validation_end = cutoff_ts + pd.Timedelta(days=7)

    frames: list[pd.DataFrame] = []
    dtype = {"customer_id": "string", "article_id": "int32"}
    usecols = ["t_dat", "customer_id", "article_id"]
    for chunk in pd.read_csv(
        path, usecols=usecols, dtype=dtype, chunksize=chunksize
    ):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"], errors="raise")
        time_mask = (chunk["t_dat"] >= history_start) & (
            chunk["t_dat"] < validation_end
        )
        chunk = chunk.loc[time_mask]
        if chunk.empty:
            continue
        chunk = chunk.loc[_customer_sample_mask(chunk["customer_id"], sample_rate)]
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        raise ValueError("no rows remain after time and customer sampling")
    recent = pd.concat(frames, ignore_index=True)
    observed_end = recent["t_dat"].max()
    if data_end is None:
        data_end = observed_end

    history = recent.loc[recent["t_dat"] < cutoff_ts].copy()
    validation = recent.loc[
        (recent["t_dat"] >= cutoff_ts) & (recent["t_dat"] < validation_end)
    ].copy()
    if validation.empty:
        raise ValueError("validation week is empty; check --cutoff")
    return TemporalSplit(history, validation, cutoff_ts, data_end)
