"""In-memory data store for the API (history per machine, cached features and latest scores)."""

from __future__ import annotations

import threading

import pandas as pd

from src import config
from src.features.build_features import make_dataset
from src.ingestion.reader import read_tables


class Store:
    def __init__(self, source: str | None = None, directory=None):
        self.source, self.directory = source, directory
        self.lock = threading.Lock()
        self.tables: dict[str, pd.DataFrame] = {}
        self.features = pd.DataFrame()
        self.by_machine: dict[str, dict[str, pd.DataFrame]] = {}
        self.loaded = False

    def refresh(self) -> None:
        tables = read_tables(self.source or config.DATA_SOURCE, self.directory)
        features = make_dataset(tables)
        by_machine = {}
        ids = tables["machines"]["machine_id"].tolist()
        grouped = {t: {k: v for k, v in tables[t].groupby("machine_id")} for t in ("telemetry", "errors", "maintenance", "failures")}
        empty = {t: tables[t].iloc[0:0] for t in ("telemetry", "errors", "maintenance", "failures")}
        for m in ids:
            by_machine[m] = {t: grouped[t].get(m, empty[t]) for t in grouped}
            by_machine[m]["machine"] = tables["machines"][tables["machines"]["machine_id"] == m]
        with self.lock:
            self.tables, self.features, self.by_machine, self.loaded = tables, features, by_machine, True

    def machine_ids(self) -> list[str]:
        return list(self.by_machine)

    def last_timestamp(self):
        return self.tables["telemetry"]["timestamp"].max() if self.loaded else None
