"""Loads the champion model exported by the registry (models/champion.joblib).

The file is re-read automatically when it changes, so a promotion or rollback is picked up
without restarting the API; /health shows which version is being served.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import joblib

from src import config
from src.training.model import ChampionModel


class ModelHolder:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or config.MODELS_DIR / "champion.joblib")
        self._mtime = 0.0
        self._lock = threading.Lock()
        self.model: ChampionModel | None = None
        self.loaded_at: str | None = None
        self.changed = False

    def get(self) -> ChampionModel | None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return self.model
        if mtime != self._mtime:
            with self._lock:
                if mtime != self._mtime:
                    self.model = joblib.load(self.path)
                    self._mtime = mtime
                    self.loaded_at = datetime.now(UTC).isoformat(timespec="seconds")
                    self.changed = True
        return self.model
