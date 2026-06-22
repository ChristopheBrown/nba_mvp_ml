from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

import mlflow.pyfunc
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings
from src.features.candidate_pool import CandidatePoolService


class ScriptModelHandler:
    def __init__(self) -> None:
        self.model = None
        self._model_version: str | None = None

    def load_model(self):
        if self.model is not None:
            return self.model
        target = settings.resolved_model_target
        self._model_version = Path(target).name
        self.model = mlflow.pyfunc.load_model(target)
        return self.model

    def predict(self, input_array: np.ndarray):
        if self.model is None:
            self.load_model()
        return self.model.predict(input_array)


KEY_PLAYERS: dict[int, Sequence[str]] = {
    2022: ["JOEL EMBIID", "NIKOLA JOKIĆ", "GIANNIS ANTETOKOUNMPO"],
    2023: ["NIKOLA JOKIĆ", "GIANNIS ANTETOKOUNMPO", "SHAI GILGEOUS-ALEXANDER"],
}

service = CandidatePoolService(ScriptModelHandler())
for season, names in KEY_PLAYERS.items():
    payload, _ = service.build_candidate_pool(season=season, pool_size=50, top_n=10, historical_mode=True)
    ranked = {row["player_name"].upper(): idx + 1 for idx, row in enumerate(payload["results"])}
    print(f"season {season}")
    for name in names:
        print(name, ranked.get(name, "not in top10"))
    print(json.dumps(payload["results"][:10], indent=2, ensure_ascii=False))
    print()
