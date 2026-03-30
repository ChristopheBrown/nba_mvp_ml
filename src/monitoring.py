from __future__ import annotations

import logging
from typing import Mapping

logger = logging.getLogger("nba_mvp_ml.monitoring")


def emit_metric(name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
    payload = {
        "name": name,
        "value": value,
        "tags": tags or {},
    }
    logger.info("metric_emitted %s", payload)
