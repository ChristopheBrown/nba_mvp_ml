from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Mapping

logger = logging.getLogger("nba_mvp_ml.monitoring")
_METRICS: dict[str, dict[str, object]] = {}


def emit_metric(name: str, value: float, tags: Mapping[str, str] | None = None) -> None:
    payload = {
        "name": name,
        "value": value,
        "tags": tags or {},
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    _METRICS[name] = payload
    logger.info("metric_emitted %s", payload)


def get_latest_metrics() -> list[dict[str, object]]:
    return list(_METRICS.values())
