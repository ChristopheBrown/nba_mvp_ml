from __future__ import annotations

import numpy as np

from src.features.pipeline import build_candidate_feature_rows
from src.features.schema import load_feature_schema


def test_candidate_feature_rows_match_schema() -> None:
    schema = load_feature_schema()
    rows, player_names, player_ids, metadata = build_candidate_feature_rows(top_n=5)

    assert len(rows) == 5
    assert len(player_names) == 5
    assert len(player_ids) == 5
    assert len(metadata) == 5

    for row in rows:
        assert set(row.keys()) == set(schema.vector_order)
        assert all(not np.isnan(value) for value in row.values())
