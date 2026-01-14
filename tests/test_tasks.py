from __future__ import annotations

import pytest

from mcode.bench.tasks import load_benchmark


def test_load_benchmark_unknown_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        load_benchmark("not-a-real-benchmark", cache_dir=tmp_path)

