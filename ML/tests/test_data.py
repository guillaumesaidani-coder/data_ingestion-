"""export_gold_dataset()/hash_file() : instantané versionnable du Gold
dataset, et son empreinte pour tracer quelle donnée a produit quel run.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indusense.data import export_gold_dataset, hash_file


def test_hash_file_is_deterministic_and_content_sensitive(tmp_path):
    f1 = tmp_path / "a.csv"
    f1.write_text("machine_id,value\nMACH-01,1\n", encoding="utf-8")
    f2 = tmp_path / "b.csv"
    f2.write_text("machine_id,value\nMACH-01,1\n", encoding="utf-8")
    f3 = tmp_path / "c.csv"
    f3.write_text("machine_id,value\nMACH-01,2\n", encoding="utf-8")

    assert hash_file(f1) == hash_file(f2)  # même contenu -> même hash
    assert hash_file(f1) != hash_file(f3)  # un octet différent -> hash différent


def test_export_gold_dataset_writes_csv_from_query(tmp_path, monkeypatch):
    captured = {}

    def fake_read_sql(query, engine):
        captured["query"] = query
        return pd.DataFrame({"machine_id": ["MACH-01"], "window_start": ["2025-06-01"]})

    monkeypatch.setattr("indusense.data.pd.read_sql", fake_read_sql)

    out_path = tmp_path / "gold" / "gold_dataset.csv"
    result = export_gold_dataset(engine=object(), out_path=out_path)

    assert result == out_path
    assert out_path.exists()
    assert "gold_machine_hourly_feature" in captured["query"]
    df = pd.read_csv(out_path)
    assert list(df["machine_id"]) == ["MACH-01"]
