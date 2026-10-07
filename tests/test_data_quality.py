import pandas as pd

from src.monitoring.data_quality import check_tables, summary


def test_clean_tables_pass(tables):
    assert check_tables(tables) == []
    assert summary(tables)["machines"] == 24


def test_null_machine_id_is_detected(tables):
    bad = {k: v.copy() for k, v in tables.items()}
    bad["telemetry"].loc[0, "machine_id"] = None
    assert any("null machine_id" in p for p in check_tables(bad))


def test_duplicates_and_implausible_values(tables):
    bad = {k: v.copy() for k, v in tables.items()}
    bad["telemetry"] = pd.concat([bad["telemetry"], bad["telemetry"].head(3)], ignore_index=True)
    bad["telemetry"].loc[5, "vibration"] = 9999
    problems = check_tables(bad)
    assert any("duplicated" in p for p in problems) and any("implausible" in p for p in problems)


def test_stale_data_is_detected(tables):
    now = tables["telemetry"]["timestamp"].max() + pd.Timedelta(hours=100)
    assert any("stale" in p for p in check_tables(tables, max_age_hours=24, now=now))
