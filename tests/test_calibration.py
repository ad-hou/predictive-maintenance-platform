import numpy as np

from src.training import metrics


def test_calibrator_is_monotonic(trained):
    cal = trained["model"].calibrator
    x = np.linspace(0, 1, 200)
    assert np.all(np.diff(cal.predict(x)) >= -1e-12)


def test_calibration_table_tracks_observed_frequency(trained):
    t = trained["calibration_table"]
    assert len(t) == 10
    # Overall, the mean predicted probability stays close to the observed rate (isotonic fit on another block).
    weights = t["n"] / t["n"].sum()
    predicted, observed = (t["predicted"] * weights).sum(), (t["observed"] * weights).sum()
    assert abs(predicted - observed) < 0.03


def test_calibration_table_on_perfectly_calibrated_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(0, 1, 50000) < p).astype(int)
    t = metrics.calibration_table(y, p)
    assert np.allclose(t["predicted"], t["observed"], atol=0.03)
