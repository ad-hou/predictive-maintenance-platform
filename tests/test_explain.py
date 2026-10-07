import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import FEATURE_COLUMNS
from src.training.explain import top_factors
from src.training.model import ChampionModel


def _model(est, dataset, with_ref=True):
    d = dataset[dataset.label_known]
    est.fit(d[FEATURE_COLUMNS], d["failure_next_24h"])
    cal = IsotonicRegression(out_of_bounds="clip").fit(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    meta = {"reference_medians": d[FEATURE_COLUMNS].median().to_dict()} if with_ref else {}
    return ChampionModel("x", est, cal, 0.1, list(FEATURE_COLUMNS), meta=meta), d


def _risky_row(d):
    return d[d.failure_next_24h == 1].iloc[[10]]


def test_trained_model_explains_a_risky_row(trained, dataset):
    row = _risky_row(dataset[dataset.label_known])
    f = top_factors(trained["model"], row, 3)
    assert 0 < len(f) <= 3 and all(x["contribution"] > 0 for x in f)
    assert [x["contribution"] for x in f] == sorted((x["contribution"] for x in f), reverse=True)


def test_gradient_boosting_uses_the_occlusion_fallback(dataset):
    model, d = _model(HistGradientBoostingClassifier(max_depth=3, max_iter=60, random_state=0), dataset)
    assert top_factors(model, _risky_row(d), 3)


def test_logistic_regression_uses_coefficients(dataset):
    model, d = _model(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), dataset)
    assert top_factors(model, _risky_row(d), 3)


def test_unsupported_model_without_reference_returns_empty(dataset):
    model, d = _model(HistGradientBoostingClassifier(max_depth=3, max_iter=30, random_state=0), dataset, with_ref=False)
    assert top_factors(model, _risky_row(d), 3) == []
