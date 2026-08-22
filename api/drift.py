"""Population Stability Index (PSI) drift monitoring for deployed model predictions.

Compares each model's recent output distribution (from `api/database.py`'s
logged predictions) against a baseline: the real OASIS training set's class
distribution (67,222 Non Demented / 13,725 Very mild / 5,002 Mild / 488
Moderate, out of 86,437 total scans). A model whose predictions drift far
from that baseline — e.g. suddenly predicting "Moderate Dementia" far more
often than the ~0.56% baseline rate — is a signal worth investigating,
whether that turns out to be data drift in incoming scans or a regression
in the model itself.
"""
import math

from api.database import get_class_counts

_RAW_BASELINE_COUNTS = {
    "Non Demented": 67222,
    "Very mild Dementia": 13725,
    "Mild Dementia": 5002,
    "Moderate Dementia": 488,
}
_BASELINE_TOTAL = sum(_RAW_BASELINE_COUNTS.values())
BASELINE_DISTRIBUTION = {k: v / _BASELINE_TOTAL for k, v in _RAW_BASELINE_COUNTS.items()}

# Floor applied to both distributions so a class with zero observations in
# either one still contributes a finite (not infinite/NaN) term to the sum.
_EPSILON = 1e-4


def compute_psi(baseline, current):
    """PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))
    over the union of classes in both distributions.
    """
    classes = set(baseline) | set(current)
    psi = 0.0
    for cls in classes:
        base_pct = max(baseline.get(cls, 0.0), _EPSILON)
        curr_pct = max(current.get(cls, 0.0), _EPSILON)
        psi += (curr_pct - base_pct) * math.log(curr_pct / base_pct)
    return psi


def psi_verdict(psi):
    """Standard PSI thresholds: <0.1 stable, 0.1-0.2 moderate, >=0.2 significant."""
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate_drift"
    return "significant_drift"


def get_drift_status(model, limit=200, db_path=None):
    counts = get_class_counts(model, limit=limit, db_path=db_path)
    total = sum(counts.values())

    if total == 0:
        return {
            "model": model,
            "sample_size": 0,
            "psi": None,
            "verdict": "no_data",
            "baseline_distribution": {k: round(v, 4) for k, v in BASELINE_DISTRIBUTION.items()},
            "current_distribution": {},
        }

    current_distribution = {cls: n / total for cls, n in counts.items()}
    psi = compute_psi(BASELINE_DISTRIBUTION, current_distribution)

    return {
        "model": model,
        "sample_size": total,
        "psi": round(psi, 4),
        "verdict": psi_verdict(psi),
        "baseline_distribution": {k: round(v, 4) for k, v in BASELINE_DISTRIBUTION.items()},
        "current_distribution": {k: round(v, 4) for k, v in current_distribution.items()},
    }
