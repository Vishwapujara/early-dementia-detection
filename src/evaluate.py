"""Manual implementations of the inter-rater agreement metrics used to
benchmark model predictions against ground truth on the 4-class dementia
severity task: Scott's Pi and Quadratic Weighted Kappa (QWK).

Both metrics treat the model's predictions as a second "rater" alongside the
ground-truth labels and measure how much they agree beyond chance. Unlike
plain accuracy, they account for the amount of agreement expected by chance
given the class distribution, and QWK further penalizes distant
misclassifications (e.g. Non Demented -> Moderate) more than adjacent ones
(e.g. Non Demented -> Very Mild).
"""
import numpy as np


def _confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def scotts_pi(y_true, y_pred, num_classes=4):
    """Scott's Pi.

    Like Cohen's Kappa, but the expected agreement is computed from a single
    POOLED distribution over both raters (true labels + predictions) rather
    than from two independent marginals. This makes it more conservative
    than Cohen's Kappa when the two raters' marginal distributions differ.
    """
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    n = cm.sum()

    po = np.trace(cm) / n

    row_marginals = cm.sum(axis=1)
    col_marginals = cm.sum(axis=0)
    pooled = (row_marginals + col_marginals) / (2 * n)
    pe = np.sum(pooled ** 2)

    return float((po - pe) / (1 - pe))


def quadratic_weighted_kappa(y_true, y_pred, num_classes=4):
    """Quadratic Weighted Kappa (QWK).

    Weights disagreements by squared class distance, so confusing adjacent
    severity stages costs far less than confusing opposite ends of the
    scale. Expected agreement uses independent row/col marginals (the
    standard Cohen's Kappa expectation), unlike Scott's Pi above.
    """
    cm = _confusion_matrix(y_true, y_pred, num_classes)
    n = cm.sum()

    i_idx, j_idx = np.meshgrid(np.arange(num_classes), np.arange(num_classes), indexing='ij')
    weights = ((i_idx - j_idx) ** 2) / ((num_classes - 1) ** 2)

    row_marginals = cm.sum(axis=1)
    col_marginals = cm.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals) / n

    observed_weighted = np.sum(weights * cm)
    expected_weighted = np.sum(weights * expected)

    return float(1 - observed_weighted / expected_weighted)


def evaluate_predictions(y_true, y_pred, num_classes=4):
    """Return accuracy, Scott's Pi, and QWK for a set of predictions in one dict."""
    return {
        'accuracy': accuracy(y_true, y_pred),
        'scotts_pi': scotts_pi(y_true, y_pred, num_classes),
        'qwk': quadratic_weighted_kappa(y_true, y_pred, num_classes),
    }
