"""Step 2: an ErrP detector that decides "oops or not?" from ONE cursor move.

Two detectors are compared:
  window-LDA : top-centre sensors, 150-600 ms after the move, shrinkage LDA
  xdawn-TS   : xDAWN spatial filters + Riemannian tangent space + logistic regression

Fair test (like the original study): train on day 1, test on day 2, and the
reverse; the two scores are averaged. Runs 0-9 come from day 1, runs 10-19 from day 2.

    python step2_detector.py --subject 1
    python step2_detector.py --synthetic      # quick test, no download
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TOP_CENTRE = ["Fz", "FC1", "FCz", "FC2", "C1", "Cz", "C2", "CPz"]


class TimeWindow(BaseEstimator, TransformerMixin):
    """Keep chosen sensors and a time window, average every `step` samples, flatten."""

    def __init__(self, ch_idx, t_idx, step=4):
        self.ch_idx, self.t_idx, self.step = ch_idx, t_idx, step

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Z = X[:, self.ch_idx][:, :, self.t_idx]
        n = Z.shape[2] // self.step * self.step
        Z = Z[:, :, :n].reshape(Z.shape[0], Z.shape[1], -1, self.step).mean(-1)
        return Z.reshape(len(Z), -1)


def make_detectors(ch_names, times):
    ch_idx = [i for i, c in enumerate(ch_names) if c in TOP_CENTRE] or list(range(min(8, len(ch_names))))
    t_idx = np.where((times >= 0.15) & (times <= 0.6))[0]
    window_lda = make_pipeline(
        TimeWindow(ch_idx, t_idx), StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]))

    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace
    xdawn_ts = make_pipeline(
        XdawnCovariances(nfilter=4, estimator="oas"), TangentSpace(metric="riemann"),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    return {"window-LDA": window_lda, "xdawn-TS": xdawn_ts}


def scores(y_true, y_pred, y_score):
    return {
        "errors caught (%)": 100 * np.mean(y_pred[y_true == 1] == 1),
        "correct moves left alone (%)": 100 * np.mean(y_pred[y_true == 0] == 0),
        "balanced accuracy (%)": 100 * balanced_accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
    }


def evaluate(X, y, run, ch_names, times):
    day = (run >= 10).astype(int) if run.max() >= 10 else (run >= np.median(run)).astype(int)
    rows = []
    for name, det in make_detectors(ch_names, times).items():
        per_fold = []
        for train_day in (0, 1):
            tr, te = day == train_day, day != train_day
            det.fit(X[tr], y[tr])
            score = det.decision_function(X[te])
            per_fold.append(scores(y[te], det.predict(X[te]), score))
        avg = {k: np.mean([f[k] for f in per_fold]) for k in per_fold[0]}
        rows.append({"detector": name, **avg})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", type=int, default=1, choices=range(1, 7))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    if a.synthetic:
        from step1_see_the_oops import synthetic
        X, y, _, ch, times = synthetic()
        run = np.repeat(np.arange(20), int(np.ceil(len(y) / 20)))[: len(y)]
        name = "synthetic"
    else:
        from errp_data import load_subject
        X, y, run, ch, times = load_subject(a.subject)
        name = f"S{a.subject:02d}"

    print(f"Subject {name}: {len(y)} moves ({int(y.sum())} errors). Training on one day, testing on the other...")
    df = evaluate(X, y, run, list(ch), times)
    print("\n" + df.to_string(index=False, float_format=lambda v: f"{v:.2f}" if v < 1.0001 else f"{v:.1f}"))
    print("\nGuessing at random would give 50% errors caught, 50% correct moves left alone, AUC 0.50.")
    df.to_csv(out / f"step2_{name}.csv", index=False)
    print(f"Saved {out / f'step2_{name}.csv'}")


if __name__ == "__main__":
    main()
