"""Step 2c: "re-zero the scale each day".

Test A from step 2b (same person: learn on day 1, test on day 2, and the reverse)
failed for person 6 because their day-2 signals shifted, so almost no error move
crossed the detector's line.

Fix tried here: re-centre each day's brain data on that day's own average
(Riemannian re-centring). It uses NO labels from the test day, only its raw EEG,
so a real user would not have to tell the system which moves were errors.

    python step2c_rezero_days.py              # uses the data already downloaded in step 2b
    python step2c_rezero_days.py --synthetic  # quick test, no download
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression

from step2_detector import scores
from step2b_all_subjects import recenter

SETTINGS = ["same person (no fix)", "same person + re-zero each day"]
COLOURS = ["#2a78d6", "#1f9e74"]


def same_person(X, y, run, use_rezero):
    """Train on one day, test on the other (both ways), average the two."""
    day = (run >= 10).astype(int) if run.max() >= 10 else (run >= np.median(run)).astype(int)
    folds = []
    for train_day in (0, 1):
        tr, te = day == train_day, day != train_day
        xd = XdawnCovariances(nfilter=4, estimator="oas").fit(X[tr], y[tr])
        Ctr, Cte = xd.transform(X[tr]), xd.transform(X[te])
        if use_rezero:
            Ctr = recenter(Ctr, np.zeros(len(Ctr)))  # day used for learning
            Cte = recenter(Cte, np.zeros(len(Cte)))  # test day: unlabelled trials only
        ts = TangentSpace(metric="riemann").fit(Ctr)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(ts.transform(Ctr), y[tr])
        Z = ts.transform(Cte)
        folds.append(scores(y[te], clf.predict(Z), clf.decision_function(Z)))
    return {k: np.mean([f[k] for f in folds]) for k in folds[0]}


def synthetic_people(n_people=6, seed=1):
    """Fake people whose day-2 signals are shifted (like a scale reading wrong)."""
    rng = np.random.default_rng(seed)
    times = np.arange(-0.2, 0.8 + 1e-9, 1 / 128)
    ch = ["Fz", "FC1", "FCz", "FC2", "C1", "Cz", "C2", "CPz"] + [f"X{i}" for i in range(8)]
    data = {}
    for s in range(1, n_people + 1):
        n = 400
        y = (rng.random(n) < 0.25).astype(int)
        run = np.repeat(np.arange(20), n // 20)
        X = rng.normal(0, 10e-6, (n, len(ch), len(times)))
        w = -9e-6 * np.exp(-((times - .275) / .03) ** 2) + 9e-6 * np.exp(-((times - .34) / .04) ** 2)
        X[y == 1, :8] += w
        mix = np.eye(len(ch)) + 0.3 * rng.normal(size=(len(ch), len(ch)))
        X = np.einsum("ij,njt->nit", mix, X)
        shift = np.eye(len(ch)) + (0.1 * s) * rng.normal(size=(len(ch), len(ch)))  # bigger shift for later people
        d2 = run >= 10
        X[d2] = np.einsum("ij,njt->nit", shift, X[d2])
        data[s] = (X.astype(np.float32), y, run, ch, times)
    return data


def plot(df, path):
    subjects = sorted(df.subject.unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width = 0.36
    for i, (setting, colour) in enumerate(zip(SETTINGS, COLOURS)):
        vals = [df[(df.subject == s) & (df.setting == setting)]["balanced accuracy (%)"].iloc[0] for s in subjects]
        ax.bar(np.arange(len(subjects)) + (i - 0.5) * width, vals, width * 0.92, color=colour, label=setting)
    ax.axhline(50, color="#52514e", linewidth=1, linestyle="--")
    ax.text(len(subjects) - 0.5, 51, "chance", color="#52514e", fontsize=8, ha="right")
    ax.set_xticks(np.arange(len(subjects)), [f"Person {s}" for s in subjects])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Balanced accuracy (%)", color="#52514e", fontsize=9)
    ax.set_title("Same person, other day: does re-zeroing each day help?", loc="left",
                 color="#0b0b0b", fontsize=11, pad=24)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    if a.synthetic:
        data = synthetic_people()
    else:
        from errp_data import load_subject
        data = {s: load_subject(s) for s in range(1, 7)}

    rows = []
    for s, (X, y, run, ch, times) in data.items():
        print(f"Person {s}...", flush=True)
        rows.append({"subject": s, "setting": SETTINGS[0], **same_person(X, y, run, False)})
        rows.append({"subject": s, "setting": SETTINGS[1], **same_person(X, y, run, True)})
    df = pd.DataFrame(rows)

    name = "synthetic" if a.synthetic else "all"
    df.to_csv(out / f"step2c_{name}_per_person.csv", index=False)

    print("\n=== Per person ===")
    show = df.copy()
    show["setting"] = show["setting"].map({SETTINGS[0]: "no fix", SETTINGS[1]: "re-zero"})
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}" if v <= 1.0001 else f"{v:.1f}"))

    summ = df.groupby("setting", sort=False).agg(
        **{"errors caught (%)": ("errors caught (%)", "mean"),
           "correct left alone (%)": ("correct moves left alone (%)", "mean"),
           "balanced acc. mean (%)": ("balanced accuracy (%)", "mean"),
           "balanced acc. WORST person (%)": ("balanced accuracy (%)", "min"),
           "AUC mean": ("AUC", "mean")}).reset_index()
    summ.to_csv(out / f"step2c_{name}_summary.csv", index=False)
    print("\n=== Summary ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}" if v <= 1.0001 else f"{v:.1f}"))

    plot(df, out / f"step2c_{name}.png")
    print(f"\nSaved results/step2c_{name}_*.csv and results/step2c_{name}.png")


if __name__ == "__main__":
    main()
