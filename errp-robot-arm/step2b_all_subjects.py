"""Step 2b: does the "oops" detector work for everyone, including brand-new users?

For all 6 people in the dataset:
  same person, other day : train on day 1, test on day 2 (and reverse), per person
  new person             : train on the other 5 people, test on this one
  new person + re-center : same, but each person's covariance matrices are first
                           re-centred on their own average (unsupervised, no labels
                           needed from the new person) - the trick from the main project

Reports the average and the WEAKEST person for each setting.

    python step2b_all_subjects.py              # downloads subjects 2-6 the first time
    python step2b_all_subjects.py --synthetic  # quick test, no download
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
from pyriemann.utils.base import invsqrtm
from sklearn.linear_model import LogisticRegression

from step2_detector import evaluate, scores

try:
    from pyriemann.geometry.mean import mean_riemann
except ImportError:
    from pyriemann.utils.mean import mean_riemann

SETTINGS = ["same person, other day", "new person", "new person + re-center"]
COLOURS = ["#2a78d6", "#eb6834", "#1f9e74"]


def recenter(C, groups):
    out = np.empty_like(C)
    for g in np.unique(groups):
        m = groups == g
        W = invsqrtm(mean_riemann(C[m]))
        out[m] = W @ C[m] @ W
    return out


def new_person(data, test_s, use_recenter):
    """Train xDAWN + tangent space + LR on all other people, test on test_s."""
    train = [s for s in data if s != test_s]
    Xtr = np.concatenate([data[s][0] for s in train])
    ytr = np.concatenate([data[s][1] for s in train])
    gtr = np.concatenate([np.full(len(data[s][1]), s) for s in train])
    Xte, yte = data[test_s][0], data[test_s][1]

    xd = XdawnCovariances(nfilter=4, estimator="oas").fit(Xtr, ytr)
    Ctr, Cte = xd.transform(Xtr), xd.transform(Xte)
    if use_recenter:
        Ctr = recenter(Ctr, gtr)
        Cte = recenter(Cte, np.zeros(len(Cte)))  # uses only the new person's unlabelled trials
    ts = TangentSpace(metric="riemann").fit(Ctr)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(ts.transform(Ctr), ytr)
    Z = ts.transform(Cte)
    return scores(yte, clf.predict(Z), clf.decision_function(Z))


def synthetic_people(n_people=6, seed=0):
    rng = np.random.default_rng(seed)
    times = np.arange(-0.2, 0.8 + 1e-9, 1 / 128)
    ch = ["Fz", "FC1", "FCz", "FC2", "C1", "Cz", "C2", "CPz"] + [f"X{i}" for i in range(8)]
    data = {}
    for s in range(1, n_people + 1):
        n = 400
        y = (rng.random(n) < 0.25).astype(int)
        mix = np.eye(len(ch)) + 0.5 * rng.normal(size=(len(ch), len(ch)))  # person-specific
        X = rng.normal(0, 10e-6, (n, len(ch), len(times)))
        strength = rng.uniform(0.4, 1.2)  # some people have a weak "oops"
        w = strength * (-9e-6 * np.exp(-((times - .275) / .03) ** 2) + 9e-6 * np.exp(-((times - .34) / .04) ** 2))
        X[y == 1, :8] += w
        X = np.einsum("ij,njt->nit", mix, X).astype(np.float32)
        data[s] = (X, y, np.repeat(np.arange(20), n // 20), ch, times)
    return data


def plot(df, path):
    subjects = sorted(df.subject.unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width = 0.26
    for i, (setting, colour) in enumerate(zip(SETTINGS, COLOURS)):
        vals = [df[(df.subject == s) & (df.setting == setting)]["balanced accuracy (%)"].iloc[0] for s in subjects]
        ax.bar(np.arange(len(subjects)) + (i - 1) * width, vals, width * 0.92, color=colour, label=setting)
    ax.axhline(50, color="#52514e", linewidth=1, linestyle="--")
    ax.text(len(subjects) - 0.5, 51, "chance", color="#52514e", fontsize=8, ha="right")
    ax.set_xticks(np.arange(len(subjects)), [f"Person {s}" for s in subjects])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Balanced accuracy (%)", color="#52514e", fontsize=9)
    ax.set_title("ErrP detection per person: same person vs brand-new person", loc="left", color="#0b0b0b", fontsize=11, pad=24)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
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
        data = {}
        for s in range(1, 7):
            print(f"Loading person {s} of 6 (downloads the first time, please wait)...", flush=True)
            data[s] = load_subject(s)

    rows = []
    for s, (X, y, run, ch, times) in data.items():
        print(f"Person {s}: {len(y)} moves, {int(y.sum())} errors", flush=True)
        same = evaluate(X, y, run, list(ch), times)
        r = same[same.detector == "xdawn-TS"].iloc[0].drop("detector").to_dict()
        rows.append({"subject": s, "setting": SETTINGS[0], **r})
        rows.append({"subject": s, "setting": SETTINGS[1], **new_person(data, s, False)})
        rows.append({"subject": s, "setting": SETTINGS[2], **new_person(data, s, True)})
    df = pd.DataFrame(rows)

    name = "synthetic" if a.synthetic else "all"
    df.to_csv(out / f"step2b_{name}_per_person.csv", index=False)
    summ = df.groupby("setting", sort=False).agg(
        **{"errors caught (%)": ("errors caught (%)", "mean"),
           "correct left alone (%)": ("correct moves left alone (%)", "mean"),
           "balanced acc. mean (%)": ("balanced accuracy (%)", "mean"),
           "balanced acc. WORST person (%)": ("balanced accuracy (%)", "min"),
           "AUC mean": ("AUC", "mean")}).reset_index()
    summ.to_csv(out / f"step2b_{name}_summary.csv", index=False)

    print("\n=== Per person: balanced accuracy (%) ===")
    print(df.pivot(index="subject", columns="setting", values="balanced accuracy (%)")[SETTINGS]
          .to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n=== Summary (xDAWN + Riemannian detector) ===")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.2f}" if v <= 1.0001 else f"{v:.1f}"))
    plot(df, out / f"step2b_{name}.png")
    print(f"\nSaved results/step2b_{name}_*.csv and results/step2b_{name}.png")


if __name__ == "__main__":
    main()
