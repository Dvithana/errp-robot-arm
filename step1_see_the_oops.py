"""Step 1: see the brain's "oops" signal.

Plots the average EEG after CORRECT vs ERROR cursor moves at a fronto-central
sensor (FCz, or Cz if FCz is missing), where error potentials are strongest.

    python step1_see_the_oops.py              # subject 1 (downloads the data)
    python step1_see_the_oops.py --subject 2
    python step1_see_the_oops.py --synthetic  # quick test, no download
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CORRECT, ERROR, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def pick_channel(ch_names):
    for name in ("FCz", "FCZ", "Cz", "CZ"):
        if name in ch_names:
            return ch_names.index(name), name
    return 0, ch_names[0]


def plot_oops(X, y, ch_names, times, title, path):
    ci, cname = pick_channel(list(ch_names))
    t = times * 1000
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for label, colour, name in [(0, CORRECT, "Correct move"), (1, ERROR, "Error move (\"oops\")")]:
        sig = X[y == label, ci, :] * 1e6  # volts -> microvolts
        m = sig.mean(0)
        se = sig.std(0, ddof=1) / np.sqrt(len(sig))
        ax.fill_between(t, m - se, m + se, color=colour, alpha=0.18, linewidth=0)
        ax.plot(t, m, color=colour, linewidth=2, label=f"{name} (n = {len(sig)})")
    ax.axvline(0, color=INK2, linewidth=1, linestyle="--")
    ax.text(5, ax.get_ylim()[1], "cursor moves", color=INK2, fontsize=8, va="top")
    ax.axhline(0, color=GRID, linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel("Time after the cursor moves (ms)", color=INK2, fontsize=9)
    ax.set_ylabel(f"EEG at {cname} (µV)", color=INK2, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # The "oops" signal: biggest difference between error and correct, 150-600 ms
    diff = (X[y == 1, ci].mean(0) - X[y == 0, ci].mean(0)) * 1e6
    win = (t >= 150) & (t <= 600)
    k = np.argmax(np.abs(diff[win]))
    return cname, t[win][k], diff[win][k]


def synthetic(seed=0):
    """Fake data with an ErrP-like wave on error trials, for testing."""
    rng = np.random.default_rng(seed)
    times = np.arange(-0.2, 0.8 + 1e-9, 1 / 128)
    n, ch = 300, ["Fz", "FCz", "Cz"]
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(0, 4e-6, (n, len(ch), len(times)))
    wave = -5e-6 * np.exp(-((times - 0.25) / 0.04) ** 2) + 7e-6 * np.exp(-((times - 0.35) / 0.06) ** 2)
    X[y == 1] += wave
    return X.astype(np.float32), y, np.zeros(n, int), ch, times


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", type=int, default=1, choices=range(1, 7))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    if a.synthetic:
        X, y, run, ch, times = synthetic()
        name = "synthetic"
    else:
        from errp_data import load_subject
        print(f"Loading subject {a.subject} (first time downloads the recordings, this can take a while)...")
        X, y, run, ch, times = load_subject(a.subject)
        name = f"S{a.subject:02d}"

    print(f"{len(y)} cursor moves: {int((y == 0).sum())} correct, {int((y == 1).sum())} errors "
          f"({100 * y.mean():.0f}% errors), {X.shape[1]} channels, {len(np.unique(run))} runs")
    path = out / f"oops_{name}.png"
    cname, t_peak, amp = plot_oops(X, y, ch, times, f"The brain's \"oops\" signal, subject {name}", path)
    print(f"Biggest error-vs-correct difference at {cname}: {amp:+.1f} µV at {t_peak:.0f} ms after the move")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
