"""Step 3: a simulated robot arm that learns where you want it to go from your "oops" signals.

The game
  * 5 cups sit in a half circle. The person secretly wants the arm to go to ONE of them.
  * The arm does not know which. It moves one cup left or right at a time.
  * After every move the person's brain reacts: a move AWAY from the wanted cup makes an
    "oops" (error potential); a move TOWARDS it does not.
  * The detector reads that brain signal. The arm updates its guess (a probability for
    every cup) and moves again, until it is 90% sure. Then it reaches for that cup.

The brain signals are REAL recorded EEG trials of one person. The detector was trained
only on the OTHER 5 people (the "new person" detector from step 2b), so this person is
a brand-new user it has never seen.

    python step3_sim_arm.py --person 2          # play 300 games with person 2 + save an animation
    python step3_sim_arm.py --synthetic         # quick test, no download
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline

N_CUPS, SURE, MAX_MOVES = 5, 0.90, 40
BLUE, ORANGE, GREEN, INK, INK2, GRID, SURFACE = "#2a78d6", "#eb6834", "#1f9e74", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ----------------------------------------------------------------------------- detector
def make_detector():
    return make_pipeline(XdawnCovariances(nfilter=4, estimator="oas"), TangentSpace(metric="riemann"),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


def train_on_others(data, person):
    """Train on the other people. Also estimate how often the detector is right, using
    only those other people (leave-one-person-out), so the test person stays unseen."""
    others = [s for s in data if s != person]
    X = np.concatenate([data[s][0] for s in others])
    y = np.concatenate([data[s][1] for s in others])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in others])
    print("  checking how reliable the detector is on the other people...", flush=True)
    pred = cross_val_predict(make_detector(), X, y, groups=g, cv=LeaveOneGroupOut())
    hit = np.mean(pred[y == 1] == 1)    # oops detected when there was an error
    false = np.mean(pred[y == 0] == 1)  # oops detected when the move was fine
    print("  training the final detector on all other people...", flush=True)
    det = make_detector().fit(X, y)
    return det, hit, false


# ----------------------------------------------------------------------------- the game
def play(oops_pool, ok_pool, hit, false, rng, target=None, record=False):
    """One game. oops_pool / ok_pool: the detector's decisions (True = 'oops') on this
    person's real error / correct trials. Each move uses one real trial of the right kind."""
    target = rng.integers(N_CUPS) if target is None else target
    pos = N_CUPS // 2
    belief = np.full(N_CUPS, 1 / N_CUPS)
    frames = [dict(pos=pos, prev=pos, belief=belief.copy(), oops=None, move=0)] if record else None
    for move in range(1, MAX_MOVES + 1):
        guess = int(np.argmax(belief))
        if guess != pos:
            step = 1 if guess > pos else -1
        else:  # already at its best guess: move to a neighbour to ask the brain again
            step = rng.choice([-1, 1]) if 0 < pos < N_CUPS - 1 else (1 if pos == 0 else -1)
        new = pos + step
        cups = np.arange(N_CUPS)
        away = np.abs(new - cups) > np.abs(pos - cups)  # for each possible wanted cup: was this move away?
        brain_error = bool(away[target])                 # what the person's brain really thinks
        pool = oops_pool if brain_error else ok_pool
        said_oops = bool(pool[rng.integers(len(pool))])  # detector's answer on a real EEG trial
        like = np.where(away, hit if said_oops else 1 - hit, false if said_oops else 1 - false)
        belief = belief * like
        belief /= belief.sum()
        if record:
            frames.append(dict(pos=new, prev=pos, belief=belief.copy(), oops=said_oops, move=move,
                               truth=brain_error))
        pos = new
        if belief.max() >= SURE:
            break
    choice = int(np.argmax(belief))
    return choice == target, move, frames, target, choice


# ----------------------------------------------------------------------------- drawing
def cup_xy(i, r=1.0):
    a = np.radians(160 - 140 * i / (N_CUPS - 1))
    return r * np.cos(a), r * np.sin(a)


def arm_points(x, y, l1=0.62, l2=0.62):
    """Two-link arm reaching (x, y) from the base at (0, 0) (simple inverse kinematics)."""
    d = np.clip(np.hypot(x, y), 1e-6, l1 + l2 - 1e-6)
    base, bend = np.arctan2(y, x), np.arccos((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d))
    elbows = [(l1 * np.cos(base + s * bend), l1 * np.sin(base + s * bend)) for s in (1, -1)]
    ex, ey = max(elbows, key=lambda e: e[1])  # keep the elbow pointing up
    return [0, ex, x], [0, ey, y]


def animate(frames, target, choice, path, person_label):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1.5, 1]}, facecolor=SURFACE)
    sub = 6  # animation frames per move
    timeline = []
    for f in frames[1:]:
        for k in range(sub):
            timeline.append((f, (k + 1) / sub))
    reach = dict(frames[-1], prev=frames[-1]["pos"], pos=choice, oops=None)  # final reach to the chosen cup
    for k in range(sub):
        timeline.append((reach, (k + 1) / sub))
    timeline += [(reach, 1.0)] * 12  # hold the end

    def draw(i):
        f, t = timeline[i]
        ax.clear(); bx.clear()
        for a in (ax, bx):
            a.set_facecolor(SURFACE)
        # cups
        for c in range(N_CUPS):
            cx, cy = cup_xy(c, 1.05)
            ax.add_patch(plt.Circle((cx, cy), 0.09, color=GRID, zorder=1))
            ax.text(cx, cy - 0.2, f"cup {c + 1}", ha="center", fontsize=8, color=INK2)
        tx, ty = cup_xy(target, 1.05)
        ax.add_patch(plt.Circle((tx, ty), 0.13, fill=False, ec=GREEN, lw=2, ls="--", zorder=2))
        ax.text(tx, ty + 0.17, "wanted", ha="center", fontsize=7, color=GREEN)
        # arm (smooth move between cups)
        p = f["prev"] + (f["pos"] - f["prev"]) * t
        a0 = np.radians(160 - 140 * p / (N_CUPS - 1))
        xs, ys = arm_points(0.95 * np.cos(a0), 0.95 * np.sin(a0))
        ax.plot(xs, ys, color=INK, lw=6, solid_capstyle="round", zorder=3)
        ax.plot(xs[:2], ys[:2], "o", color=INK2, ms=9, zorder=4)
        ax.add_patch(plt.Rectangle((-0.18, -0.12), 0.36, 0.12, color=INK2, zorder=2))
        done = i >= len(timeline) - 12 - sub
        if f["oops"] is not None and t == 1.0:
            ax.text(xs[2], ys[2] + 0.12, "oops!" if f["oops"] else "ok", ha="center", fontsize=10,
                    color=ORANGE if f["oops"] else BLUE, weight="bold", zorder=5)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.2, 1.4); ax.set_aspect("equal"); ax.axis("off")
        title = f"Move {f['move']}"
        if done:
            ok = choice == target
            title = f"Done after {f['move']} moves: reaches for cup {choice + 1} " + ("(correct!)" if ok else "(wrong)")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        # belief bars
        cols = [GREEN if c == target else BLUE for c in range(N_CUPS)]
        bx.bar(np.arange(N_CUPS), 100 * f["belief"], color=cols, width=0.7)
        bx.axhline(100 * SURE, color=INK2, lw=1, ls="--")
        bx.text(N_CUPS - 0.6, 100 * SURE + 2, "sure enough", fontsize=7, color=INK2, ha="right")
        bx.set_ylim(0, 105); bx.set_xticks(range(N_CUPS), [f"{c + 1}" for c in range(N_CUPS)])
        bx.set_xlabel("cup", fontsize=8, color=INK2); bx.set_ylabel("robot's guess (%)", fontsize=8, color=INK2)
        bx.set_title("Which cup does the person want?", loc="left", fontsize=9, color=INK)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        bx.tick_params(labelsize=8, colors=INK2)
        fig.suptitle(f"Robot arm guided by real brain 'oops' signals, {person_label}", x=0.02, ha="left",
                     fontsize=11, color=INK)

    anim = FuncAnimation(fig, draw, frames=len(timeline), interval=90)
    anim.save(path, writer=PillowWriter(fps=11))
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--person", type=int, default=2, choices=range(1, 7))
    p.add_argument("--games", type=int, default=300)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    if a.synthetic:
        from step2b_all_subjects import synthetic_people
        data = synthetic_people()
        label = f"fake person {a.person}"
    else:
        from errp_data import load_subject
        data = {s: load_subject(s) for s in range(1, 7)}
        label = f"person {a.person} (never seen in training)"

    print(f"Training the detector on everyone except person {a.person}...", flush=True)
    det, hit, false = train_on_others(data, a.person)
    print(f"  on the other people it catches {100 * hit:.0f}% of errors and gives a false 'oops' on "
          f"{100 * false:.0f}% of fine moves")

    X, y = data[a.person][0], data[a.person][1]
    said = det.predict(X) == 1
    oops_pool, ok_pool = said[y == 1], said[y == 0]
    print(f"On person {a.person}'s real EEG: catches {100 * oops_pool.mean():.0f}% of errors, "
          f"false 'oops' on {100 * ok_pool.mean():.0f}% of fine moves")

    rng = np.random.default_rng(0)
    results = [play(oops_pool, ok_pool, hit, false, rng)[:2] for _ in range(a.games)]
    wins = np.array([r[0] for r in results]); moves = np.array([r[1] for r in results])
    print(f"\n=== {a.games} games, person {a.person} ===")
    print(f"Arm reached the WANTED cup: {100 * wins.mean():.0f}% of games  (guessing would be {100 / N_CUPS:.0f}%)")
    print(f"Moves needed: {np.median(moves):.0f} on average (median), at most {MAX_MOVES}")

    # one example game for the animation (a fresh game, not cherry-picked)
    name = "synthetic" if a.synthetic else f"P{a.person}"
    _, _, frames, target, choice = play(oops_pool, ok_pool, hit, false, np.random.default_rng(1), record=True)
    path = out / f"step3_arm_{name}.gif"
    print("\nMaking the animation...", flush=True)
    animate(frames, target, choice, path, label)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
