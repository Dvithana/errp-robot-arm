"""Step 3b: can the arm make fewer wrong grabs?

Same cup game as step 3, three rules compared:
  90% sure               : as before, grab as soon as one cup reaches 90%
  97% sure               : be more careful, wait until 97%
  90% sure + final check : at 90%, reach towards the cup and read the brain ONE more time.
                           "oops" -> don't grab, update the guesses and keep going.
                           "ok"   -> grab.
(Every reach counts as one move.)

    python step3b_safer_arm.py              # all 6 people (a few minutes each)
    python step3b_safer_arm.py --person 6   # one person
    python step3b_safer_arm.py --synthetic  # quick test, no download
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from step3_sim_arm import MAX_MOVES, N_CUPS, train_on_others

RULES = {"90% sure": (0.90, False), "97% sure": (0.97, False), "90% sure + final check": (0.90, True)}


def play(oops_pool, ok_pool, hit, false, rng, sure, final_check):
    target = rng.integers(N_CUPS)
    pos = N_CUPS // 2
    belief = np.full(N_CUPS, 1 / N_CUPS)
    cups = np.arange(N_CUPS)

    def brain_says_oops(is_error):  # detector's answer on one real EEG trial of the right kind
        pool = oops_pool if is_error else ok_pool
        return bool(pool[rng.integers(len(pool))])

    moves = 0
    while moves < MAX_MOVES:
        if belief.max() >= sure:
            choice = int(np.argmax(belief))
            if not final_check:
                return choice == target, moves
            moves += 1  # reach towards the cup and listen to the brain once more
            oops = brain_says_oops(choice != target)
            if not oops:
                return choice == target, moves
            # "oops" while reaching: less likely that `choice` is the wanted cup
            like = np.where(cups == choice, false, hit)
            belief = belief * like / (belief * like).sum()
            continue
        guess = int(np.argmax(belief))
        if guess != pos:
            step = 1 if guess > pos else -1
        else:
            step = rng.choice([-1, 1]) if 0 < pos < N_CUPS - 1 else (1 if pos == 0 else -1)
        new = pos + step
        away = np.abs(new - cups) > np.abs(pos - cups)
        oops = brain_says_oops(bool(away[target]))
        like = np.where(away, hit if oops else 1 - hit, false if oops else 1 - false)
        belief = belief * like / (belief * like).sum()
        pos = new
        moves += 1
    return int(np.argmax(belief)) == target, moves  # ran out of moves: grab the best guess


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--person", type=int, choices=range(1, 7))
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    if a.synthetic:
        from step2b_all_subjects import synthetic_people
        data = synthetic_people()
    else:
        from errp_data import load_subject
        data = {s: load_subject(s) for s in range(1, 7)}
    people = [a.person] if a.person else list(data)

    rows = []
    for s in people:
        print(f"\nPerson {s}: training the detector on the other people...", flush=True)
        det, hit, false = train_on_others(data, s)
        X, y = data[s][0], data[s][1]
        said = det.predict(X) == 1
        for rule, (sure, check) in RULES.items():
            rng = np.random.default_rng(0)
            res = [play(said[y == 1], said[y == 0], hit, false, rng, sure, check) for _ in range(a.games)]
            rows.append({"person": s, "rule": rule,
                         "right cup (%)": 100 * np.mean([r[0] for r in res]),
                         "moves (median)": np.median([r[1] for r in res])})
    df = pd.DataFrame(rows)

    name = "synthetic" if a.synthetic else ("all" if not a.person else f"P{a.person}")
    df.to_csv(out / f"step3b_{name}.csv", index=False)
    print(f"\n=== Right cup (%), {a.games} games per person and rule (guessing = {100 / N_CUPS:.0f}%) ===")
    t = df.pivot(index="person", columns="rule", values="right cup (%)")[list(RULES)]
    if len(t) > 1:
        t.loc["average"] = t.mean()
        t.loc["WORST person"] = t.drop("average").min()
    print(t.to_string(float_format=lambda v: f"{v:.1f}"))
    print("\n=== Moves needed (median) ===")
    m = df.pivot(index="person", columns="rule", values="moves (median)")[list(RULES)]
    print(m.to_string(float_format=lambda v: f"{v:.0f}"))
    print(f"\nSaved {out / f'step3b_{name}.csv'}")


if __name__ == "__main__":
    main()
