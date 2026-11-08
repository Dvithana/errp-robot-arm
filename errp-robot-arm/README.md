# ErrP Robot Arm

A passion project: detect the brain's "oops" signal (error-related potential, ErrP) from EEG and use it to let a robot arm correct its own mistakes.

## Plan

1. **See the oops signal** in real EEG ✅ (subject 1: +10.5 µV difference at FCz, 336 ms)
2. **Build an ErrP detector** ✅, tested on all 6 people, including brand-new users
3. **Simulated robot arm** ← *we are here*: that undoes a move when an ErrP is detected (replaying recorded EEG)
4. Small Arduino servo arm driven by the detector

## Data

[BNCI 2015-013](https://doi.org/10.1109/TNSRE.2010.2053387): Chavarriaga & Millán (2010), EPFL. Six people watched a cursor move left or right towards a target; on 20–40% of moves it went the wrong way. 64 EEG channels, 512 Hz, two sessions per person. Downloaded automatically through [MOABB](https://moabb.neurotechx.com). Licence CC-BY-NC-ND 4.0 (non-commercial).

## Step 1: see the oops signal

```bash
pip install -r requirements.txt
python step1_see_the_oops.py --synthetic   # quick test, no download
python step1_see_the_oops.py --subject 1   # real data (large download the first time)
```

The figure in `results/` shows the average EEG at a fronto-central sensor (FCz) after correct and after wrong cursor moves.

## Step 2: detect "oops" from a single move

```bash
python step2_detector.py --subject 1
```

Trains on day 1 and tests on day 2 (and the reverse). Reports how many errors the detector catches, how many correct moves it leaves alone, balanced accuracy and AUC, for a simple window-LDA detector and an xDAWN + Riemannian detector.

Subject 1 result (xDAWN + Riemannian): 79.6% of errors caught, 91.6% of correct moves left alone, balanced accuracy 85.6%, AUC 0.94.

## Step 2b: does it work for everyone, including brand-new users?

```bash
python step2b_all_subjects.py              # downloads people 2-6 the first time
python step2b_all_subjects.py --synthetic  # quick test, no download
```

For each of the 6 people, three tests:

- **same person, other day**: train on day 1, test on day 2 (and the reverse)
- **new person**: train on the other 5 people, test on this one (no calibration)
- **new person + re-center**: same, but each person's data is first re-centred on their own average. This needs no labels from the new person.

Reports the average and the weakest person for each test, and saves a chart to `results/step2b_all.png`.

Result (6 people, balanced accuracy): same person 74.3% mean (worst 50.4%, person 6), new person 76.7% (worst 71.6%), new person + re-center 77.4% (worst 70.9%). Person 6 failed in the same-person test because their day-2 scores shifted below the detector's line (2% of errors caught, AUC still 0.70).

## Step 2c: re-zero each day

```bash
python step2c_rezero_days.py
```

Same-person test again, but each day's data is re-centred on that day's own average first (no labels needed). Does it rescue person 6?

Result: no real change (74.3% → 74.5%, person 6 still at chance, 51.8%). Re-centring the background activity did not fix the day-to-day drift of the error signal. The "new person" detector stays the most reliable.

## Step 3: a simulated arm that learns from "oops"

```bash
python step3_sim_arm.py --person 2
```

Five cups; the person secretly wants one. The arm moves one cup at a time, and after every move it reads a real recorded EEG trial from that person (an error trial if the move went away from the wanted cup, a correct trial otherwise). The detector, trained only on the other 5 people, says "oops" or "ok", and the arm updates its guess for every cup until it is 90% sure. Plays 300 games and saves an animation of one game to `results/step3_arm_P2.gif`.
