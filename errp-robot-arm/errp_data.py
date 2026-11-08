"""Load error-related potentials (ErrPs) from the EPFL monitoring dataset.

Dataset: BNCI 2015-013 (Chavarriaga & Millan, 2010, IEEE TNSRE 18(4)).
6 subjects watched a cursor move left/right towards a target; on 20-40% of
moves it went the WRONG way. The brain's "oops" response to those moves is
the error-related potential.

Each recording has a stimulus channel 'STI': 1 = correct move, 2 = error move.
MOABB downloads the files automatically (CC-BY-NC-ND 4.0, non-commercial).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

EVENT_ID = {"correct": 1, "error": 2}


def epochs_from_raw(raw, l_freq=1.0, h_freq=10.0, tmin=-0.2, tmax=0.8, decim=4):
    """Band-pass filter one run and cut epochs around every cursor move.

    1-10 Hz follows the original study. decim=4 turns 512 Hz into 128 Hz.
    Returns an mne.Epochs object (EEG channels only).
    """
    import mne

    raw = raw.copy().load_data()
    events = mne.find_events(raw, stim_channel="STI", shortest_event=1, verbose=False)
    raw.filter(l_freq, h_freq, picks="eeg", verbose=False)
    present = {k: v for k, v in EVENT_ID.items() if v in set(events[:, 2])}
    return mne.Epochs(raw, events, event_id=present, tmin=tmin, tmax=tmax,
                      baseline=(tmin, 0), picks="eeg", decim=decim,
                      preload=True, verbose=False)


def load_subject(subject: int, cache_dir: str | Path = "cache"):
    """Download (first time) and epoch all runs of one subject.

    Returns X (n_trials, n_channels, n_times) float32, y (1 = error, 0 = correct),
    run index per trial, channel names and the time axis in seconds.
    Results are cached as .npz so later runs are fast.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"errp_S{subject:02d}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=False)
        return d["X"], d["y"], d["run"], list(d["ch_names"]), d["times"]

    import mne
    from moabb.datasets import BNCI2015_013

    mne.set_log_level("ERROR")
    data = BNCI2015_013().get_data(subjects=[subject])[subject]
    Xs, ys, runs = [], [], []
    run_idx, ch_names, times = 0, None, None
    for session in sorted(data, key=lambda s: int(s) if str(s).isdigit() else s):
        for _, raw in sorted(data[session].items()):
            ep = epochs_from_raw(raw)
            if len(ep) == 0:
                continue
            Xs.append(ep.get_data(copy=False).astype(np.float32))
            ys.append((ep.events[:, 2] == EVENT_ID["error"]).astype(int))
            runs.append(np.full(len(ep), run_idx))
            ch_names, times = ep.ch_names, ep.times
            run_idx += 1
    X, y, run = np.concatenate(Xs), np.concatenate(ys), np.concatenate(runs)
    np.savez_compressed(f, X=X, y=y, run=run, ch_names=np.array(ch_names), times=times)
    return X, y, run, ch_names, times
