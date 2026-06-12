"""Per-rep engine: the frozen probe gives ONE fixed scores/loss_tensor on the
calibtest pool; each rep s in 0..99 draws a fresh random calib/test partition of the POOL indices
(seed s). The probe is never touched — only the calib/test boundary moves. This samples "the draw
of the calibration set", exactly what the PAC histogram (exp ①) quantifies over (Thm 4.7).
"""

import numpy as np


def rep_split(n_pool, seed, calib_frac=0.5):
    """Fresh random calib/test partition of the calibtest-pool indices for one rep."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_pool)
    rng.shuffle(idx)
    n_calib = int(round(n_pool * calib_frac))
    calib = np.sort(idx[:n_calib])
    test = np.sort(idx[n_calib:])
    return calib, test


def rep_seeds(n_reps, seed_start=0):
    return list(range(seed_start, seed_start + n_reps))
