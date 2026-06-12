"""RAND (floor; NOT judged PASS/FAIL).

Accept a RANDOM ξ-fraction (selector replaced by noise), then run the NM-SCRC second stage. Shows the
informed selector g carries information: an informed g should beat RAND (smaller accepted-region sets /
lower conditional risk at equal coverage). Same frozen Λ2 grid (so |C| is comparable); pinned at ξ.
"""

from dataclasses import replace

import numpy as np

from nmscrc.methods import nmscrc_i


def run(A, calib_idx, test_idx, xi, alpha, delta, seed, variant="eb"):
    rng = np.random.default_rng(10_000 + seed)               # independent of the rep's calib/test draw
    A_rand = replace(A, g=rng.random(A.n))                    # selector replaced by uniform noise
    out = nmscrc_i.run(A_rand, calib_idx, test_idx, xi, alpha, delta, variant=variant, lam1_mode="pinned")
    out["selector"] = "random"
    return out
