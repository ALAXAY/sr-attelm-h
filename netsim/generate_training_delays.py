# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Delay realizations of the preliminary stage: seeds 101 (split A) and 1101 (split B), levels 1, 2, 4.
Output: delays_network_train.npz (keys delay_<level>_<seed>), about 1 min on one CPU core."""
import numpy as np
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import generate_delays as gd
import netsim as ns

if __name__ == '__main__':
    TR = {}
    for seed in (101, 1101):
        for lvl in ('level1', 'level2', 'level4'):
            r = ns.simulate(gd.HOURS, seed, (lambda t, l=lvl: l), gd.PARAMS, tail_h=24 * 60)
            TR[f'delay_{lvl}_{seed}'] = r['delay'].astype(np.float32)
            print(seed, lvl, round(float(np.nanmean(r['delay'][..., 0])), 2), r['undelivered'], flush=True)
    np.savez_compressed('delays_network_train.npz', **TR)
