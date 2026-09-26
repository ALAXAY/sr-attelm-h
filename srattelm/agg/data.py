# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Oleksii Fesenko
"""Shim for air_quality_two_clocks/src/data.py: load_raw, protocol constants and delay models.
DELAY_MODE=geometric (default) or DELAY_MODE=network (delays_network*.npz of the network model)."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import protocol as PR

MODE = os.environ.get('DELAY_MODE', 'geometric')
P = {'folds': PR.FOLDS}
_NET = {}


def _net():
    if not _NET:
        base = Path(__file__).resolve().parents[2] / 'netsim'
        for name in ('delays_network.npz', 'delays_network_train.npz'):
            with np.load(base / name) as z:
                _NET.update({k: z[k] for k in z.files if k.startswith('delay_')})
    return _NET


def load_raw():
    X, stations = PR.load_raw()
    return X, PR.INDEX, None, None, None, stations, None


def delays(raw, index, seed, regime, fold):
    if MODE == 'network':
        d = PR.network_delays(_net(), seed, 'shifted' if regime == 'shifted' else 'level1', fold)
    else:
        d = PR.geometric_delays(seed, regime, fold)
    return np.stack([d, d], axis=2)


def delays_mult(raw, index, seed, mult):
    if MODE == 'network':
        d = PR.network_delays(_net(), seed, {1: 'level1', 2: 'level2', 4: 'level4'}[int(mult)])
    else:
        d = PR.geometric_delays(seed, 'stationary', mult=mult)
    return np.stack([d, d], axis=2)
