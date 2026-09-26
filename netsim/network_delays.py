# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Drop-in replacement for delays() of air_quality_two_clocks/src/data.py with network-derived delays.

Usage in the experiment code:
    from network_delays import delays            # instead of data.delays
    d = delays(raw, index, seed, regime, fold)     # int32 array (T, 12, 2), whole hours

regime: 'clean'      - no delays;
        'stationary' - nominal network (delay intensity 1);
        'level2', 'level4' - network stress levels used for multi-level training (nu = 2, 4);
        'shifted'    - nominal network until the end of the validation part of `fold`, degraded after it.
The integer delay is floor(arrival - generation), i.e. the record generated at hour h becomes available in
hour h + delay, as in the geometric model of data.py.
"""
from pathlib import Path
import numpy as np

_NPZ = Path(__file__).with_name('delays_network.npz')
_CACHE = {}
FIRST = np.datetime64('2013-03-01T00')


def _load():
    if not _CACHE:
        with np.load(_NPZ) as z:
            _CACHE.update({k: z[k] for k in z.files if k.startswith('delay_')})
    return _CACHE


def delays(raw, index, seed, regime, fold):
    shape = raw.shape[:2] + (2,)
    if regime == 'clean':
        return np.zeros(shape, dtype=np.int32)
    name = {'stationary': 'level1', 'level1': 'level1', 'level2': 'level2', 'level4': 'level4',
            'shifted': f'shifted_{fold}'}[regime]
    key = f'delay_{name}_{seed}'
    store = _load()
    if key not in store:
        raise KeyError(f'{key} is not in {_NPZ.name}; available seeds are 11, 23, 37')
    d = store[key]
    offset = int((np.datetime64(str(index[0])[:13]) - FIRST) / np.timedelta64(1, 'h'))
    d = d[offset:offset + shape[0]]
    if d.shape != shape:
        raise ValueError(f'delay grid {d.shape} does not match data grid {shape}')
    return np.floor(np.nan_to_num(d, nan=1e6)).astype(np.int32)
