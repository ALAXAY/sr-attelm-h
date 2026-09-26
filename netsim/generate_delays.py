# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Generates network-derived arrival delays for the 2013-03-01 ... 2017-02-28 hourly grid.

Regimes (network state, by physical time):
  level1  - nominal network (delay intensity 1, training/validation);
  level2  - stress level: slower link dynamics and longer outages (mean delay about 2x nominal, training);
  level4  - stress level: slower link dynamics and longer outages (mean delay about 4x nominal, training);
  shifted_A / shifted_B - nominal network until the end of the validation part of split A / B, then the
            degraded regime: area-wide interference episodes (all terrestrial links down) on top of the
            nominal outages; the UAV ferry keeps operating.
Output: delays_network.npz with float32 delays in hours, shape (35064, 12, 2):
        [..., 0] pollutant bundle, [..., 1] meteorological bundle; plus UAV flags and hop counts.
"""
import json
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import time
import numpy as np
import pandas as pd
import netsim as ns

INDEX = pd.date_range('2013-03-01', '2017-02-28 23:00', freq='h')
HOURS = len(INDEX)
VAL_END = {'A': '2015-07-01', 'B': '2016-07-01'}
SEEDS = [11, 23, 37]

NOM = dict(ns.NOMINAL, down_ref_h=4.0, up_mean_h=8.0)
PARAMS = {
    'level1': dict(NOM),
    'level2': dict(NOM, up_mean_h=16.0, down_scale=2.25),   # link-process time scale x2, longer outages
    'level4': dict(NOM, up_mean_h=32.0, down_scale=4.7),    # link-process time scale x4, longer outages
    'degraded': dict(NOM, jam_rate_per_h=1 / 41.0, jam_mean_h=12.0),
}


def run_all(out='delays_network.npz'):
    store, meta = {}, {'params': PARAMS, 'seeds': SEEDS, 'hours': HOURS, 'first': str(INDEX[0]),
                       'validation_end': VAL_END, 'stations': ns.STATIONS, 'runs': {}}
    for seed in SEEDS:
        runs = {'level1': (lambda t: 'level1'), 'level2': (lambda t: 'level2'), 'level4': (lambda t: 'level4')}
        for f, ve in VAL_END.items():
            h0 = int((pd.Timestamp(ve) - INDEX[0]) / pd.Timedelta(hours=1))
            runs[f'shifted_{f}'] = (lambda t, h0=h0: 'level1' if t < h0 else 'degraded')
        for name, reg in runs.items():
            t0 = time.time()
            r = ns.simulate(HOURS, seed, reg, PARAMS, tail_h=24 * 60)
            key = f'{name}_{seed}'
            store[f'delay_{key}'] = r['delay'].astype(np.float32)
            store[f'uav_{key}'] = r['via_uav']
            store[f'hops_{key}'] = r['hops']
            meta['runs'][key] = {'seconds': round(time.time() - t0, 1), 'undelivered': r['undelivered'],
                                 'mean_pollutant_h': float(np.nanmean(r['delay'][..., 0]))}
            print(key, meta['runs'][key], flush=True)
    np.savez_compressed(out, **store)
    with open(out.replace('.npz', '_meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)
    return store, meta


if __name__ == '__main__':
    run_all()
