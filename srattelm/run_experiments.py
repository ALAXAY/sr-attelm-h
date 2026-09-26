# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Runs the reconstructed technique, its variants and LOCF under geometric and network-derived delays.

Usage:  python run_experiments.py            (all conditions, about 6 min on 8 CPU threads; SRATTELM_THREADS)
Outputs: results_geometric.csv, results_network.csv in this folder.
Requires: protocol.py, technique.py; for network delays ../netsim/delays_network.npz and
          ../netsim/delays_network_train.npz (training seeds 101 / 1101, levels 1, 2, 4).
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as PR
import technique as TQ

torch.set_num_threads(int(os.environ.get('SRATTELM_THREADS', '8')))
X, STATIONS = PR.load_raw()
NET = {}
for name in ('delays_network.npz', 'delays_network_train.npz'):
    p = HERE.parent / 'netsim' / name
    if p.exists():
        with np.load(p) as z:
            NET.update({k: z[k] for k in z.files if k.startswith('delay_')})


def views_geo(f):
    mean, std = PR.train_stats(X, f)
    trv = [PR.make_view(X, PR.geometric_delays(PR.TRAIN_SEED[f], 'stationary', mult=m), mean, std) for m in (1, 2, 4)]
    tev = [PR.make_view(X, PR.geometric_delays(s, 'shifted', f), mean, std) for s in PR.SEEDS]
    return trv, tev


def views_net(f):
    mean, std = PR.train_stats(X, f)
    trv = [PR.make_view(X, PR.network_delays(NET, PR.TRAIN_SEED[f], lv), mean, std) for lv in ('level1', 'level2', 'level4')]
    tev = [PR.make_view(X, PR.network_delays(NET, s, 'shifted', f), mean, std) for s in PR.SEEDS]
    return trv, tev


def locf_rows(tev, f, tag):
    bva, bte = PR.bidx(PR.FOLDS[f]['validation_end']), PR.bidx(PR.FOLDS[f]['test_end'])
    return [dict(delays=tag, train=tag, fold=f, seed=s, method='LOCF', **PR.metrics(v['locf'], v, PR.omega(v, bva, bte)))
            for s, v in zip(PR.SEEDS, tev)]


def technique_rows(trv, tev, f, variant, delays, train):
    t0 = time.time()
    out, info = TQ.fit_and_run(trv, trv[0], tev, f, variant=variant, seed=0)
    return [dict(delays=delays, train=train, fold=f, seed=s, method=variant, lam=info['lam'], beta=str(info['beta']),
                 train_s=round(time.time() - t0, 1), **o) for s, o in zip(PR.SEEDS, out)]


if __name__ == '__main__':
    geo, net = [], []
    for f in 'AB':
        trg, teg = views_geo(f)
        geo += locf_rows(teg, f, 'geometric')
        for var in ('no_attention', 'spatial_context', 'st_context', 'proposed_noav', 'proposed'):
            geo += technique_rows(trg, teg, f, var, 'geometric', 'geometric')
        if NET:
            trn, ten = views_net(f)
            net += locf_rows(ten, f, 'network')
            for var in ('no_attention', 'proposed_noav', 'proposed'):
                net += technique_rows(trn, ten, f, var, 'network', 'network')
            for var in ('proposed_noav', 'proposed'):
                net += technique_rows(trg, ten, f, var, 'network', 'geometric')
    pd.DataFrame(geo).to_csv(HERE / 'results_geometric.csv', index=False)
    if net:
        pd.DataFrame(net).to_csv(HERE / 'results_network.csv', index=False)
    print(pd.DataFrame(geo + net).groupby(['delays', 'train', 'method'])['RMSE'].mean().round(4))
