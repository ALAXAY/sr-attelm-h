# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Delay statistics (Tables 10, 11) and autocorrelation of hourly delays, network model versus geometric model.
Outputs in out/: delay_statistics_by_seed.csv, delay_statistics_summary.csv, delay_autocorrelation_by_seed.csv,
delay_autocorrelation_summary.csv."""
from pathlib import Path
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats as sst
import analyze_delays as ad

HERE = Path(__file__).resolve().parent
LAGS = np.arange(0, 73)
COLS = ['mean_h', 'median_h', 'p90_h', 'p99_h', 'share_gt24h', 'lag1_autocorr', 'cross_station_corr',
        'out_of_order_share', 'mean_aoi_h', 'pending_last_bin', 'uav_share']


def load_store(path=HERE / 'delays_network.npz'):
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k.startswith('delay_') or k.startswith('uav_')}


def statistics(STORE):
    rows = []
    for f in 'AB':
        mtr, mt = ad.mask('train', f), ad.mask('test', f)
        for seed in ad.SEEDS:
            cases = {
                ('geometric', 'stationary, training part'): (ad.geometric(seed, 'stationary')[..., 0], mtr),
                ('network', 'nominal, training part'): (STORE[f'delay_level1_{seed}'][..., 0].astype(float), mtr),
                ('network', 'level 2, training part'): (STORE[f'delay_level2_{seed}'][..., 0].astype(float), mtr),
                ('network', 'level 4, training part'): (STORE[f'delay_level4_{seed}'][..., 0].astype(float), mtr),
                ('geometric', 'shifted, test interval'): (ad.geometric(seed, 'shifted', f)[..., 0], mt),
                ('network', 'degraded, test interval'): (STORE[f'delay_shifted_{f}_{seed}'][..., 0].astype(float), mt),
            }
            for (model, regime), (d, m) in cases.items():
                st = ad.stats(d, m)
                st.update(model=model, regime=regime, fold=f, seed=seed)
                if model == 'network':
                    key = {'nominal, training part': 'level1', 'level 2, training part': 'level2',
                           'level 4, training part': 'level4', 'degraded, test interval': f'shifted_{f}'}[regime]
                    st['uav_share'] = float(STORE[f'uav_{key}_{seed}'][..., 0][m].mean())
                rows.append(st)
    S = pd.DataFrame(rows)
    return S, S.groupby(['fold', 'model', 'regime'], sort=False)[COLS].mean().round(3)


def acf_series(y, lags=LAGS):
    y = y - np.nanmean(y)
    den = np.nansum(y * y)
    return np.array([1.0 if k == 0 else np.nansum(y[:-k] * y[k:]) / den for k in lags])


def acf_block(d, m):
    """Per-station autocorrelation (12, len(LAGS)), Ljung-Box statistic and p-value for 24 lags."""
    A, Q, P = [], [], []
    for s in range(12):
        y = d[m, s].astype(float)
        r = acf_series(y)
        n = len(y)
        q = n * (n + 2) * np.sum(r[1:25] ** 2 / (n - np.arange(1, 25)))
        A.append(r); Q.append(q); P.append(sst.chi2.sf(q, 24))
    return np.array(A), np.array(Q), np.array(P)


def autocorrelation(STORE):
    cases = []
    for f in 'AB':
        mtr, mt = ad.mask('train', f), ad.mask('test', f)
        for seed in ad.SEEDS:
            for lab, d, m in [('geometric, stationary', ad.geometric(seed, 'stationary')[..., 0], mtr),
                              ('network, nominal', STORE[f'delay_level1_{seed}'][..., 0], mtr),
                              ('network, level 2', STORE[f'delay_level2_{seed}'][..., 0], mtr),
                              ('network, level 4', STORE[f'delay_level4_{seed}'][..., 0], mtr),
                              ('geometric, shifted ×4', ad.geometric(seed, 'shifted', f)[..., 0], mt),
                              ('network, degraded', STORE[f'delay_shifted_{f}_{seed}'][..., 0], mt)]:
                A, Q, P = acf_block(d, m)
                cases.append(dict(fold=f, seed=seed, series=lab, part='training' if m is mtr else 'test',
                                  n=int(m.sum()), acf=A.mean(0), Q24=float(np.median(Q)),
                                  p_median=float(np.median(P)), share_sig=float((P < 0.05).mean())))
    C = pd.DataFrame(cases)
    C['rho1'] = C.acf.apply(lambda r: r[1]); C['rho6'] = C.acf.apply(lambda r: r[6])
    C['rho24'] = C.acf.apply(lambda r: r[24]); C['tau_int_h'] = C.acf.apply(lambda r: 1 + 2 * np.sum(r[1:49]))
    C['band95'] = 1.96 / np.sqrt(C.n)
    g = C.groupby(['fold', 'part', 'series'], sort=False)
    summ = g[['rho1', 'rho6', 'rho24', 'tau_int_h', 'Q24', 'p_median', 'share_sig', 'band95']].mean()
    rng = g[['rho1', 'rho24', 'tau_int_h']].agg(['min', 'max'])
    rng.columns = [f'{a}_{b}' for a, b in rng.columns]
    return C.drop(columns='acf'), summ.join(rng)


if __name__ == '__main__':
    STORE = load_store()
    (HERE / 'out').mkdir(exist_ok=True)
    S, T = statistics(STORE)
    S.to_csv(HERE / 'out' / 'delay_statistics_by_seed.csv', index=False)
    T.to_csv(HERE / 'out' / 'delay_statistics_summary.csv')
    C, A = autocorrelation(STORE)
    C.to_csv(HERE / 'out' / 'delay_autocorrelation_by_seed.csv', index=False)
    A.to_csv(HERE / 'out' / 'delay_autocorrelation_summary.csv')
    print(T.to_string()); print(A.round(3).to_string())
