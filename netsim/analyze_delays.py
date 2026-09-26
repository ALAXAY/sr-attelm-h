# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Statistics of network-derived delays versus the geometric delay model of data.py (same seeds)."""
import numpy as np
import pandas as pd

INDEX = pd.date_range('2013-03-01', '2017-02-28 23:00', freq='h')
HOURS = len(INDEX)
VAL_END = {'A': '2015-07-01', 'B': '2016-07-01'}
TEST_END = {'A': '2016-01-01', 'B': '2017-03-01'}
TRAIN_END = {'A': '2015-01-01', 'B': '2016-01-01'}
SEEDS = [11, 23, 37]


def geometric(seed, regime, fold=None, hours=HOURS):
    """Exact replica of delays() in air_quality_two_clocks/src/data.py (float, whole hours)."""
    shape = (hours, 12, 2)
    rng = np.random.default_rng(seed + 8100)
    u = np.clip(rng.random(shape), 1e-12, 1 - 1e-12)
    mean = np.asarray([6.0, 2.0])[None, None, :] * np.linspace(.75, 1.25, 12)[None, :, None]
    mean = np.broadcast_to(mean, shape).copy()
    if regime == 'shifted':
        after = INDEX[:hours] >= pd.Timestamp(VAL_END[fold])
        mean[after] *= 4
    return np.floor(np.log1p(-u) / np.log(mean / (mean + 1))).astype(float)


def mask(kind, fold):
    if kind == 'test':
        return (INDEX >= pd.Timestamp(VAL_END[fold])) & (INDEX < pd.Timestamp(TEST_END[fold]))
    if kind == 'train':
        return INDEX < pd.Timestamp(TRAIN_END[fold])
    if kind == 'all':
        return np.ones(HOURS, bool)
    raise ValueError(kind)


def aoi_and_pending(d, m, step=6):
    """Mean age of information at decision times T = 0, 6, 12, ... h inside mask m, and the mean share of the
    latest 6-h bin not yet received at T (record of hour h is available at T if h + floor(d) < T)."""
    hours = np.arange(d.shape[0])
    Ts = hours[m][::step]
    Ts = Ts[Ts >= 24]
    aoi, pend = [], []
    for s in range(d.shape[1]):
        a = hours + np.floor(np.nan_to_num(d[:, s], nan=1e6))
        order = np.argsort(a, kind='stable')
        a_sorted = a[order]
        latest = np.maximum.accumulate(hours[order])
        idx = np.searchsorted(a_sorted, Ts, side='left') - 1          # arrivals strictly before T
        lat = np.where(idx >= 0, latest[np.clip(idx, 0, None)], -10 ** 6)
        aoi.append(Ts - lat)
        last_bin = np.stack([a[T - step:T] >= T for T in Ts])
        pend.append(last_bin.mean(1))
    return float(np.mean(aoi)), float(np.mean(pend))


def stats(d, m):
    """d: (hours, 12) pollutant delays in hours; m: boolean mask of generation hours."""
    x = d[m]
    v = x[np.isfinite(x)]
    ac = []
    for s in range(x.shape[1]):
        y = x[:, s]
        ok = np.isfinite(y[:-1]) & np.isfinite(y[1:])
        ac.append(np.corrcoef(y[:-1][ok], y[1:][ok])[0, 1])
    C = pd.DataFrame(x).corr().to_numpy()
    off = C[~np.eye(12, dtype=bool)]
    arr = np.arange(len(d))[:, None] + np.floor(np.nan_to_num(d, nan=1e6))
    am = arr[m]
    ooo = float((am[1:] < am[:-1]).mean())
    aoi, pend = aoi_and_pending(d, m)
    return {'mean_h': float(v.mean()), 'median_h': float(np.median(v)), 'p90_h': float(np.percentile(v, 90)),
            'p99_h': float(np.percentile(v, 99)), 'share_gt24h': float((v > 24).mean()),
            'lag1_autocorr': float(np.nanmean(ac)), 'cross_station_corr': float(np.nanmean(off)),
            'out_of_order_share': ooo, 'mean_aoi_h': aoi, 'pending_last_bin': pend}
