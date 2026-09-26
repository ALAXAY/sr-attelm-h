# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Reconstructed two-clock protocol of the SR-AttELM-H benchmark (Beijing Multi-Site Air Quality).

The reconstruction reproduces the frozen protocol exactly where it can be checked against the saved results:
  - the set Omega of delayed components: 31722 / 31718 / 31728 (split A, seeds 11/23/37) and
    42042 / 42044 / 42038 (split B), 221292 in total;
  - the LOCF RMSE of Table 6: 0.5089 / 0.5015 / 0.5166 / 0.5714 / 0.5749 / 0.5554.
Conventions (recovered): 6-h bins; decision time of bin t is the end of the bin, T_t = 6(t+1) h; an hourly record
generated at hour h with delay d is available at T if h + d < T; qt - bin mean of all finite records (z-scored by
training statistics); qo - bin mean of records available at T_t; ratio - share of finite records available at T_t;
Omega - components with finite qt and ratio < 0.999 in bins whose 48 truths are all finite (common support);
LOCF - forward fill of qo including the current partial bin, zero (training mean) before the first observation.
"""
import hashlib
import io
import os
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

# Data: Beijing Multi-Site Air Quality, UCI Machine Learning Repository, https://doi.org/10.24432/C5RK5G.
# The path is taken from the environment variable SRATTELM_DATA (a zip archive at any nesting level or a folder
# with the PRSA_Data_*.csv files); otherwise the first *.zip in <repository>/data is used.
DATA_ENV = 'SRATTELM_DATA'
UCI_URL = 'https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip'
LEGACY_ZIP = Path(r'D:\ЗАГР\Fesenko_Q2C_Research.zip')
ZIP = LEGACY_ZIP
DATA_SHA256 = '26b1943d781766805c8a841f9730aa5956f973946b25e1dbeed2738d6ae8e997'  # loaded array, NaN -> -9999
INDEX = pd.date_range('2013-03-01', '2017-02-28 23:00', freq='h')
FOLDS = {'A': dict(train_end='2015-01-01', validation_end='2015-07-01', test_end='2016-01-01'),
         'B': dict(train_end='2016-01-01', validation_end='2016-07-01', test_end='2017-03-01')}
SEEDS = (11, 23, 37)
TRAIN_SEED = {'A': 101, 'B': 1101}
CH = ['PM2.5', 'PM10', 'SO2', 'NO2']
N, P, BIN = 12, 4, 6


def _csv_items(zf, depth=0):
    """(file name, bytes) of the PRSA_Data_*.csv files of a zip archive, including nested zip archives."""
    for n in sorted(zf.namelist()):
        base = n.rsplit('/', 1)[-1]
        if base.startswith('PRSA_Data_') and base.endswith('.csv'):
            yield base, zf.read(n)
        elif n.endswith('.zip') and depth < 3:
            with zipfile.ZipFile(io.BytesIO(zf.read(n))) as inner:
                yield from _csv_items(inner, depth + 1)


def default_data_path():
    env = os.environ.get(DATA_ENV)
    root = Path(__file__).resolve().parents[1] / 'data'
    cands = ([Path(env)] if env else []) + (sorted(root.glob('*.zip')) if root.exists() else []) + [root, LEGACY_ZIP]
    for c in cands:
        if c.is_file() or (c.is_dir() and any(c.glob('**/PRSA_Data_*.csv'))):
            return c
    raise FileNotFoundError(f'Data set not found: set {DATA_ENV} or place the UCI archive ({UCI_URL}) in {root}')


def load_raw(zip_path=None, verify=True):
    """Hourly (35064, 12, 4) array of PM2.5, PM10, SO2, NO2; stations in alphabetical order."""
    path = Path(zip_path) if zip_path is not None else default_data_path()
    if path.is_dir():
        items = [(p.name, p.read_bytes()) for p in sorted(path.glob('**/PRSA_Data_*.csv'))]
    elif path == LEGACY_ZIP:
        with zipfile.ZipFile(path) as z:
            outer_bytes = z.read('air_quality_two_clocks/data/beijing_uci.zip')
        with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
            items = list(_csv_items(outer))
    else:
        with zipfile.ZipFile(path) as z:
            items = list(_csv_items(z))
    seen, frames = set(), []
    for base, b in items:
        if base not in seen:
            seen.add(base); frames.append(pd.read_csv(io.BytesIO(b)))
    if len(frames) != 12:
        raise ValueError(f'{path}: expected 12 station files, found {len(frames)}')
    stations, arrays = [], []
    for df in frames:
        df.index = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df = df.reindex(INDEX)
        stations.append(df.station.dropna().iloc[0]); arrays.append(df[CH].to_numpy(float))
    order = np.argsort(stations)
    X = np.stack(arrays, 1)[:, order]
    if verify and DATA_SHA256 is not None and data_sha256(X) != DATA_SHA256:
        raise ValueError('The loaded data differ from the data set used in the article (SHA-256 mismatch)')
    return X, [stations[i] for i in order]


def data_sha256(X):
    return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(X, nan=-9999.0), dtype='<f8').tobytes()).hexdigest()


def geometric_delays(seed, regime='stationary', fold=None, mult=1.0, hours=len(INDEX)):
    """Geometric delay model of air_quality_two_clocks/src/data.py (pollutant packet), whole hours, (T, 12)."""
    shape = (hours, N, 2)
    rng = np.random.default_rng(seed + 8100)
    u = np.clip(rng.random(shape), 1e-12, 1 - 1e-12)
    mean = np.asarray([6.0, 2.0])[None, None, :] * np.linspace(.75, 1.25, N)[None, :, None] * mult
    mean = np.broadcast_to(mean, shape).copy()
    if regime == 'shifted':
        mean[INDEX[:hours] >= pd.Timestamp(FOLDS[fold]['validation_end'])] *= 4
    return np.floor(np.log1p(-u) / np.log(mean / (mean + 1)))[:, :, 0]


def network_delays(store, seed, regime, fold=None):
    """Network-derived delays (whole hours) from delays_network.npz; regime: level1/level2/level4/shifted."""
    key = f'delay_shifted_{fold}_{seed}' if regime == 'shifted' else f'delay_{regime}_{seed}'
    return np.floor(store[key][:, :, 0].astype(float))


def bidx(date):
    return min(int((pd.Timestamp(date) - INDEX[0]) / pd.Timedelta(hours=1)) // BIN, len(INDEX) // BIN)


def train_stats(X, fold):
    tr = INDEX < pd.Timestamp(FOLDS[fold]['train_end'])
    return np.nanmean(X[tr], (0, 1)), np.maximum(np.nanstd(X[tr], (0, 1)), 1e-5)


def build_binned(X, d, mean, std):
    """Returns dict with qt, qo, ratio (nb, 12, 4), completion time of every bin (nb, 12) in hours and the mean
    arrival delay of records received within the last 24 h at every decision time (nb, 12)."""
    nb = len(X) // BIN
    x = ((X[:nb * BIN] - mean) / std).reshape(nb, BIN, N, P)
    hrs = np.arange(nb * BIN).reshape(nb, BIN)
    arr = hrs[:, :, None] + d[:nb * BIN].reshape(nb, BIN, N)
    T = (np.arange(nb) * BIN + BIN)[:, None, None]
    got = arr < T
    fin = np.isfinite(x)
    got4 = got[..., None] & fin
    cnt, cg = fin.sum(1), got4.sum(1)
    qt = np.where(cnt > 0, np.where(fin, x, 0).sum(1) / np.maximum(cnt, 1), np.nan)
    qo = np.where(cg > 0, np.where(got4, x, 0).sum(1) / np.maximum(cg, 1), np.nan)
    ratio = np.where(cnt > 0, cg / np.maximum(cnt, 1), np.nan)
    rec_fin = fin.any(3)                                                   # (nb, 6, 12) record has any value
    complete_at = np.where(rec_fin, arr + 1, 0).max(1)                     # hour after the last arrival
    # mean delay of records that arrived in the window [T-24, T): hourly arrivals
    delay_h = d[:nb * BIN].astype(float)
    arr_h = np.arange(nb * BIN)[:, None] + delay_h
    Td = np.arange(nb) * BIN + BIN
    mdel = np.zeros((nb, N))
    order = np.argsort(arr_h, axis=0, kind='stable')
    for s in range(N):
        a = arr_h[order[:, s], s]; dl = delay_h[order[:, s], s]
        cs = np.concatenate([[0.0], np.cumsum(dl)])
        hi = np.searchsorted(a, Td, side='left'); lo = np.searchsorted(a, Td - 24, side='left')
        n = hi - lo
        mdel[:, s] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    mdel = pd.DataFrame(mdel).ffill().fillna(0).to_numpy()
    return dict(qt=qt, qo=qo, ratio=ratio, complete_at=complete_at, mean_delay=mdel)


def fill_locf(qo):
    nb = qo.shape[0]
    out = np.zeros_like(qo); gap = np.zeros_like(qo)
    last = np.full(qo.shape[1:], np.nan); lg = np.zeros(qo.shape[1:])
    for t in range(nb):
        fin = np.isfinite(qo[t])
        last = np.where(fin, qo[t], last); lg = np.where(fin, 0, lg + 1)
        out[t] = np.nan_to_num(last, nan=0.0); gap[t] = lg
    return out, gap


def omega(v, lo, hi):
    """Boolean (nb, 12, 4) mask of the evaluated delayed components in bins [max(1, lo), hi)."""
    m = np.zeros(v['qt'].shape, bool)
    sl = slice(max(1, lo), hi)
    cs = np.isfinite(v['qt'][sl]).all((1, 2))
    m[sl] = np.isfinite(v['qt'][sl]) & (v['ratio'][sl] < .999) & cs[:, None, None]
    return m


def make_view(X, d, mean, std):
    v = build_binned(X, d, mean, std)
    v['locf'], v['gap'] = fill_locf(v['qo'])
    return v


def metrics(pred, v, m):
    e = (pred - v['qt'])[m]
    ep = (pred - v['qt'])[:, :, 0][m[:, :, 0]]
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                PM25_RMSE=float(np.sqrt(np.mean(ep ** 2))), n=int(m.sum()))
