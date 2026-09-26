# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
# Figure: network model and comparison of network-derived delays with the geometric delay model.
# Requires in the namespace: apply_figure_style, STORE (delays_network.npz content), ad (analyze_delays), ns.
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

LABELS = {
    'en': dict(a='(a)', b='(b)', c='(c)', d='(d)', east='East of fusion node, km', north='North of fusion node, km',
               sink='fusion node', uav='UAV ferry tour', delay='Pollutant bundle delay, h',
               ccdf='Share of bundles with larger delay', net_nom='Network model, nominal',
               net_deg='Network model, degraded', geo_st='Geometric model, stationary',
               geo_sh='Geometric model, shifted ×4', daily='Daily mean delay, h', lag='Lag, h',
               acf='Autocorrelation of delay', shift='start of test interval', net_l2='Network model, stress level 2',
               net_l4='Network model, stress level 4', band='95% band for uncorrelated delays'),
    'uk': dict(a='а', b='б', c='в', d='г', east='Відстань на схід від вузла злиття, км',
               north='Відстань на північ від вузла злиття, км', sink='вузол злиття', uav='маршрут БпЛА',
               delay='Затримка пакета забруднювачів, год', ccdf='Частка пакетів з більшою затримкою',
               net_nom='Модель мережі, номінальний режим', net_deg='Модель мережі, режим деградації',
               geo_st='Геометрична модель, стаціонарний режим', geo_sh='Геометрична модель, середнє ×4',
               daily='Середньодобова затримка, год', lag='Зсув, год', acf='Автокореляція затримки',
               shift='початок тестового інтервалу', net_l2='Модель мережі, рівень напруженості 2',
               net_l4='Модель мережі, рівень напруженості 4', band='95 %-на смуга для некорельованих затримок'),
}
SHORT = ['Aot', 'Cha', 'Din', 'Don', 'Gua', 'Guc', 'Hua', 'Non', 'Shu', 'Tia', 'Wli', 'Wsx']


def _fmt(lang):
    if lang == 'uk':
        return mpl.ticker.FuncFormatter(lambda v, _p: f'{v:g}'.replace('.', ',').replace('-', '\u2212'))
    return mpl.ticker.FuncFormatter(lambda v, _p: f'{v:g}'.replace('-', '\u2212'))


STY = {'net_nom': ('black', '-', 1.0), 'net_l2': ('black', (0, (5, 2)), 1.0), 'net_l4': ('black', (0, (1, 1.5)), 1.3),
       'net_deg': ('black', '-', 2.0), 'geo_st': ('#7a7a7a', '--', 1.0), 'geo_sh': ('#7a7a7a', '--', 2.0)}


def draw(path, STORE, ad, ns, lang='en', fold='B', seed_ts=11):
    L = LABELS[lang]
    topo = ns.topology(30.0)
    pos, sink = topo['pos'], topo['sink']
    xy = np.stack([(pos[:, 1] - pos[sink, 1]) * 111.2 * np.cos(np.deg2rad(pos[sink, 0])),
                   (pos[:, 0] - pos[sink, 0]) * 111.2], 1)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    # (a) topology
    ax = axs[0, 0]
    nom = dict(ns.NOMINAL, down_ref_h=4.0, up_mean_h=8.0)
    for i, j, d in topo['links']:
        A = nom['up_mean_h'] / (nom['up_mean_h'] + nom['down_ref_h'] * max(d, 1) / 10)
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color='#7a7a7a', lw=0.3 + 1.6 * (A - 0.4), zorder=1)
    legs, _ = ns.uav_tour(topo)
    tour = [sink] + [v for v, _ in legs] + [sink]
    ax.plot(xy[tour, 0], xy[tour, 1], ls=(0, (4, 2)), color='black', lw=1.0, zorder=2, label=L['uav'])
    ax.scatter(xy[:-1, 0], xy[:-1, 1], s=22, facecolor='white', edgecolor='black', lw=0.9, zorder=3)
    ax.scatter(xy[sink, 0], xy[sink, 1], s=40, marker='s', color='black', zorder=4, label=L['sink'])
    OFF = {'Gua': (-17, 5), 'Don': (-3, 6), 'Tia': (4, -9), 'Wsx': (-21, -4), 'Non': (5, -3), 'Aot': (3, 4),
           'Wli': (-4, 5), 'Guc': (-4, 5)}
    for k in range(12):
        ax.annotate(SHORT[k], xy[k], xytext=OFF.get(SHORT[k], (3, 3)), textcoords='offset points', fontsize=7)
    ax.set_xlabel(L['east']); ax.set_ylabel(L['north'])
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='lower right', fontsize=7)
    ax.set_title(L['a'], loc='left', fontsize=9)
    # (b) CCDF on the test interval of the fold, pooled over seeds
    ax = axs[0, 1]
    mt = ad.mask('test', fold)
    series = [
        (np.concatenate([STORE[f'delay_level1_{s}'][mt, :, 0].ravel() for s in ad.SEEDS]), *STY['net_nom'], L['net_nom']),
        (np.concatenate([STORE[f'delay_shifted_{fold}_{s}'][mt, :, 0].ravel() for s in ad.SEEDS]), *STY['net_deg'], L['net_deg']),
        (np.concatenate([ad.geometric(s, 'stationary')[mt, :, 0].ravel() for s in ad.SEEDS]), '#7a7a7a', '--', 1.0, L['geo_st']),
        (np.concatenate([ad.geometric(s, 'shifted', fold)[mt, :, 0].ravel() for s in ad.SEEDS]), '#7a7a7a', '--', 2.0, L['geo_sh']),
    ]
    grid = np.arange(0, 200.5, 0.5)
    for v, c, ls, lw, lab in series:
        v = np.sort(v[np.isfinite(v)])
        cc = 1 - np.searchsorted(v, grid, side='right') / len(v)
        ax.semilogy(grid, np.maximum(cc, 1e-5), color=c, ls=ls, lw=lw, label=lab)
    ax.set_xlim(0, 200); ax.set_ylim(1e-4, 1.2)
    ax.set_xlabel(L['delay']); ax.set_ylabel(L['ccdf'])
    ax.set_title(L['b'], loc='left', fontsize=9)
    # (c) daily mean delay around the regime change (one seed)
    ax = axs[1, 0]
    idx = ad.INDEX
    win = (idx >= pd.Timestamp('2016-01-01')) & (idx < pd.Timestamp('2017-03-01'))
    dn = pd.Series(np.nanmean(STORE[f'delay_shifted_{fold}_{seed_ts}'][:, :, 0], 1), index=idx)[win].resample('D').mean()
    dg = pd.Series(ad.geometric(seed_ts, 'shifted', fold)[:, :, 0].mean(1), index=idx)[win].resample('D').mean()
    ax.plot(dg.index, dg.values, color='#7a7a7a', lw=1.0, ls='--')
    ax.plot(dn.index, dn.values, color='black', lw=1.0)
    ax.axvline(pd.Timestamp(ad.VAL_END[fold]), color='black', lw=0.6, ls=':')
    ax.set_ylabel(L['daily'])
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%m.%Y'))
    ax.tick_params(axis='x', labelsize=7)
    ax.set_title(L['c'], loc='left', fontsize=9)
    # (d) autocorrelation of hourly delays: all network regimes and both geometric regimes, split `fold`
    ax = axs[1, 1]
    mtr, mte = ad.mask('train', fold), ad.mask('test', fold)
    lags = np.arange(0, 73)

    def acf(d, m):
        out = []
        for s in range(12):
            y = d[m, s].astype(float); y = y - np.nanmean(y)
            den = np.nansum(y * y)
            out.append([1.0 if k == 0 else np.nansum(y[:-k] * y[k:]) / den for k in lags])
        return np.mean(out, 0)
    SER = [('net_nom', lambda s: STORE[f'delay_level1_{s}'][:, :, 0], mtr),
           ('net_l2', lambda s: STORE[f'delay_level2_{s}'][:, :, 0], mtr),
           ('net_l4', lambda s: STORE[f'delay_level4_{s}'][:, :, 0], mtr),
           ('net_deg', lambda s: STORE[f'delay_shifted_{fold}_{s}'][:, :, 0], mte),
           ('geo_st', lambda s: ad.geometric(s, 'stationary')[:, :, 0], mtr),
           ('geo_sh', lambda s: ad.geometric(s, 'shifted', fold)[:, :, 0], mte)]
    band = 1.96 / np.sqrt(mte.sum())
    ax.axhspan(-band, band, color='#d9d9d9', lw=0, zorder=0)
    res = {}
    for key, fun, m in SER:
        r = np.mean([acf(fun(s), m) for s in ad.SEEDS], 0)
        res[key] = r
        c, ls, lw = STY[key]
        ax.plot(lags, r, color=c, ls=ls, lw=lw, zorder=2)
    ax.set_xlim(0, 72); ax.set_ylim(-0.1, 1.05); ax.set_xticks([0, 12, 24, 36, 48, 60, 72])
    ax.set_xlabel(L['lag']); ax.set_ylabel(L['acf'])
    ax.set_title(L['d'], loc='left', fontsize=9)
    for a in axs.ravel():
        if a is not axs[1, 0]:
            a.xaxis.set_major_formatter(_fmt(lang))
        if a is not axs[0, 1]:
            a.yaxis.set_major_formatter(_fmt(lang))
    keys = ['net_nom', 'net_l2', 'net_l4', 'net_deg', 'geo_st', 'geo_sh']
    handles = [mpl.lines.Line2D([], [], color=STY[k][0], ls=STY[k][1], lw=STY[k][2]) for k in keys]
    handles.append(mpl.patches.Patch(color='#d9d9d9'))
    fig.tight_layout(rect=(0, 0.08 if lang == 'en' else 0.1, 1, 1))
    fig.legend(handles, [L[k] for k in keys] + [L['band']], loc='lower center', ncol=3 if lang == 'en' else 2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path, dpi=600, bbox_inches='tight')
    fig.savefig(path[:-4] + '.pdf', bbox_inches='tight')
    fig.savefig(path[:-4] + '.tif', dpi=1000, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    return {k: [float(v[1]), float(v[24])] for k, v in res.items()}
