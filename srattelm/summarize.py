# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Pooled RMSE over the six runs and paired run-level comparisons; writes summary_network.csv, paired_network.csv."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABEL = {('LOCF', None): 'LOCF', ('no_attention', 'network'): 'Sector recurrent ELM without attention',
         ('proposed_noav', 'network'): 'Technique without availability parameters',
         ('proposed', 'network'): 'Proposed technique',
         ('proposed', 'geometric'): 'Proposed technique',
         ('proposed_noav', 'geometric'): 'Technique without availability parameters',
         ('no_attention', 'geometric'): 'Sector recurrent ELM without attention'}


def load():
    g = pd.read_csv(HERE / 'results_geometric.csv'); g['train'] = 'geometric'; g['delays'] = 'geometric'
    n = pd.read_csv(HERE / 'results_network.csv')
    a = pd.read_csv(HERE / 'results_official_agg.csv')
    a = a.assign(method='official_agg', train=a['delays'])
    rows = []
    for df in (g, n, a):
        for _, r in df.iterrows():
            m, tr, te = r['method'], r['train'], r['delays']
            if m == 'LOCF':
                lab = 'LOCF'
            elif m == 'official_agg':
                lab = 'Official AGG core'
            elif m == 'proposed' and te == 'network' and tr == 'geometric':
                lab = 'Proposed technique, trained on geometric delays'
            elif m == 'proposed_noav' and te == 'network' and tr == 'geometric':
                lab = 'Technique without availability parameters, trained on geometric delays'
            else:
                lab = LABEL.get((m, tr), m)
            rows.append(dict(label=lab, test=te, fold=r['fold'], seed=int(r['seed']), RMSE=r['RMSE'], MAE=r['MAE'],
                             n=int(r['n'])))
    return pd.DataFrame(rows).drop_duplicates(['label', 'test', 'fold', 'seed'])


def pooled(df):
    out = []
    for (lab, te), d in df.groupby(['label', 'test']):
        w = d['n'].to_numpy()
        out.append(dict(label=lab, test=te, runs=len(d), n=int(w.sum()),
                        RMSE=float(np.sqrt(np.sum(w * d['RMSE'] ** 2) / w.sum())),
                        MAE=float(np.sum(w * d['MAE']) / w.sum()),
                        RMSE_A=float(np.sqrt(np.sum((w * d['RMSE'] ** 2)[d.fold == 'A']) / w[d.fold == 'A'].sum())),
                        RMSE_B=float(np.sqrt(np.sum((w * d['RMSE'] ** 2)[d.fold == 'B']) / w[d.fold == 'B'].sum()))))
    return pd.DataFrame(out)


def paired(df, test='network'):
    d = df[df.test == test].pivot_table(index=['fold', 'seed'], columns='label', values='RMSE')
    rows = []
    for a in d.columns:
        for b in d.columns:
            if a == b:
                continue
            diff = d[a] - d[b]
            rows.append(dict(a=a, b=b, wins=int((diff < 0).sum()), mean_diff=float(diff.mean()),
                             mean_rel_reduction_pct=float(100 * (1 - d[a] / d[b]).mean())))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    df = load()
    S = pooled(df); S.to_csv(HERE / 'summary_network.csv', index=False)
    R = paired(df); R.to_csv(HERE / 'paired_network.csv', index=False)
    print(S.round(4).to_string())
