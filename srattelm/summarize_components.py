# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Pooled RMSE and attention-weight entropy of all variants (Table 13) -> availability_check_summary.csv."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ORDER = ['LOCF', 'no_attention', 'spatial_pure', 'st_explicit', 'spatial_context', 'st_context', 'weights_noav',
         'proposed_noav_all', 'proposed_noav', 'weights_av_explicit', 'proposed']


def pooled(d):
    return float(np.sqrt((d.RMSE ** 2 * d.n).sum() / d.n.sum()))


def main():
    geo = pd.read_csv(HERE / 'results_geometric.csv').assign(train='geometric')
    net = pd.read_csv(HERE / 'results_network.csv')
    comp = pd.read_csv(HERE / 'results_component_check.csv')
    allr = pd.concat([geo, net[net.train == net.delays], comp], ignore_index=True)
    rows = []
    for dl in ('geometric', 'network'):
        d = allr[allr.delays == dl]
        for m in ORDER:
            x = d[d.method == m]
            if not len(x):
                continue
            ent = x.att_entropy.mean() / np.log(12) if x.att_entropy.notna().any() else np.nan
            mx = x.att_max.mean() if x.att_max.notna().any() else np.nan
            rows.append(dict(delays=dl, variant=m, runs=len(x), RMSE_pooled=round(pooled(x), 4),
                             RMSE_A=round(pooled(x[x.fold == 'A']), 4), RMSE_B=round(pooled(x[x.fold == 'B']), 4),
                             att_entropy_rel=round(ent, 3), att_max=round(mx, 3)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / 'availability_check_summary.csv', index=False)
    return out


if __name__ == '__main__':
    print(main().to_string())
