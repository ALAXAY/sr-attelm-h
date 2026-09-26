# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Component exclusion of the re-implementation with the availability parameters separated by route (Table 13).

Variants (see technique.py): spatial_pure, st_explicit - the second and third models of experiment 2 without the
availability parameters in the inputs of W_Q, W_K; weights_av_explicit - availability parameters only in the terms of
(10); weights_noav - availability parameters excluded from the source weights; proposed_noav_all - availability
parameters excluded from the feature vector f. Output: results_component_check.csv (about 5 min on 8 CPU threads).
"""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_experiments as RE

VARS = ('spatial_pure', 'st_explicit', 'weights_av_explicit', 'weights_noav', 'proposed_noav_all')

if __name__ == '__main__':
    rows = []
    for f in 'AB':
        trg, teg = RE.views_geo(f)
        trn, ten = RE.views_net(f)
        for var in VARS:
            rows += RE.technique_rows(trg, teg, f, var, 'geometric', 'geometric')
            rows += RE.technique_rows(trn, ten, f, var, 'network', 'network')
        print(f, 'done', flush=True)
    pd.DataFrame(rows).to_csv(HERE / 'results_component_check.csv', index=False)
