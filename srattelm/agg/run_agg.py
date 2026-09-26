# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Oleksii Fesenko
"""Official AGG core under the frozen two-clock protocol (Tables 12, 13 of the article).

Usage:  python run_agg.py [--modes geometric network] [--folds A B]
Output: ../results_official_agg.csv (rows of the computed modes and splits are replaced), meta_<mode>_<fold>.json.
About 20-25 min per split and delay model on 12 CPU threads (AGG_THREADS); only the CPU device was tested.
The rows for the geometric delays in the shipped results_official_agg.csv were computed before the empty-graph
guard of AGGTargets was added; the guard is triggered only when the previous 6-h interval has no received value,
which did not occur for the geometric delays (no NaN in training or evaluation).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main(modes, folds):
    rows_all = []
    for mode in modes:
        os.environ['DELAY_MODE'] = mode
        for name in ('data', 'benchmark_publication_methods', 'official_agg_same_protocol'):
            sys.modules.pop(name, None)
        import official_agg_same_protocol as OA
        for f in folds:
            t0 = time.time()
            rows, meta = OA.run_fold(f, partial=True)
            for r in rows:
                r.update(delays=mode, wall_s=round(time.time() - t0, 1))
            rows_all += rows
            (HERE / f'meta_{mode}_{f}.json').write_text(json.dumps(meta, indent=1, default=str), encoding='utf-8')
            print(mode, f, [round(r['RMSE'], 4) for r in rows], [r['n'] for r in rows], round(time.time() - t0, 1),
                  flush=True)
    out = HERE.parent / 'results_official_agg.csv'
    new = pd.DataFrame(rows_all)
    if out.exists():
        old = pd.read_csv(out)
        old = old[~(old.delays.isin(modes) & old.fold.isin(folds))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=['geometric', 'network'], choices=['geometric', 'network'])
    ap.add_argument('--folds', nargs='+', default=['A', 'B'], choices=['A', 'B'])
    a = ap.parse_args()
    main(a.modes, a.folds)
