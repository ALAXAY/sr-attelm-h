# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Fig. 8 (network model, CCDF, daily mean delay, autocorrelation): out/Fig8_network_delays.* (EN) and
out/Рис_мережеві_затримки.* (UK)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_delays as ad
import netsim as ns
import fig_delays as fd

mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.linewidth': 0.6, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'savefig.dpi': 600})

if __name__ == '__main__':
    with np.load(HERE / 'delays_network.npz') as z:
        STORE = {k: z[k] for k in z.files}
    (HERE / 'out').mkdir(exist_ok=True)
    fd.draw(str(HERE / 'out' / 'Fig8_network_delays.png'), STORE, ad, ns, lang='en')
    fd.draw(str(HERE / 'out' / 'Рис_мережеві_затримки.png'), STORE, ad, ns, lang='uk')
