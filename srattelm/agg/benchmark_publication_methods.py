# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Oleksii Fesenko
"""Shim that restores the interface of the lost benchmark_publication_methods.py from protocol.py."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import protocol as PR
import data as D


def bidx(index, date):
    return PR.bidx(date)


def delays_mult(raw, index, seed, mult):
    return D.delays_mult(raw, index, seed, mult)


def build_binned(raw, index, d, mean, std):
    d2 = d[..., 0] if d.ndim == 3 else d
    v = PR.build_binned(raw[..., :4], d2, mean[:4], std[:4])
    return v['qt'], v['qo'], v['ratio'], v['ratio'] >= .999


def fill_locf(qo):
    return PR.fill_locf(qo)
