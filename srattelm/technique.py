# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Reconstructed implementation of the technique (SR-AttELM-H) as described in the manuscript.

Stage 1  causal set O(T), availability parameters rho (5), tau (6), dt (7), local estimate q^L (8), deviation r (9)
Stage 2  availability-aware spatiotemporal attention (10)-(13): e_ij = Q_i.K_j/sqrt(d_a) + b_rho*rho_j
         - b_tau*tau_j - b_dt*dt_j, softmax over the 12 stations, context c_i = sum_j a_ij W_V q^L_j,
         attention estimate r_att = F_att(xi), xi = (f_i, c_i)
Stage 3  sector mapping (14)-(16): d = c - q^L, psi = atan2(u2.d, u1.d), two radial levels (median), S = 8
Stage 4  recurrent ELM hidden layer (17) with fixed random weights, spectral radius < 1, H = 20
Stage 5  sector correction (18) and estimate (19): q_hat = q^L + r_att + B_s^T h
Stage 6-7 verification queue and RLS refinement (20)-(23) of B_s, P_s of the active sector only, after all records
         of the bin have arrived
Preliminary stage: gradient training of the attention block on delay levels nu = 1, 2, 4 (24); regularized
         least-squares initialisation of B_s, P_s (25).
Variants for the component-exclusion study: 'no_attention', 'spatial_context', 'st_context', 'proposed',
'proposed_noav' (attention estimate used as the main correction, availability parameters excluded from (10));
'proposed_noav_all' (as 'proposed_noav', and the availability parameters are also excluded from the feature vector f);
'weights_noav' (availability parameters excluded from the source weights only: from (10) and from the inputs of Q, K);
'weights_av_explicit' (availability parameters enter the source weights only through the explicit terms of (10));
'spatial_pure', 'st_explicit' (as 'spatial_context', 'st_context', with the availability parameters excluded from
the inputs of Q, K, i.e., the second and third variants of experiment 2 without the implicit route).
"""
import math
import numpy as np
import torch
from torch import nn

import protocol as PR

H_DIM, S_SECT, D_ATT = 20, 8, 16
F_DIM = 11                                           # q^L(4), dq^L(4), rho, tau, dt


def features(v):
    """Causal features f (nb, 12, 11) and availability parameters from a view of protocol.make_view."""
    qL = v['locf']
    dq = np.vstack([np.zeros((1,) + qL.shape[1:]), np.diff(qL, axis=0)])
    rho = np.nan_to_num(v['ratio'], nan=0.0).mean(2)                  # share of received components (5)
    tau = np.log1p(v['gap'].mean(2) * PR.BIN) / np.log1p(24.0)        # information gap (6), scaled
    dt = np.log1p(v['mean_delay']) / np.log1p(24.0)                    # mean arrival delay (7), scaled
    f = np.concatenate([qL, dq, rho[..., None], tau[..., None], dt[..., None]], axis=2)
    return f.astype(np.float32), rho, tau, dt


class Attention(nn.Module):
    def __init__(self, use_availability=True, hidden=64, qk_availability=True):
        super().__init__()
        self.use_av = use_availability
        self.qk_av = qk_availability                                   # availability columns of f in Q, K
        self.WQ = nn.Linear(F_DIM, D_ATT, bias=False)
        self.WK = nn.Linear(F_DIM, D_ATT, bias=False)
        self.WV = nn.Linear(PR.P, PR.P, bias=False)
        self.beta = nn.Parameter(torch.zeros(3))
        self.F = nn.Sequential(nn.Linear(F_DIM + PR.P, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, PR.P))

    def forward(self, f):                                              # f: (B, 12, 11)
        fq = f if self.qk_av else torch.cat([f[..., :8], torch.zeros_like(f[..., 8:])], dim=-1)
        Q, K = self.WQ(fq), self.WK(fq)
        e = Q @ K.transpose(1, 2) / math.sqrt(D_ATT)                   # (B, 12, 12)
        if self.use_av:
            b = nn.functional.softplus(self.beta)
            av = b[0] * f[..., 8] - b[1] * f[..., 9] - b[2] * f[..., 10]   # rho, tau, dt of source j
            e = e + av[:, None, :]
        a = torch.softmax(e, dim=-1)
        c = a @ self.WV(f[..., :PR.P])                                  # context (12)
        r_att = self.F(torch.cat([f, c], dim=-1))                      # attention estimate (13)
        return r_att, c, a


def train_attention(train_views, val_views, lo_tr, hi_tr, lo_va, hi_va, use_availability=True, seed=0,
                    epochs=40, lr=2e-3, batch=64, fmean=None, fstd=None, qk_availability=True):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Attention(use_availability, qk_availability=qk_availability)

    def pack(views, lo, hi):
        F_, R_, M_ = [], [], []
        for v in views:
            f = (v['f'] - fmean) / fstd
            r = np.nan_to_num(v['qt'] - v['locf'])
            m = np.isfinite(v['qt']) & (v['ratio'] < .999)
            F_.append(f[max(1, lo):hi]); R_.append(r[max(1, lo):hi]); M_.append(m[max(1, lo):hi])
        return (torch.tensor(np.concatenate(F_)), torch.tensor(np.concatenate(R_), dtype=torch.float32),
                torch.tensor(np.concatenate(M_), dtype=torch.float32))
    Ftr, Rtr, Mtr = pack(train_views, lo_tr, hi_tr)
    Fva, Rva, Mva = pack(val_views, lo_va, hi_va)
    # keep the availability columns unscaled for the attention score
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float('inf'), None, 0
    n = len(Ftr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            r_att, _, _ = model(Ftr[idx])
            loss = (((r_att - Rtr[idx]) ** 2) * Mtr[idx]).sum() / Mtr[idx].sum().clamp_min(1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            r_att, _, _ = model(Fva)
            vl = float((((r_att - Rva) ** 2) * Mva).sum() / Mva.sum().clamp_min(1))
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best_state)
    return model, math.sqrt(best)


def attention_outputs(model, v, fmean, fstd):
    with torch.no_grad():
        f = torch.tensor((v['f'] - fmean) / fstd)
        r_att, c, a = model(f)
    return r_att.numpy().astype(float), c.numpy().astype(float), a.numpy().astype(float)


class Reservoir:
    """Recurrent ELM hidden layer (17) with fixed random weights."""

    def __init__(self, in_dim, seed=7, rho_max=0.8, H=H_DIM):
        g = np.random.default_rng(seed)
        self.Wx = g.uniform(-1, 1, (H, in_dim)) / math.sqrt(in_dim)
        Wr = g.uniform(-1, 1, (H, H))
        self.Wr = Wr * rho_max / max(abs(np.linalg.eigvals(Wr)))
        self.b = g.uniform(-0.5, 0.5, H)

    def run(self, xi):                                               # xi: (nb, 12, in_dim)
        nb, n, _ = xi.shape
        h = np.zeros((nb, n, self.Wx.shape[0])); prev = np.zeros((n, self.Wx.shape[0]))
        for t in range(nb):
            prev = np.tanh(xi[t] @ self.Wx.T + prev @ self.Wr.T + self.b)
            h[t] = prev
        return h


def sector_fit(d_train):
    X = d_train.reshape(-1, PR.P)
    C = np.cov(X.T)
    w, U = np.linalg.eigh(C)
    u1, u2 = U[:, -1], U[:, -2]
    r0 = float(np.median(np.linalg.norm(X, axis=1)))
    return u1, u2, r0


def sector_map(d, u1, u2, r0):
    psi = np.arctan2(d @ u2, d @ u1)                                  # (15)
    l = 1 + (np.linalg.norm(d, axis=-1) > r0)
    q = np.minimum(np.floor(2 * (psi + np.pi) / np.pi), 3)
    return (4 * (l - 1) + q).astype(int)                             # 0..7 (16)


def init_sectors(Hs, Es, Ms, sect, gamma=1.0):
    B = np.zeros((S_SECT, H_DIM, PR.P)); Pm = np.zeros((S_SECT, H_DIM, H_DIM))
    for s in range(S_SECT):
        sel = (sect == s) & Ms
        Hm, Em = Hs[sel], Es[sel]
        Pm[s] = np.linalg.inv(Hm.T @ Hm + gamma * np.eye(H_DIM))
        B[s] = Pm[s] @ Hm.T @ Em                                      # (25)
    return B, Pm


def rls_update(B, Pm, s, h, eps, lam):
    Ph = Pm[s] @ h
    k = Ph / (lam + h @ Ph)                                           # (21)
    B[s] = B[s] + np.outer(k, eps - B[s].T @ h)                       # (22)
    Pm[s] = (Pm[s] - np.outer(k, h @ Pm[s])) / lam                    # (23)


def stream(v, lo, hi, h, r_att, sect, B, Pm, lam, use_att=True, online=True):
    """Streaming estimation and refinement over bins [lo, hi); returns q_hat (nb, 12, 4)."""
    B, Pm = B.copy(), Pm.copy()
    q_hat = np.array(v['locf'], dtype=float)
    queue = []                                                       # (completion hour, t, i)
    target = v['qt'] - v['locf'] - (r_att if use_att else 0.0)        # (20)
    delayed = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
    for t in range(max(1, lo), hi):
        T = (t + 1) * PR.BIN
        if online:                                                   # refine with every completed entry
            keep = []
            for (ca, tt, i) in queue:
                if ca <= T:
                    rls_update(B, Pm, sect[tt, i], h[tt, i], target[tt, i], lam)
                else:
                    keep.append((ca, tt, i))
            queue = keep
        for i in range(PR.N):
            s = sect[t, i]
            q_hat[t, i] = v['locf'][t, i] + (r_att[t, i] if use_att else 0.0) + B[s].T @ h[t, i]   # (18), (19)
            if online and delayed[t, i]:
                queue.append((v['complete_at'][t, i], t, i))
    return q_hat


def fit_and_run(train_views, val_view, test_views, fold, variant='proposed', lam=0.995, seed=0, gamma=1.0,
                select_lambda=True, verbose=False):
    """Preliminary stage on train_views (levels 1, 2, 4) + validation view, then streaming on test views."""
    btr, bva, bte = (PR.bidx(PR.FOLDS[fold][k]) for k in ('train_end', 'validation_end', 'test_end'))
    for v in train_views + [val_view] + test_views:
        v['f'], v['rho'], v['tau'], v['dt'] = features(v)
        if variant == 'proposed_noav_all':                             # availability parameters excluded from f too
            v['f'][..., 8:] = 0.0
    allf = np.concatenate([v['f'][1:btr].reshape(-1, F_DIM) for v in train_views])
    fmean, fstd = allf.mean(0), allf.std(0) + 1e-6
    fmean[8:], fstd[8:] = 0.0, 1.0                                     # availability parameters kept unscaled
    use_av = variant in ('st_context', 'proposed', 'weights_av_explicit', 'st_explicit')
    use_ctx = variant != 'no_attention'
    use_att = variant in ('proposed', 'proposed_noav', 'proposed_noav_all', 'weights_noav', 'weights_av_explicit')
    qk_av = variant not in ('weights_noav', 'weights_av_explicit', 'spatial_pure', 'st_explicit')
    model, val_rmse = (train_attention(train_views, [val_view], 1, btr, btr, bva, use_availability=use_av,
                                       seed=seed, fmean=fmean, fstd=fstd, qk_availability=qk_av)
                       if use_ctx else (None, float('nan')))
    res = Reservoir(F_DIM + (PR.P if use_ctx else 0), seed=seed + 7)

    def prep(v):
        f = (v['f'] - fmean) / fstd
        if use_ctx:
            r_att, c, a = attention_outputs(model, v, fmean, fstd)
            xi = np.concatenate([f, c], axis=2); d = c - v['locf']
        else:
            r_att = np.zeros_like(v['locf']); xi = f; d = np.concatenate([f[..., :PR.P]], axis=2); a = None
        return xi, r_att, d, a
    P_train = [prep(v) for v in train_views]
    u1, u2, r0 = sector_fit(np.concatenate([p[2][1:btr] for p in P_train]))
    Hs, Es, Ms, Ss = [], [], [], []
    for v, (xi, r_att, d, _) in zip(train_views, P_train):
        h = res.run(xi); sect = sector_map(d, u1, u2, r0)
        tgt = v['qt'] - v['locf'] - (r_att if use_att else 0.0)
        m = np.isfinite(v['qt']).all(2) & (np.nan_to_num(v['ratio'], nan=1.0) < .999).any(2)
        Hs.append(h[1:btr].reshape(-1, H_DIM)); Es.append(np.nan_to_num(tgt[1:btr]).reshape(-1, PR.P))
        Ms.append(m[1:btr].reshape(-1)); Ss.append(sect[1:btr].reshape(-1))
    B0, P0 = init_sectors(np.concatenate(Hs), np.concatenate(Es), np.concatenate(Ms), np.concatenate(Ss), gamma)

    def run_view(v, lo, hi, lam_):
        xi, r_att, d, a = prep(v)
        h = res.run(xi); sect = sector_map(d, u1, u2, r0)
        return stream(v, lo, hi, h, r_att, sect, B0, P0, lam_, use_att=use_att), a, sect
    if select_lambda:                                                # forgetting factor from the validation part
        best = None
        for lam_ in (0.98, 0.99, 0.995, 0.999, 1.0):
            qh, _, _ = run_view(val_view, btr, bva, lam_)
            sc = PR.metrics(qh, val_view, PR.omega(val_view, btr, bva))['RMSE']
            if best is None or sc < best[0]:
                best = (sc, lam_)
        lam = best[1]
    out = []
    for v in test_views:
        qh, a, sect = run_view(v, bva, bte, lam)
        m = PR.omega(v, bva, bte)
        met = PR.metrics(qh, v, m)
        if a is not None:
            aw = a[m.any(2)]
            ent = -(aw * np.log(np.clip(aw, 1e-12, 1))).sum(-1)
            met.update(att_entropy=float(ent.mean()), att_max=float(aw.max(-1).mean()))
        out.append(met)
    return out, dict(lam=lam, val_rmse_attention=val_rmse, beta=(nn.functional.softplus(model.beta).tolist()
                                                                     if model is not None else None))
