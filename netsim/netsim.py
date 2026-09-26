# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Oleksii Fesenko
"""Packet-level model of message delivery in an intermittently connected monitoring network.

Twelve air-quality stations (Beijing Multi-Site Air Quality data set, alphabetical order) and a fusion node
form a terrestrial ad hoc network with intermittent links; a UAV data ferry periodically collects the bundles
buffered at multi-hop stations. Every station generates two bundles per hour: a pollutant bundle (class 0,
normal priority) and a meteorological bundle (class 1, high priority). Nodes store and forward bundles
(custody transfer, no loss). The model returns the arrival time of every bundle at the fusion node.

Time unit: hour; simulation step DT = 0.1 h. Station positions are approximate (about 1 km) and are used only
to derive inter-node distances.
"""
import collections
import heapq
import numpy as np

STATIONS = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou',
            'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']
LATLON = np.array([
    [39.982, 116.397], [40.217, 116.230], [40.292, 116.220], [39.929, 116.417], [39.929, 116.339],
    [39.914, 116.184], [40.328, 116.628], [39.937, 116.461], [40.127, 116.655], [39.886, 116.407],
    [39.987, 116.287], [39.878, 116.352]])
SINK_LATLON = np.array([39.930, 116.390])            # fusion node (city centre)
DT = 0.1

NOMINAL = dict(
    R_km=30.0,            # maximum terrestrial link distance, km
    up_mean_h=8.0,        # mean duration of the link "up" state, h
    down_ref_h=4.0,       # mean "down" duration of a 10 km link, h (grows linearly with distance)
    down_scale=1.0,       # outage-duration multiplier
    capacity_bph=12.0,    # node forwarding capacity, bundles per hour
    capacity_scale=1.0,
    bg_mean=0.5,          # mean share of capacity used by background traffic
    bg_amp=0.3,           # diurnal amplitude of background traffic (maximum at 14:00)
    uav_period_h=12.0,    # interval between UAV sorties, h
    uav_offset_h=2.0,     # first sortie at 02:00
    uav_cancel=0.15,      # probability that a sortie is cancelled
    uav_speed_kmh=60.0,
    uav_service_h=0.1,    # time on station per visit, h
    jam_rate_per_h=0.0,   # rate of area-wide interference episodes (all terrestrial links down), 1/h
    jam_mean_h=12.0,      # mean duration of an interference episode, h
)


def _km(a, b):
    dlat = (a[..., 0] - b[..., 0]) * 111.2
    dlon = (a[..., 1] - b[..., 1]) * 111.2 * np.cos(np.deg2rad(0.5 * (a[..., 0] + b[..., 0])))
    return np.hypot(dlat, dlon)


def topology(R_km=30.0):
    pos = np.vstack([LATLON, SINK_LATLON])
    n = len(pos)
    D = _km(pos[:, None, :], pos[None, :, :])
    links = [(i, j, float(D[i, j])) for i in range(n) for j in range(i + 1, n) if D[i, j] <= R_km]
    nbr = collections.defaultdict(list)
    for li, (i, j, _d) in enumerate(links):
        nbr[i].append((j, li)); nbr[j].append((i, li))
    sink = n - 1
    level = np.full(n, 99); level[sink] = 0
    frontier = [sink]
    while frontier:                                   # hop count to the fusion node
        nxt = []
        for u in frontier:
            for v, _ in nbr[u]:
                if level[v] > level[u] + 1:
                    level[v] = level[u] + 1; nxt.append(v)
        frontier = nxt
    return dict(pos=pos, D=D, links=links, nbr=nbr, level=level, sink=sink, dsink=D[:, sink])


def link_states(links, steps, p, seed):
    """Alternating renewal (Gilbert-Elliott) process of every link sampled on the DT grid.

    Every link has its own random stream, so regimes with different outage statistics share the same
    uniform draws (common random numbers)."""
    up = np.zeros((steps, len(links)), dtype=bool)
    T = steps * DT
    for li, (_i, _j, d) in enumerate(links):
        rng = np.random.default_rng([seed, 60_000, li])
        mu_up = p['up_mean_h']
        mu_dn = p['down_ref_h'] * max(d, 1.0) / 10.0 * p['down_scale']
        t = 0.0
        state = rng.random() < mu_up / (mu_up + mu_dn)
        while t < T:
            dur = -np.log1p(-rng.random()) * (mu_up if state else mu_dn)
            a, b = int(t / DT), int(min(T, t + dur) / DT)
            if state:
                up[a:b, li] = True
            t += dur; state = not state
    if p.get('jam_rate_per_h', 0.0) > 0:            # area-wide interference episodes
        rng = np.random.default_rng([seed, 80_000])
        jam = np.zeros(steps, dtype=bool)
        t = -np.log1p(-rng.random()) / p['jam_rate_per_h']
        while t < T:
            dur = -np.log1p(-rng.random()) * p['jam_mean_h']
            jam[int(t / DT):int(min(T, t + dur) / DT)] = True
            t += dur - np.log1p(-rng.random()) / p['jam_rate_per_h']
        up &= ~jam[:, None]
    return up


def uav_tour(topo):
    """Stations with hop level >= 2 are visited in angular order around the fusion node."""
    pos, sink = topo['pos'], topo['sink']
    targets = [i for i in range(len(pos) - 1) if topo['level'][i] >= 2]
    ang = [np.arctan2(pos[i, 0] - pos[sink, 0], pos[i, 1] - pos[sink, 1]) for i in targets]
    order = [targets[k] for k in np.argsort(ang)]
    legs, prev, t = [], sink, 0.0
    for v in order:
        t += topo['D'][prev, v]; legs.append((v, t)); prev = v
    t += topo['D'][prev, sink]
    return legs, t


def simulate(hours, seed, regime_of_hour, params_by_regime, tail_h=24 * 30):
    """hours: number of generation hours; regime_of_hour(t) -> key of params_by_regime (network state at t).

    Returns arrival times, delays, hop counts and UAV flags, shape (hours, 12, 2):
    [..., 0] pollutant bundle, [..., 1] meteorological bundle."""
    keys = sorted(params_by_regime)
    p0 = params_by_regime[keys[0]]
    topo = topology(p0['R_km'])
    links, nbr, level, sink, dsink = topo['links'], topo['nbr'], topo['level'], topo['sink'], topo['dsink']
    steps = int(round((hours + tail_h) / DT))
    reg_hour = [regime_of_hour(float(h)) for h in range(hours + tail_h + 1)]
    states = {key: link_states(links, steps, params_by_regime[key], seed) for key in set(reg_hour)}
    n_st = len(STATIONS)
    arr = np.full((hours, n_st, 2), np.nan)
    hops = np.zeros((hours, n_st, 2), np.int16)
    via_uav = np.zeros((hours, n_st, 2), bool)
    queues = [[collections.deque(), collections.deque()] for _ in range(n_st + 1)]   # [high, normal]
    credit = np.zeros(n_st + 1)
    legs, tour_len = uav_tour(topo)
    events = []
    urng = np.random.default_rng([seed, 70_000])
    t_s = p0['uav_offset_h']
    while t_s < hours + tail_h:
        p = params_by_regime[reg_hour[int(t_s)]]
        if urng.random() >= p['uav_cancel']:
            for v, dcum in legs:
                heapq.heappush(events, (t_s + dcum / p['uav_speed_kmh'], 0, v, t_s))
            t_ret = t_s + tour_len / p['uav_speed_kmh'] + p['uav_service_h'] * len(legs)
            heapq.heappush(events, (t_ret, 1, -1, t_s))
        t_s += p['uav_period_h']
    on_board = collections.defaultdict(list)
    delivered, total = 0, hours * n_st * 2
    per_hour = int(round(1 / DT))
    for k in range(steps):
        t = k * DT
        h = k // per_hour
        p = params_by_regime[reg_hour[h]]
        if k % per_hour == 0 and h < hours:          # hourly bundle generation
            for s in range(n_st):
                queues[s][0].append((h, s, 1, 0))       # meteorological, high priority
                queues[s][1].append((h, s, 0, 0))       # pollutant, normal priority
        while events and events[0][0] <= t + 1e-9:     # UAV visits and returns
            te, kind, v, sid = heapq.heappop(events)
            if kind == 0:
                for c in (0, 1):
                    on_board[sid].extend(queues[v][c]); queues[v][c].clear()
            else:
                for (g, s, cls, nh) in on_board.pop(sid, []):
                    arr[g, s, cls] = te; hops[g, s, cls] = nh + 1; via_uav[g, s, cls] = True
                    delivered += 1
        up = states[reg_hour[h]][k]
        bg = p['bg_mean'] + p['bg_amp'] * np.cos(2 * np.pi * ((t % 24.0) - 14.0) / 24.0)
        cap = p['capacity_bph'] * p['capacity_scale'] * max(0.0, 1.0 - bg) * DT
        moves = []
        for n in range(n_st):
            credit[n] = min(credit[n] + cap, max(1.0, 10 * cap))
            if not queues[n][0] and not queues[n][1]:
                continue
            cands = [(level[m], len(queues[m][0]) + len(queues[m][1]), dsink[m], m)
                     for m, li in nbr[n] if up[li] and level[m] < level[n]]   # queue-aware next hop
            if not cands:
                continue
            m = min(cands)[3]
            while credit[n] >= 1.0 and (queues[n][0] or queues[n][1]):
                q = queues[n][0] if queues[n][0] else queues[n][1]
                g, s, cls, nh = q.popleft(); credit[n] -= 1.0
                if m == sink:
                    arr[g, s, cls] = t + DT; hops[g, s, cls] = nh + 1; delivered += 1
                else:
                    moves.append((m, (g, s, cls, nh + 1)))
        for m, b in moves:
            queues[m][0 if b[2] == 1 else 1].append(b)
        if h >= hours and delivered >= total:
            break
    gen = np.arange(hours, dtype=float)[:, None, None]
    return dict(arrival=arr, delay=arr - gen, hops=hops, via_uav=via_uav, topology=topo,
                undelivered=int(np.isnan(arr).sum()))
