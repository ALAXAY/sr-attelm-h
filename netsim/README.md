# Network-derived arrival delays

Packet-level model of an intermittently connected monitoring network: 12 stations and a fusion node, links up to
30 km with Gilbert–Elliott states (mean up time 8 h, mean down time 4 h per 10 km of link length), store-and-forward
queues with a capacity of 12 bundles per hour and 20–80% diurnal background traffic, two priority classes,
queue-aware next-hop selection, a UAV data ferry every 12 h at 60 km/h with a cancellation probability of 0.15, and
area-wide interference episodes in the degraded regime (rate 1/41 h⁻¹, mean duration 12 h).

| File | Purpose |
|---|---|
| `netsim.py` | simulator |
| `generate_delays.py` | test realizations 11, 23, 37; regimes level1, level2, level4, shifted_A, shifted_B → `delays_network.npz` |
| `generate_training_delays.py` | realizations of the preliminary stage 101 (split A), 1101 (split B), levels 1, 2, 4 → `delays_network_train.npz` |
| `network_delays.py` | drop-in replacement of the geometric delay function |
| `analyze_delays.py` | geometric delay model of the article and delay statistics |
| `delay_statistics.py` | Tables 10, 11 and the autocorrelation of hourly delays → `out/*.csv` |
| `fig_delays.py`, `make_figure.py` | Fig. 8 (EN and UK labels) → `out/` |

Delays are float32 hours, shape (35064, 12, 2): `[..., 0]` pollutant bundle, `[..., 1]` meteorological bundle.
Both generators are deterministic; the shipped `.npz` files are reproduced bit-identically. Station positions are
approximate (about 1 km) and are used only for inter-node distances.
