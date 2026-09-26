# SR-AttELM-H: technique for adaptive reconstruction of asynchronously delivered sensor data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ALAXAY/sr-attelm-h/blob/main/notebooks/SR_AttELM_H_Colab.ipynb)

Code for the article "A technique for adaptive reconstruction of asynchronously delivered sensor data using
availability-aware spatiotemporal attention and a sector-structured recurrent extreme learning machine"
(O. D. Fesenko, under review in Ad Hoc Networks). Ukrainian description: [README_UA.md](README_UA.md).

## Purpose

The code reproduces the experimental evaluation of the technique. The state vector at the decision time is
estimated only from the messages received by that time. Availability-aware spatiotemporal attention weights the
sources by their relation to the target and by the age of their data; a recurrent extreme learning machine with a
sector-structured output layer maps the attention summary to the estimate, and recursive least squares (RLS)
refines the output matrix of the active sector once the verified value arrives.

## Contents

| Folder | Contents |
|---|---|
| `srattelm/` | evaluation protocol with two clocks per message (generation time and arrival time of each component), stages 1–7 and the preliminary stage of the technique, component-exclusion variants, last observation carried forward (LOCF) |
| `srattelm/agg/` | adapter of the Asynchronous Graph Generator (AGG) core to the same protocol |
| `netsim/` | packet-level model of an intermittently connected monitoring network, network delay realizations, delay statistics and Fig. 8 of the article |
| `notebooks/` | Google Colab notebook that reruns the experiments |
| `data/` | location of the Beijing Multi-Site Air Quality archive; the data set is not redistributed |

## Agreement with the frozen protocol

The source code of the configuration frozen for Experiments 1–4 of the article is not available. The technique
was re-implemented from its description in Sections 4 and 5, and this re-implementation is used in Section 7.8
(Tables 12 and 13 of the article). The set of delayed components is reproduced exactly; the LOCF RMSE agrees to
four decimals in five of six runs and differs by 0.0001 in the sixth (Table 1).

Table 1. Agreement of the re-implementation with the frozen evaluation protocol

| Quantity | Frozen configuration | Re-implementation |
|---|---|---|
| Delayed components, split A (test seeds 11 / 23 / 37) | 31722 / 31718 / 31728 | 31722 / 31718 / 31728 |
| Delayed components, split B | 42042 / 42044 / 42038 | 42042 / 42044 / 42038 |
| LOCF RMSE (Table 6 of the article) | 0.5089 / 0.5015 / 0.5166 / 0.5714 / 0.5749 / 0.5553 | 0.5089 / 0.5015 / 0.5166 / 0.5714 / 0.5749 / 0.5554 |
| Pooled RMSE of the technique, geometric delays | 0.2909 | 0.2941 |

Parameters not given in the article are set in `srattelm/technique.py`: attention dimension 16, hidden layer of the
attention block 64, recurrent state dimension 20, spectral radius 0.8, ridge coefficient 1. The forgetting factor
is selected on the validation part from the range 0.98–1.0.

## Running

In Google Colab, open `notebooks/SR_AttELM_H_Colab.ipynb` with the badge above and run all cells. The notebook
downloads the data set from the UCI repository, checks the SHA-256 checksum of the data array and the evaluation
protocol, and then runs the experiments. Without the AGG core, the run takes about 40–60 min on the Colab CPU.

Locally (Python 3.10 or later):

```
pip install -r requirements.txt
curl -L -o data/beijing_multi_site_air_quality.zip https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
cd srattelm
python run_experiments.py        # technique, variants, LOCF; geometric and network delays
python run_component_check.py    # Table 13 of the article
python agg/run_agg.py            # official AGG core (20-25 min per split and delay model, 12 threads)
python summarize.py              # summary_network.csv, paired_network.csv (Table 12)
python summarize_components.py   # availability_check_summary.csv (Table 13)
cd ../netsim
python delay_statistics.py       # Tables 10 and 11, delay autocorrelation
python make_figure.py            # Fig. 8
```

The data path can be set with the environment variable `SRATTELM_DATA` (an archive at any nesting level or a
folder with the `PRSA_Data_*.csv` files). The network delay realizations are stored in `netsim/`;
`generate_delays.py` and `generate_training_delays.py` regenerate them bit for bit.

## Results

Table 2. Pooled RMSE on the delayed components over six paired runs

| Method | Geometric delays | Network delays |
|---|---|---|
| Last observation carried forward (LOCF) | 0.5430 | 0.7968 |
| Official AGG core | 0.3451 | 0.4974 |
| Sector recurrent ELM without attention | 0.5066 | 0.6927 |
| Proposed technique | 0.2941 | 0.3894 |
| Proposed technique, preliminary stage on geometric delays | – | 0.4214 |

Table 3. Effect of the availability parameters in the source weights

| Model | Availability parameters in the source weights | RMSE, geometric | RMSE, network | Relative entropy of the weights, geometric |
|---|---|---|---|---|
| Sector recurrent ELM without attention | – | 0.5066 | 0.6927 | – |
| Spatial attention | not used | 0.4088 | 0.5568 | 0.985 |
| Spatiotemporal attention | terms of the relation score (10) | 0.3749 | 0.4873 | 0.833 |
| Proposed technique | terms of (10) | 0.2931 | 0.3904 | 0.833 |
| Proposed technique | terms of (10) and query/key vectors | 0.2941 | 0.3894 | 0.814 |
| Proposed technique | not used | 0.3278 | 0.4608 | 0.985 |

On network delays, the technique reduced the RMSE by 51.13% relative to LOCF and by 21.73% relative
to the official AGG core, in six of six paired runs. Including the availability parameters in the source weights
reduced the RMSE by 10.28% on geometric and by 15.49% on network delays; without them, the source weights
are close to uniform.

## Limitations

1. The technique is a re-implementation. Parameters absent from the article were set anew, so the 0.15% change for
   the spatial-attention variant and the magnitude of the entropy reduction (Table 8 of the article) belong to the
   frozen configuration and are not reproduced.
2. One data set (twelve air-quality monitoring stations) is used, and arrival times come from delay models. Station
   coordinates in the network model are approximate (about 1 km) and serve only to compute inter-node distances.
3. The AGG core is trained with the per-target schedule of the adapter (7,000 targets, up to 18 epochs); the full
   training pipeline of the official repository is not used.
4. Results were obtained on a CPU with Python 3.12.14, PyTorch 2.13.0, NumPy 2.5.3 and pandas 3.0.6. On other
   hardware the RMSE of the technique may differ in the fourth decimal; the set of delayed components does not
   change. GPU execution was not tested.

## License

The code is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). The folder `srattelm/agg/` includes model
classes transcribed from ChristopherLey/AsyncGraphGenerator, which is distributed under GNU GPL 3.0; this folder is
therefore licensed under GPL-3.0 (`srattelm/agg/LICENSE`, `THIRD_PARTY_NOTICE.md`). The data set is used under the
terms of the UCI Machine Learning Repository (https://doi.org/10.24432/C5RK5G).

## Citation

See `CITATION.cff`.
