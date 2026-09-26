# Third-party code

`srattelm/agg/official_agg_same_protocol.py` includes the model classes `Time2Vec`, `FeedForward`,
`SelfAttentionBlock`, `ConditionalAttentionBlock` and `AsynchronousGraphGenerator` of the Asynchronous Graph
Generator:

- repository: https://github.com/ChristopherLey/AsyncGraphGenerator, commit 83ac07dad753536caeb2c38846822a7391a32569,
  files `AGG/model.py` and `AGG/utils.py`; license GNU GPL 3.0;
- article: C. P. Ley, F. Tobar, Asynchronous graph generator, Signal Process. 238 (2026) 110183,
  https://doi.org/10.1016/j.sigpro.2025.110183.

Changes (September 2026): the classes are combined in one file with a data adapter that enforces the two-clock
evaluation protocol of this repository, a per-target training schedule and a guard that adds one neutral node when
the previous 6-hour interval has no received value.

According to the terms of GPL-3.0, the whole folder `srattelm/agg/` is distributed under GPL-3.0
(`srattelm/agg/LICENSE`). The rest of the repository is distributed under the Apache License 2.0 (`LICENSE`).
