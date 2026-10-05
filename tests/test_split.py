import numpy as np

from recsys.data import make_synthetic, temporal_split


def test_temporal_split_is_chronological_and_disjoint():
    data = make_synthetic(n_users=50, n_items=200, mean_inter=40, seed=1)
    s = temporal_split(data.interactions)
    order = ["retriever", "ranker", "valid", "test"]
    for u in s.test["user"].unique():
        spans = [getattr(s, name).loc[lambda d: d["user"] == u, "timestamp"] for name in order]
        assert all(len(x) > 0 for x in spans)
        for earlier, later in zip(spans, spans[1:]):
            assert earlier.max() <= later.min()
    keys = [set(zip(getattr(s, n)["user"], getattr(s, n)["item"])) for n in order]
    assert sum(len(k) for k in keys) == len(set().union(*keys))


def test_min_positives_filter():
    data = make_synthetic(n_users=30, n_items=100, mean_inter=30, seed=2)
    s = temporal_split(data.interactions, min_positives=10)
    counts = np.bincount(np.concatenate([getattr(s, n)["user"].to_numpy() for n in ("retriever", "ranker", "valid", "test")]))
    assert counts[counts > 0].min() >= 10
