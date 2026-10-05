import numpy as np
import scipy.sparse as sp

from recsys.metrics import hit_matrix, ndcg_at_k, recall_at_k


def _target():
    # user 0 : items 1 et 3 pertinents ; user 1 : item 0 pertinent
    return sp.csr_matrix(np.array([[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=np.float32))


def test_hit_matrix_with_padding():
    rec = np.array([[1, 2, 3], [4, -1, -1]])
    hits = hit_matrix(rec, np.array([0, 1]), _target())
    assert hits.tolist() == [[True, False, True], [False, False, False]]


def test_recall_at_k():
    hits = np.array([[True, False, True], [False, False, False]])
    n_rel = np.array([2, 1])
    assert recall_at_k(hits, n_rel, 1).tolist() == [0.5, 0.0]
    assert recall_at_k(hits, n_rel, 3).tolist() == [1.0, 0.0]


def test_ndcg_perfect_and_known_value():
    perfect = np.array([[True, True, False]])
    assert np.isclose(ndcg_at_k(perfect, np.array([2]), 3)[0], 1.0)
    # un seul pertinent classé 2e : DCG = 1/log2(3), IDCG = 1
    second = np.array([[False, True, False]])
    assert np.isclose(ndcg_at_k(second, np.array([1]), 3)[0], 1 / np.log2(3))


def test_ndcg_caps_ideal_at_k():
    # 5 pertinents mais k=2 : l'idéal ne compte que 2 positions
    hits = np.array([[True, True]])
    assert np.isclose(ndcg_at_k(hits, np.array([5]), 2)[0], 1.0)
