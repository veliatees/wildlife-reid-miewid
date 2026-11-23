import numpy as np
import pytest

from src.metrics import bootstrap_ci, eval_retrieval, openset_scores, selective_r1


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_id_margin_skips_same_identity_gallery_duplicates():
    """Identity margin must compare best identity vs. best *other* identity, not 2nd photo of same animal."""
    g_emb = _norm([
        [1.0, 0.0],  # cow A photo 1 (sim=1.0)
        [0.99, 0.1], # cow A photo 2 (same ID, must be skipped!)
        [0.5, 0.866], # cow B (different ID, sim=0.5)
    ])
    g_ids = np.array(["cow_A", "cow_A", "cow_B"])
    q_emb = _norm([[1.0, 0.0]])
    q_ids = np.array(["cow_A"])

    res = eval_retrieval(q_emb, q_ids, g_emb, g_ids)
    assert res["r1"] == 1.0
    assert res["mAP"] == 1.0
    assert pytest.approx(res["id_margin"][0], abs=1e-3) == 0.5


def test_cross_camera_mask_excludes_same_camera_matches():
    g_emb = _norm([
        [1.0, 0.0],  # cow A on cam1 (exact match, must be masked out in cross mode)
        [0.0, 1.0],  # cow A on cam2 (orthogonal -> worse than cow B on cam2)
        [0.8, 0.6],  # cow B on cam2
    ])
    g_ids = np.array(["cow_A", "cow_A", "cow_B"])
    g_cams = np.array(["cam1", "cam2", "cam2"])
    q_emb = _norm([[1.0, 0.0]])
    q_ids = np.array(["cow_A"])
    q_cams = np.array(["cam1"])

    same = eval_retrieval(q_emb, q_ids, g_emb, g_ids, q_cams, g_cams, cam_mode="same")
    cross = eval_retrieval(q_emb, q_ids, g_emb, g_ids, q_cams, g_cams, cam_mode="cross")
    assert same["r1"] == 1.0
    assert cross["r1"] == 0.0
    assert cross["pred_ids"][0] == "cow_B"


def test_leave_one_out_excludes_self():
    emb = _norm([
        [1.0, 0.0],  # id1 left
        [0.0, 1.0],  # id1 right (only other sample of id1)
        [0.9, 0.436], # id2 right (closer to sample 0 than sample 1 is!)
    ])
    ids = np.array(["id1", "id1", "id2"])
    ori = np.array(["left", "right", "right"])
    res = eval_retrieval(emb, ids, emb, ids, ori, ori, cam_mode="cross", leave_one_out=True)
    assert res["n_valid"] == 2
    # sample 0 (id1 left) matches id2 right -> wrong; sample 1 (id1 right) matches id1 left -> correct
    assert res["r1"] == 0.5


def test_openset_scores_perfect_separation():
    g_emb = _norm([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    g_ids = np.array(["A", "B"])
    known = _norm([[0.99, 0.1, 0.0], [0.1, 0.99, 0.0]])
    unknown = _norm([[0.0, 0.0, 1.0], [0.1, 0.1, 0.99]])
    res = openset_scores(known, unknown, g_emb, g_ids)
    assert res["top1_sim"]["auroc"] == 1.0
    assert res["id_margin"]["auroc"] == 1.0


def test_bootstrap_ci_and_selective_r1():
    r1 = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    conf = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    assert selective_r1(r1, conf, coverage=0.6) == 1.0
    lo, hi = bootstrap_ci(r1, n_boot=500, seed=42)
    assert 0.0 <= lo <= hi <= 1.0
