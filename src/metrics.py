"""Core retrieval, identity-margin, open-set, and bootstrap CI metrics for wildlife re-ID."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eval_retrieval(
    q_emb: np.ndarray,
    q_ids: np.ndarray,
    g_emb: np.ndarray,
    g_ids: np.ndarray,
    q_cams: np.ndarray | None = None,
    g_cams: np.ndarray | None = None,
    cam_mode: str = "all",
    leave_one_out: bool = False,
) -> dict:
    """Evaluate closed-set Rank-1, Rank-5, mAP, top-1 similarity, and distinct-identity margin."""
    sims = q_emb @ g_emb.T
    n_q = len(q_ids)
    r1 = np.zeros(n_q, dtype=float)
    r5 = np.zeros(n_q, dtype=float)
    aps = np.zeros(n_q, dtype=float)
    top1_sim = np.zeros(n_q, dtype=float)
    id_margin = np.zeros(n_q, dtype=float)
    pred_ids: list[str] = []
    valid = np.ones(n_q, dtype=bool)
    g_ids_arr = np.asarray(g_ids)

    for i in range(n_q):
        row = sims[i].copy()
        if cam_mode == "same" and q_cams is not None and g_cams is not None:
            mask = g_cams == q_cams[i]
        elif cam_mode == "cross" and q_cams is not None and g_cams is not None:
            mask = g_cams != q_cams[i]
        else:
            mask = np.ones(len(g_ids_arr), dtype=bool)

        if leave_one_out:
            mask = mask.copy()
            mask[i] = False

        if not np.any(mask & (g_ids_arr == q_ids[i])):
            valid[i] = False
            pred_ids.append("")
            continue

        row[~mask] = -1e9
        order = np.argsort(-row)
        ranked_ids = g_ids_arr[order]
        ranked_sims = row[order]

        pred_ids.append(str(ranked_ids[0]))
        r1[i] = float(ranked_ids[0] == q_ids[i])
        r5[i] = float(q_ids[i] in ranked_ids[:5])
        rel = np.where(ranked_ids == q_ids[i])[0]
        aps[i] = float((np.arange(1, len(rel) + 1) / (rel + 1)).mean())

        top1_sim[i] = float(ranked_sims[0])
        other = ranked_sims[(ranked_ids != ranked_ids[0]) & (ranked_sims > -1e8)]
        id_margin[i] = top1_sim[i] - (float(other[0]) if len(other) else 0.0)

    return {
        "n_valid": int(valid.sum()),
        "r1": float(r1[valid].mean()) if valid.any() else 0.0,
        "r5": float(r5[valid].mean()) if valid.any() else 0.0,
        "mAP": float(aps[valid].mean()) if valid.any() else 0.0,
        "r1_arr": r1[valid],
        "top1_sim": top1_sim[valid],
        "id_margin": id_margin[valid],
        "valid_mask": valid,
        "pred_ids": np.array(pred_ids)[valid],
    }


def openset_scores(
    known_emb: np.ndarray,
    unknown_emb: np.ndarray,
    g_emb: np.ndarray,
    g_ids: np.ndarray,
) -> dict:
    """Score known vs. never-seen individuals via top-1 cosine similarity and identity margin."""
    g_ids_arr = np.asarray(g_ids)

    def _score(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sims = emb @ g_emb.T
        t1, mg = [], []
        for i in range(len(emb)):
            order = np.argsort(-sims[i])
            r_ids = g_ids_arr[order]
            r_sims = sims[i][order]
            t1.append(float(r_sims[0]))
            other = r_sims[r_ids != r_ids[0]]
            mg.append(float(r_sims[0] - (other[0] if len(other) else 0.0)))
        return np.array(t1), np.array(mg)

    kt1, kmg = _score(known_emb)
    ut1, umg = _score(unknown_emb)
    y = np.concatenate([np.ones(len(kt1)), np.zeros(len(ut1))])

    out = {}
    for name, s in [("top1_sim", np.concatenate([kt1, ut1])), ("id_margin", np.concatenate([kmg, umg]))]:
        auc = float(roc_auc_score(y, s))
        fpr, tpr, _ = roc_curve(1 - y, -s)
        tpr5 = float(tpr[np.searchsorted(fpr, 0.05, side="right") - 1]) if np.any(fpr <= 0.05) else 0.0
        out[name] = {"auroc": round(auc, 4), "unk_tpr_at_5fpr": round(tpr5, 4)}
    return out


def bootstrap_ci(arr: np.ndarray, n_boot: int = 1000, seed: int = 42) -> list[float]:
    """Query-level bootstrap 95% confidence interval."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def selective_r1(r1_arr: np.ndarray, conf_scores: np.ndarray, coverage: float) -> float:
    """Rank-1 accuracy when keeping only the top `coverage` fraction of highest-confidence queries."""
    k = max(1, int(coverage * len(conf_scores)))
    order = np.argsort(-conf_scores)
    return float(r1_arr[order[:k]].mean())
