"""PERSONA-MOSAIC: a mixture of LightGCN experts whose experts are clusters of LLM-written user personas.

Phase 1 - experts from personas
    Every user has 1..n LLM-written personas (``data/<dataset>/persona/<persona_file>``: ``record_ids`` 'u<uid>:p<j>',
    ``field_mean`` [records, dim]); every item has one text vector, its RLMRec profile embedding
    (``data/<dataset>/itm_emb_np.pkl``, same embedder).  Text vectors are mean-centred and L2-normalised.
    The personas are clustered in two stages: stage 1 splits the pool into ``clustering_kind_split`` persona kinds
    (k-means), stage 2 clusters every kind on kind-centred vectors with K = the silhouette maximum over
    [``clustering_k_min``, ``clustering_k_max``].  The experts are the union of the per-kind clusters; no validation
    data is used.  The result is cached in ``data/<dataset>/persona/persona_clusters_<spec>.pkl`` so that every model
    build, and every later run, uses the same experts.

Phase 2 - mixture of experts
    Expert k is a LightGCN whose layers are modulated by a persona prior.  For node v the prior input is the persona of
    v closest to cluster k times the cluster centroid (an item's text vector plays that role for every k), projected by
    MLP^k; layer l multiplies the embeddings by 1 + lambda_l * tanh(LN(MLP_l(prior))) before propagation and the
    read-out is a softmax-weighted sum of the layers.  One shared gate (an MLP over the node's cosine similarities to
    the K clusters, softmax, top-k) mixes the expert outputs.
    Loss: BPR + L2 regularisation + gate balance + gate-weighted InfoNCE alignment of the routed output to the priors of
    the selected experts + a text-free SimGCL-type contrast between two noisy views of the same propagation.

Phase 3 - fixed content score (prediction only)
    y*_ui = s_ui + b * std_i(s_u.) * z_i(c_u.): s is the model score, c_ui the cosine between the unit mean of user u's
    taste-kind personas and the text vector of item i, z the standardisation over the catalogue for that user (the
    ranking of z(s_u.) + b * z(c_u.)).  Nothing is trained and the loss never sees the term.
"""
from __future__ import annotations

import math
import os
import pickle
import re
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch as t
from torch import nn
import torch.nn.functional as F

from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import reg_params
from models.model_utils import SpAdjEdgeDrop


init = nn.init.xavier_uniform_
PriorCache = Dict[Tuple[str, int], t.Tensor]

ITEM_PROFILE_FILE = "itm_emb_np.pkl"  # RLMRec item profile embeddings (configs['itmprf_embeds'])
CLUSTER_CACHE_TEMPLATE = "persona/persona_clusters_{spec}.pkl"
_REQUIRED = object()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _as_csr_binary(mat) -> sp.csr_matrix:
    if not sp.issparse(mat):
        mat = sp.csr_matrix(mat)
    return (mat.tocsr() != 0).astype(np.float32)


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def _combined_zscore(user_features: np.ndarray, item_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score every column over users and items together (one shared gate reads both)."""
    all_features = np.vstack([user_features, item_features]).astype(np.float32)
    mean = all_features.mean(axis=0, keepdims=True)
    std = all_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((user_features - mean) / std).astype(np.float32), ((item_features - mean) / std).astype(np.float32)


_RECORD_RE = re.compile(r"^([ui])(\d+):p(\d+)$")


def _load_user_personas(path: Path, user_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """Persona generation output {'record_ids': ['u<uid>:p<j>', ...], 'field_mean': [records, dim]} ->
    (embeddings [personas, dim], owner [personas] = the user of every persona).  Item records, if any, are skipped."""
    obj = _load_pickle(path)
    if not isinstance(obj, dict) or "record_ids" not in obj or "field_mean" not in obj:
        raise ValueError(f"{path} must be a dict with 'record_ids' and 'field_mean' (persona generation output).")
    record_ids = [str(r) for r in obj["record_ids"]]
    emb = np.asarray(obj["field_mean"], dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(record_ids):
        raise ValueError(f"'field_mean' must be [records, dim] aligned with 'record_ids' in {path}: {emb.shape} vs {len(record_ids)} ids.")
    if not np.isfinite(emb).all():
        raise ValueError(f"Persona embeddings contain NaN/inf: {path}")
    is_user = np.zeros(len(record_ids), dtype=bool)
    owner = np.empty(len(record_ids), dtype=np.int64)
    for r, rid in enumerate(record_ids):
        match = _RECORD_RE.match(rid)
        if match is None:
            raise ValueError(f"Unrecognised persona record id {rid!r} in {path}; expected 'u<uid>:p<j>'.")
        is_user[r] = match.group(1) == "u"
        owner[r] = int(match.group(2))
    if not is_user.any():
        raise ValueError(f"No user persona records ('u<uid>:p<j>') found in {path}.")
    if "uids" in obj:
        uids = np.asarray(obj["uids"]).reshape(-1)
        if uids.shape[0] == len(record_ids) and not np.array_equal(uids[is_user], owner[is_user]):
            raise ValueError(f"'uids' disagree with the user ids parsed from 'record_ids' in {path}.")
    emb, owner = np.ascontiguousarray(emb[is_user], dtype=np.float32), owner[is_user]
    if int(owner.min()) < 0 or int(owner.max()) >= user_num:
        raise ValueError(f"User persona ids in {path} are outside [0, {user_num}): min={int(owner.min())}, max={int(owner.max())}")
    return emb, owner


def _load_item_profiles(data_dir: Path, item_num: int, raw_dim: int) -> np.ndarray:
    """RLMRec item profile embeddings [item_num, raw_dim] (configs['itmprf_embeds'] or data/<dataset>/itm_emb_np.pkl)."""
    profile_path = data_dir / ITEM_PROFILE_FILE
    profiles = configs.get("itmprf_embeds")
    if profiles is None:
        if not profile_path.exists():
            raise FileNotFoundError(f"PersonaMosaic needs the RLMRec item profile embeddings {profile_path}.")
        profiles = _load_pickle(profile_path)
    profiles = np.asarray(profiles, dtype=np.float32)
    if profiles.ndim != 2 or profiles.shape[0] != item_num or profiles.shape[1] != raw_dim:
        raise ValueError(
            f"Item profile embeddings must be [{item_num}, {raw_dim}] to live in the persona text space, "
            f"got {profiles.shape} from {profile_path}."
        )
    if not np.isfinite(profiles).all():
        raise ValueError(f"Item profile embeddings contain NaN/inf: {profile_path}")
    return profiles


# ---------------------------------------------------------------------------
# Persona clustering: k-means++ / Lloyd with silhouette-based K selection (torch, GPU when available)
# ---------------------------------------------------------------------------


def _clustering_device(preference: str = "auto") -> t.device:
    """Device of the k-means / silhouette computation: 'auto' follows the run device, 'cpu' is bit-exact."""
    preference = str(preference or "auto").lower()
    if preference == "cpu":
        return t.device("cpu")
    if preference in {"cuda", "gpu"}:
        if not t.cuda.is_available():
            raise ValueError("clustering_device=cuda but CUDA is not available.")
        return t.device("cuda")
    if preference != "auto":
        raise ValueError("clustering_device must be 'auto', 'cpu' or 'cuda'.")
    want = str(configs.get("device", "cpu")).lower()
    if want.startswith("cuda") and t.cuda.is_available():
        return t.device("cuda")
    return t.device("cpu")


def _kmeans_pp_init(X: t.Tensor, K: int, gen: t.Generator) -> t.Tensor:
    """k-means++ seeding on unit-norm rows (squared distance = 2 - 2 cos)."""
    N = X.shape[0]
    first = t.randint(N, (1,), generator=gen, device=X.device)
    centers = X[first]
    d2 = (2.0 - 2.0 * (X @ centers[0])).clamp(min=0.0)
    for _ in range(1, K):
        probs = d2 / d2.sum().clamp(min=1e-12)
        idx = t.multinomial(probs, 1, generator=gen)
        c = X[idx]
        centers = t.cat([centers, c], dim=0)
        d2 = t.minimum(d2, (2.0 - 2.0 * (X @ c[0])).clamp(min=0.0))
    return centers


def _kmeans(X: t.Tensor, K: int, n_init: int, max_iter: int, seed: int, tol: float = 1e-6) -> Tuple[t.Tensor, float]:
    """Lloyd k-means on unit-norm rows; returns the labels and inertia of the restart with the lowest inertia."""
    best = None
    for restart in range(max(1, int(n_init))):
        gen = t.Generator(device=X.device).manual_seed(int(seed) * 100003 + int(K) * 1009 + restart)
        centers = _kmeans_pp_init(X, K, gen)
        for _ in range(max(1, int(max_iter))):
            d2 = (centers * centers).sum(dim=1)[None, :] - 2.0 * (X @ centers.T) + 1.0
            labels = d2.argmin(dim=1)
            counts = t.bincount(labels, minlength=K).to(X.dtype)
            new_centers = t.zeros_like(centers).index_add_(0, labels, X)
            empty = counts == 0
            if bool(empty.any()):  # re-seed empty clusters with the points farthest from their centroid
                farthest = d2.gather(1, labels[:, None]).squeeze(1).topk(int(empty.sum())).indices
                new_centers[empty] = X[farthest]
                counts[empty] = 1.0
            new_centers = new_centers / counts[:, None]
            shift = float(((new_centers - centers) ** 2).sum())
            centers = new_centers
            if shift < tol:
                break
        d2 = (centers * centers).sum(dim=1)[None, :] - 2.0 * (X @ centers.T) + 1.0
        labels = d2.argmin(dim=1)
        inertia = float(d2.gather(1, labels[:, None]).sum())
        if best is None or inertia < best[1]:
            best = (labels.clone(), inertia)
    return best


@t.no_grad()
def _silhouette_cosine(X: t.Tensor, labels: t.Tensor, K: int, sample_size: int, seed: int, chunk: int = 2048) -> float:
    """Mean silhouette with cosine distance on a random subsample (sklearn sample_size semantics)."""
    N = X.shape[0]
    if sample_size and 0 < sample_size < N:
        gen = t.Generator(device="cpu").manual_seed(int(seed))
        idx = t.randperm(N, generator=gen)[:sample_size].to(X.device)
        Xs, ls = X[idx], labels[idx]
    else:
        Xs, ls = X, labels
    S = Xs.shape[0]
    onehot = t.zeros(S, K, device=X.device, dtype=X.dtype).scatter_(1, ls[:, None], 1.0)
    counts = onehot.sum(dim=0)
    scores = []
    for start in range(0, S, chunk):
        dist = (1.0 - Xs[start : start + chunk] @ Xs.T).clamp(min=0.0)
        cluster_sum = dist @ onehot
        own = ls[start : start + chunk]
        own_count = counts[own]
        a = cluster_sum.gather(1, own[:, None]).squeeze(1) / (own_count - 1.0).clamp(min=1.0)
        mean_other = cluster_sum / counts.clamp(min=1.0)[None, :]
        mean_other.scatter_(1, own[:, None], float("inf"))
        b = mean_other.min(dim=1).values
        sil = (b - a) / t.maximum(a, b).clamp(min=1e-12)
        scores.append(t.where(own_count > 1, sil, t.zeros_like(sil)))
    return float(t.cat(scores).mean())


def _two_stage_clusters(X: t.Tensor, params: Dict) -> Dict:
    """Kind-aware two-stage clustering of the unit-norm persona pool X.

    Stage 1: k-means with ``kind_split`` clusters = the persona kinds (the content-taste persona of every user vs the
    personas of the other kinds dominate the pool as its leading direction; a one-stage silhouette rule returns exactly
    that split, K = 2).  Stage 2: inside every kind, k-means on kind-centred unit vectors with K = the silhouette
    maximum over [k_min, k_max].  The experts are the union of the per-kind clusters and their centroids the pool-space
    unit means of their members, so cluster similarities and the closest-persona choice live in the pool space.
    """
    N = int(X.shape[0])
    kind_split, k_min, k_max = int(params["kind_split"]), int(params["k_min"]), int(params["k_max"])
    if N <= kind_split:
        raise ValueError(f"Cannot split {N} persona records into {kind_split} kinds.")
    kind_labels, _ = _kmeans(X, kind_split, params["n_init"], params["max_iter"], params["seed"])
    kind_counts = t.bincount(kind_labels, minlength=kind_split)
    order = sorted(range(kind_split), key=lambda j: (-int(kind_counts[j]), j))  # largest kind first (stable numbering)
    labels = t.full((N,), -1, dtype=t.long, device=X.device)
    centroids, kinds, offset = [], [], 0
    for j in order:
        idx = (kind_labels == j).nonzero(as_tuple=False).squeeze(1)
        n_j = int(idx.numel())
        Xj = X[idx]
        Xj_c = F.normalize(Xj - Xj.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
        candidates = list(range(k_min, min(k_max, max(n_j - 1, 1)) + 1)) if n_j > k_min else []
        if not candidates:  # a tiny kind: one expert
            lab, K_j, silhouette = t.zeros(n_j, dtype=t.long, device=X.device), 1, {1: float("nan")}
        else:
            swept = {}
            for K in candidates:
                lab_K, _ = _kmeans(Xj_c, K, params["n_init"], params["max_iter"], params["seed"])
                swept[K] = (_silhouette_cosine(Xj_c, lab_K, K, params["silhouette_sample"], params["seed"]), lab_K)
            K_j = int(max(swept, key=lambda K: swept[K][0]))  # ties -> smallest K
            lab = swept[K_j][1]
            silhouette = {int(K): float(score) for K, (score, _) in swept.items()}
        labels[idx] = lab + offset
        for k in range(K_j):
            centroids.append(X[idx[lab == k]].mean(dim=0).cpu().numpy())
        kinds.append({"kind": int(j), "size": n_j, "K": K_j, "silhouette": silhouette,
                      "sizes": t.bincount(lab, minlength=K_j).tolist(), "offset": int(offset)})
        offset += K_j
    labels_np = labels.cpu().numpy().astype(np.int64)
    return {
        "K": int(offset),
        "centroids": _l2_normalize_np(np.stack(centroids)),
        "labels": labels_np,
        "sizes": np.bincount(labels_np, minlength=offset).tolist(),
        "kind_split": kind_split,
        "kinds": kinds,
    }


def _compute_clusters(embeddings: np.ndarray, params: Dict, device: t.device) -> Dict:
    """Centre and L2-normalise the user persona pool, then cluster it in two stages."""
    mean = embeddings.mean(axis=0, keepdims=True)
    X = t.from_numpy(np.ascontiguousarray(_l2_normalize_np(embeddings - mean), dtype=np.float32)).to(device)
    out = _two_stage_clusters(X, params)
    out.update({"projection": {"mean": mean}, "params": params, "fingerprint": None, "device": str(device)})
    return out


def _read_clustering_cache(cache_path: Path, params: Dict, fingerprint: Dict) -> Optional[Dict]:
    """Cached clustering if it exists and matches the persona file and the clustering parameters."""
    if not cache_path.exists():
        return None
    try:
        cached = _load_pickle(cache_path)
    except Exception as exc:  # corrupted / partially written cache
        warnings.warn(f"Could not read persona cluster cache {cache_path} ({exc}); recomputing.")
        return None
    if (
        isinstance(cached, dict)
        and cached.get("params") == params
        and cached.get("fingerprint") == fingerprint
        and all(key in cached for key in ("centroids", "labels", "projection", "kinds"))
    ):
        return cached
    warnings.warn(
        f"Persona cluster cache {cache_path} is stale (persona file or clustering parameters changed); recomputing."
    )
    return None


def _write_clustering_cache(cache_path: Path, out: Dict) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        with tmp_path.open("wb") as f:
            pickle.dump(out, f)
        os.replace(tmp_path, cache_path)
    except OSError as exc:
        warnings.warn(f"Could not write persona cluster cache {cache_path}: {exc}")


_CLUSTER_LOCK_TIMEOUT = 900.0  # seconds to wait for another process that is computing the same cache


def _cluster_personas(embeddings: np.ndarray, params: Dict, cache_path: Path, fingerprint: Dict, device: t.device) -> Dict:
    """Cluster the user personas, through the on-disk cache.  A lock file lets runs launched at the same time share one
    computation instead of racing to write different clusterings."""
    cached = _read_clustering_cache(cache_path, params, fingerprint)
    if cached is not None:
        return cached

    lock_path = cache_path.with_name(cache_path.name + ".lock")
    have_lock = False
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        have_lock = True
    except FileExistsError:
        print(f"[persona_mosaic] waiting for another process to write {cache_path} (lock: {lock_path})")
        deadline = time.time() + _CLUSTER_LOCK_TIMEOUT
        while time.time() < deadline:
            time.sleep(2.0)
            if not lock_path.exists():
                cached = _read_clustering_cache(cache_path, params, fingerprint)
                if cached is not None:
                    return cached
                break
        else:
            warnings.warn(
                f"Lock {lock_path} has been held for more than {_CLUSTER_LOCK_TIMEOUT:.0f}s; computing the "
                "persona clusters in this process. Delete the lock file if no other run is active."
            )
    except OSError as exc:
        warnings.warn(f"Could not create lock {lock_path}: {exc}")

    try:
        out = _compute_clusters(embeddings, params, device)
        out["fingerprint"] = fingerprint
        _write_clustering_cache(cache_path, out)
    finally:
        if have_lock:
            try:
                os.unlink(lock_path)
            except OSError:
                pass
    return out


# ---------------------------------------------------------------------------
# Per-node persona tables
# ---------------------------------------------------------------------------


def _node_slots(owner: np.ndarray, node_num: int) -> np.ndarray:
    """[node_num, max_records_per_node] record indices of every node, padded with -1."""
    counts = np.bincount(owner, minlength=node_num) if owner.size else np.zeros(node_num, dtype=np.int64)
    max_per_node = max(1, int(counts.max()) if counts.size else 1)
    order = np.argsort(owner, kind="stable")
    sorted_owner = owner[order]
    starts = np.searchsorted(sorted_owner, np.arange(node_num))
    slots = np.full((node_num, max_per_node), -1, dtype=np.int64)
    for j in range(max_per_node):
        has = counts > j
        slots[has, j] = order[starts[has] + j]
    return slots


def _best_persona_per_cluster(table: np.ndarray, slots: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every node and cluster: the node's text row with the largest cosine to the cluster centroid.

    ``table`` is [records + 1, dim] whose last row is the fallback (zero) row used by nodes without records.
    Returns (record index [nodes, K], similarity [nodes, K], missing mask [nodes]).
    """
    sims = table @ centroids.T  # [records + 1, K]
    fallback_row = table.shape[0] - 1
    valid = slots >= 0
    gathered = sims[np.where(valid, slots, fallback_row)]  # [nodes, max_per_node, K]
    gathered = np.where(valid[:, :, None], gathered, -np.inf)
    best_slot = gathered.argmax(axis=1)  # [nodes, K]
    best_sim = np.take_along_axis(gathered, best_slot[:, None, :], axis=1)[:, 0, :]
    best_idx = np.take_along_axis(slots, best_slot, axis=1)
    missing = ~valid.any(axis=1)
    best_idx[missing] = fallback_row
    if missing.any():
        best_sim[missing] = best_sim[~missing].mean(axis=0) if (~missing).any() else 0.0
    return best_idx.astype(np.int64), best_sim.astype(np.float32), missing


def _taste_content_rows(bundle: SimpleNamespace, trn_mat: sp.csr_matrix, user_num: int) -> Tuple[np.ndarray, str]:
    """User rows of the fixed content score: [users, dim] unit mean of each user's taste-kind personas (zero row = the
    user has none, so the term vanishes for that user), and a note for the run summary.

    The taste kind is the stage-1 kind whose personas are, on average, closest to the mean text vector of the items
    their own user interacted with in TRAINING (no validation / test data, nothing trained).
    """
    kinds = bundle.clusters["kinds"]
    pool, owner = bundle.user_table[:-1], bundle.user_owner
    kind_of_cluster = np.zeros(int(bundle.K), dtype=np.int64)
    for kd in kinds:
        kind_of_cluster[int(kd["offset"]) : int(kd["offset"]) + int(kd["K"])] = int(kd["kind"])
    kind_of = kind_of_cluster[bundle.labels]
    user_item_mean = _l2_normalize_np(np.asarray(trn_mat @ bundle.item_table[:-1], dtype=np.float32))  # [users, dim]
    own = np.einsum("ij,ij->i", pool, user_item_mean[owner])
    kind_ids = sorted({int(kd["kind"]) for kd in kinds})
    own_by_kind = {k: float(own[kind_of == k].mean()) for k in kind_ids}
    taste_kind = max(kind_ids, key=lambda k: own_by_kind[k])
    sel = kind_of == taste_kind
    rows = np.zeros((user_num, pool.shape[1]), dtype=np.float32)
    np.add.at(rows, owner[sel], pool[sel])  # the sum of unit persona rows has the direction of their mean
    has = np.bincount(owner[sel], minlength=user_num) > 0
    note = (
        f"taste kind = stage-1 kind {taste_kind} ({int(sel.sum())} personas; mean cosine to the own user's training items by kind "
        + ", ".join(f"{k}: {v:.4f}" for k, v in own_by_kind.items())
        + f"); users with a content row {int(has.sum())}/{user_num}"
    )
    return _l2_normalize_np(rows), note


_BUNDLE_MEMO: Dict[tuple, SimpleNamespace] = {}


def _build_persona_bundle(data_dir: Path, user_num: int, item_num: int, persona_file: str, clustering: Dict, device: str) -> SimpleNamespace:
    """Load the user personas and the item profiles, cluster the personas into experts and derive the per-node text
    tables and cluster similarities.  Memoised: the trainer builds the model more than once per run."""
    memo_key = (str(data_dir.resolve()), persona_file, tuple(sorted(clustering.items())), device)
    if memo_key in _BUNDLE_MEMO:
        return _BUNDLE_MEMO[memo_key]

    persona_path = data_dir / "persona" / persona_file
    if not persona_path.exists():
        raise FileNotFoundError(f"Missing LLM persona embeddings: {persona_path} (persona generation step; see persona_file).")
    personas, owner = _load_user_personas(persona_path, user_num)
    raw_dim = int(personas.shape[1])
    profiles = _load_item_profiles(data_dir, item_num, raw_dim)

    # Every persona file has its own clustering cache: persona_emb_v22.pkl -> persona_clusters_v22_<spec>.pkl.
    stem = persona_file[:-4] if persona_file.endswith(".pkl") else persona_file
    prefix = "" if stem == "persona_emb" else (stem[len("persona_emb_") :] if stem.startswith("persona_emb_") else stem) + "_"
    spec = f"auto-silhouette-k{clustering['k_min']}-{clustering['k_max']}-kind{clustering['kind_split']}_center_seed{clustering['seed']}"
    # The last five entries are constants of the method.  They stay in the key because a cached clustering is only
    # reused when its stored parameters equal this dict, and the caches of the reported runs carry them.
    cache_key = {**clustering, "expert_num": 0, "select": "silhouette", "center": True, "text_dim": 0, "cache": True}
    stat = persona_path.stat()
    fingerprint = {"size": int(stat.st_size), "mtime": int(stat.st_mtime), "records": int(personas.shape[0]), "dim": raw_dim}
    cache_path = data_dir / CLUSTER_CACHE_TEMPLATE.format(spec=prefix + spec)
    clusters = _cluster_personas(personas, cache_key, cache_path, fingerprint, _clustering_device(device))

    K = int(clusters["K"])
    centroids = np.asarray(clusters["centroids"], dtype=np.float32)
    labels = np.asarray(clusters["labels"], dtype=np.int64).reshape(-1)
    if centroids.shape != (K, raw_dim):
        raise ValueError(f"Cached centroids {centroids.shape} do not match K={K}, dim={raw_dim}.")
    if labels.shape[0] != personas.shape[0] or labels.min() < 0 or labels.max() >= K:
        raise ValueError(f"Cached cluster labels {labels.shape} do not match {personas.shape[0]} personas / K={K}.")

    # Text space: centred, unit-normalised embeddings.  Personas are centred by the persona mean stored with the
    # clustering, item profiles by their own mean (a different kind of text).  The last table row is the zero fallback.
    persona_mean = np.asarray(clusters["projection"]["mean"], dtype=np.float32).reshape(1, -1)
    fallback = np.zeros((1, raw_dim), dtype=np.float32)
    user_table = np.vstack([_l2_normalize_np(personas - persona_mean), fallback])
    item_table = np.vstack([_l2_normalize_np(profiles - profiles.mean(axis=0, keepdims=True)), fallback])
    user_index, user_sim, user_missing = _best_persona_per_cluster(user_table, _node_slots(owner, user_num), centroids)
    _, item_sim, _ = _best_persona_per_cluster(item_table, np.arange(item_num, dtype=np.int64)[:, None], centroids)

    bundle = SimpleNamespace(
        K=K,
        dim=raw_dim,
        centroids=centroids,
        labels=labels,
        clusters=clusters,
        user_table=user_table,  # [personas + 1, dim]
        user_owner=owner,
        user_index=user_index,  # [users, K] row of user_table: the user's persona closest to every cluster
        user_sim=user_sim,  # [users, K] cosine of that persona to the cluster centroid
        user_missing=user_missing,
        item_table=item_table,  # [items + 1, dim]
        item_sim=item_sim,  # [items, K]
        cache_path=str(cache_path),
    )
    _BUNDLE_MEMO[memo_key] = bundle
    return bundle


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class PriorProjector(nn.Module):
    """MLP^k: projects a node's text prior into the embedding space of expert k."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden_dim = max(output_dim, (input_dim + output_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                init(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x.float())


class LayerWisePriorModulator(nn.Module):
    """Prior-conditioned gate applied right before one LightGCN propagation step:
    g = 1 + lambda * tanh(LN(MLP(prior))), lambda a learnable scalar clamped to [0, max_scale]."""

    def __init__(self, embedding_size: int, max_scale: float, lambda_init: float, init_std: float):
        super().__init__()
        if max_scale < 0 or lambda_init < 0:
            raise ValueError("layer_modulation_scale and layer_modulation_lambda_init must be non-negative.")
        self.max_scale = float(max_scale)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Identity(),  # keeps the second linear layer at index 3, the parameter name in saved checkpoints
            nn.Linear(4 * embedding_size, embedding_size),
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.lambda_param = nn.Parameter(t.tensor(min(float(lambda_init), self.max_scale), dtype=t.float32))
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                if init_std > 0:
                    nn.init.normal_(module.weight, mean=0.0, std=float(init_std))
                else:
                    nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, prior: t.Tensor) -> t.Tensor:
        delta = self.norm(self.mlp(prior.float()))
        return 1.0 + t.clamp(self.lambda_param, min=0.0, max=self.max_scale) * t.tanh(delta)


class SharedSoftmaxTopKGate(nn.Module):
    """One softmax router shared by user and item nodes: logits = MLP(features), the top-k experts are kept and their
    probabilities renormalised."""

    def __init__(self, input_dim: int, hidden_dim: int, expert_num: int):
        super().__init__()
        self.expert_num = int(expert_num)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expert_num),
        )
        for module in self.gate:
            if isinstance(module, nn.Linear):
                init(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: t.Tensor, topk: int) -> Tuple[t.Tensor, t.Tensor]:
        """Returns (routing weights [nodes, K], zero outside the top-k; balance term K * sum_k importance_k * load_k)."""
        probs = F.softmax(self.gate(features.float()), dim=-1)
        topk = max(1, min(int(topk), self.expert_num))
        top_probs, top_idx = t.topk(probs, k=topk, dim=-1)
        top_weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-12)
        weights = t.zeros_like(probs)
        weights.scatter_(1, top_idx, top_weights)
        importance = probs.mean(dim=0)
        load = weights.mean(dim=0) if topk == self.expert_num else (weights > 0).float().mean(dim=0)
        return weights, self.expert_num * t.sum(importance * load)


class LightGCNExpert(nn.Module):
    """One MoE expert (ExpertGCN): a LightGCN whose embeddings are multiplied, at every layer and right before the
    propagation, by a gate computed from the node's projected persona prior.  The read-out is a softmax-weighted sum
    of the modulated layer representations."""

    def __init__(
        self,
        user_num: int,
        item_num: int,
        embedding_size: int,
        layer_num: int,
        modulation_scale: float,
        modulation_lambda_init: float,
        modulation_init_std: float,
    ):
        super().__init__()
        self.user_num = int(user_num)
        self.layer_num = int(layer_num)
        self.user_embeds = nn.Parameter(init(t.empty(user_num, embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(item_num, embedding_size)))
        self.layer_modulators = nn.ModuleList(
            [
                LayerWisePriorModulator(embedding_size, modulation_scale, modulation_lambda_init, modulation_init_std)
                for _ in range(self.layer_num + 1)
            ]
        )
        self.edge_dropper = SpAdjEdgeDrop()

    def forward(
        self,
        adj: t.Tensor,
        keep_rate: float,
        user_prior: t.Tensor,
        item_prior: t.Tensor,
        layer_logits: t.Tensor,
        cl_views: int = 0,
        cl_eps: float = 0.0,
    ) -> Tuple[t.Tensor, t.Tensor, List[Tuple[t.Tensor, t.Tensor]]]:
        """Returns (user_out, item_out, views).  views = ``cl_views`` extra read-outs of the same propagation with
        SimGCL-type noise sign(E) * eps * unit(U(0, 1)) added after every propagation step.  The layer gates and the
        read-out weights are constants in the views, so a contrast between them trains only the free embeddings."""
        embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
        priors = t.cat([user_prior, item_prior], dim=0)
        prop_adj = self.edge_dropper(adj, keep_rate) if self.training else adj
        start_embeds = embeds
        gates, terms = [], []
        for layer_idx in range(self.layer_num + 1):
            gate = self.layer_modulators[layer_idx](priors)
            gates.append(gate)
            modulated = gate * embeds
            terms.append(modulated)
            if layer_idx < self.layer_num:
                embeds = t.spmm(prop_adj, modulated)
        alpha = F.softmax(layer_logits.reshape(-1), dim=0)
        out = sum(alpha[layer_idx] * terms[layer_idx] for layer_idx in range(self.layer_num + 1))

        views = []
        if cl_views > 0:
            gates = [gate.detach() for gate in gates]
            alpha = alpha.detach()
            for _ in range(cl_views):
                embeds = start_embeds
                terms = []
                for layer_idx in range(self.layer_num + 1):
                    modulated = gates[layer_idx] * embeds
                    terms.append(modulated)
                    if layer_idx < self.layer_num:
                        embeds = t.spmm(prop_adj, modulated)
                        embeds = embeds + t.sign(embeds) * F.normalize(t.rand_like(embeds), dim=-1) * cl_eps
                view = sum(alpha[layer_idx] * terms[layer_idx] for layer_idx in range(self.layer_num + 1))
                views.append((view[: self.user_num], view[self.user_num :]))
        return out[: self.user_num], out[self.user_num :], views


def _view_infonce(a: t.Tensor, b: t.Tensor, tau: float) -> t.Tensor:
    """InfoNCE between two views of the same nodes (rows aligned): -log softmax over the in-batch nodes of the other
    view, cosine / tau, mean over rows."""
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    pos = (a * b).sum(dim=-1) / tau
    logits = a @ b.T / tau
    return (t.logsumexp(logits, dim=1) - pos).mean()


def _weighted_infonce(embeds: t.Tensor, ids: t.Tensor, prior_all: t.Tensor, weights: t.Tensor, temp: float) -> t.Tensor:
    """Sum over the rows with weight > 0 of weight * InfoNCE(embed_row, prior[id_row] | all priors)."""
    rows = (weights > 0).nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return embeds.new_tensor(0.0)
    anchor = F.normalize(embeds.index_select(0, rows), dim=-1, eps=1e-8)
    prior_all = F.normalize(prior_all, dim=-1, eps=1e-8)
    positive = prior_all.index_select(0, ids.index_select(0, rows))
    pos_logit = (anchor * positive).sum(dim=-1) / temp
    all_logits = anchor @ prior_all.T / temp
    per_row = t.logsumexp(all_logits, dim=-1) - pos_logit
    return (weights.index_select(0, rows) * per_row).sum()


# ---------------------------------------------------------------------------
# PersonaMosaic recommender
# ---------------------------------------------------------------------------

_SUMMARY_PRINTED = set()


class PersonaMosaic(BaseModel):
    """Mixture of K persona-cluster LightGCN experts with persona priors, persona-similarity routing and a fixed
    persona-content score at prediction."""

    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.dataset = configs["data"]["name"]
        data_dir = Path(configs["data"]["dir"])  # data/<dataset>, set by config/configurator.py
        trn_mat = _as_csr_binary(data_handler.trn_mat)

        self.keep_rate = float(self._hp("keep_rate"))
        self.layer_num = int(self._hp("layer_num"))
        self.top_k = int(self._hp("top_k"))
        self.reg_weight = float(self._hp("reg_weight"))
        self.balance_loss_weight = float(self._hp("balance_loss_weight"))
        self.persona_align_weight = float(self._hp("persona_align_weight"))
        self.llm_align_temperature = float(self._hp("llm_align_temperature"))
        self.graph_cl_weight = float(self._hp("graph_cl_weight"))
        self.graph_cl_eps = float(self._hp("graph_cl_eps"))
        self.graph_cl_temperature = float(self._hp("graph_cl_temperature"))
        self.content_score_weight = float(self._hp("content_score_weight"))
        for name in ("balance_loss_weight", "persona_align_weight", "graph_cl_weight", "graph_cl_eps", "content_score_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")
        if self.llm_align_temperature <= 0 or self.graph_cl_temperature <= 0:
            raise ValueError("llm_align_temperature and graph_cl_temperature must be > 0.")

        # ----- Phase 1: persona clusters -> experts, per-node text tables, cluster similarities -----
        self.persona_file = str(self._hp("persona_file"))
        clustering = {
            "kind_split": int(self._hp("clustering_kind_split")),
            "k_min": int(self._hp("clustering_k_min")),
            "k_max": int(self._hp("clustering_k_max")),
            "seed": int(self._hp("clustering_seed")),
            "n_init": int(self._hp("clustering_n_init")),
            "max_iter": int(self._hp("clustering_max_iter")),
            "silhouette_sample": int(self._hp("clustering_silhouette_sample")),
        }
        if clustering["kind_split"] < 2:
            raise ValueError("clustering_kind_split must be >= 2 (number of persona kinds split in stage 1).")
        if not 1 <= clustering["k_min"] <= clustering["k_max"]:
            raise ValueError("clustering_k_min / clustering_k_max must satisfy 1 <= k_min <= k_max.")
        bundle = _build_persona_bundle(
            data_dir, self.user_num, self.item_num, self.persona_file, clustering, str(self._hp("clustering_device")).lower()
        )
        self.expert_num = int(bundle.K)
        if not 1 <= self.top_k <= self.expert_num:
            raise ValueError(f"top_k must be in [1, {self.expert_num}] (K persona clusters), got {self.top_k}.")
        # persona (.) centroid of unit vectors has norm ~ 1 / sqrt(dim); rescale it to unit order.
        self._product_scale = math.sqrt(float(bundle.dim))
        self.register_buffer("cluster_centroids", t.from_numpy(bundle.centroids), persistent=False)
        self.register_buffer("user_persona_table", t.from_numpy(bundle.user_table), persistent=False)
        self.register_buffer("user_persona_index", t.from_numpy(bundle.user_index), persistent=False)
        self.register_buffer("item_text_table", t.from_numpy(bundle.item_table[:-1]), persistent=False)

        # ----- Gate: the node's cosine similarity to every cluster (z-scored) -> shared softmax top-k gate -----
        user_gate_features, item_gate_features = _combined_zscore(bundle.user_sim, bundle.item_sim)
        self.register_buffer("user_gate_features", t.from_numpy(user_gate_features), persistent=False)
        self.register_buffer("item_gate_features", t.from_numpy(item_gate_features), persistent=False)
        gate_hidden = int(self._hp("gate_hidden_size"))
        rng_state = t.random.get_rng_state()
        self.gate = SharedSoftmaxTopKGate(self.expert_num, gate_hidden, self.expert_num)
        # Seed compatibility with the reported runs: there the modules below were initialised from the random stream as
        # it stands after drawing a gate with 4 extra input columns.  Replaying those draws keeps a given seed on the
        # same initial experts / projectors; it has no other effect.
        t.random.set_rng_state(rng_state)
        SharedSoftmaxTopKGate(4 + self.expert_num, gate_hidden, self.expert_num)

        # ----- Phase 2: K ExpertGCNs, each with its own prior projector -----
        self.prior_projectors = nn.ModuleList(
            [PriorProjector(bundle.dim, self.embedding_size) for _ in range(self.expert_num)]
        )
        self.lightgcn_experts = nn.ModuleList(
            [
                LightGCNExpert(
                    self.user_num,
                    self.item_num,
                    self.embedding_size,
                    self.layer_num,
                    float(self._hp("layer_modulation_scale")),
                    float(self._hp("layer_modulation_lambda_init")),
                    float(self._hp("layer_modulation_init_std")),
                )
                for _ in range(self.expert_num)
            ]
        )
        self.layer_logits = nn.Parameter(t.zeros(self.expert_num, self.layer_num + 1))

        # ----- Phase 3: fixed content score (prediction only; a non-persistent table, nothing trained) -----
        content_note = ""
        if self.content_score_weight > 0:
            content_rows, content_note = _taste_content_rows(bundle, trn_mat, self.user_num)
            self.register_buffer("content_user_rows", t.from_numpy(content_rows), persistent=False)

        self._cached_embeds = None
        self._log_summary(bundle, clustering, content_note)

    # ------------------------------------------------------------------ config / logging helpers

    def _hp(self, key: str, default=_REQUIRED):
        """Hyper-parameter: the dataset block of the yml (also where --override lands), then the model block."""
        if key in self.hyper_config:
            return self.hyper_config[key]
        if key in configs["model"]:
            return configs["model"][key]
        if default is _REQUIRED:
            raise KeyError(
                f"persona_mosaic.yml has no '{key}' for dataset {configs['data']['name']!r} "
                "(add it to the dataset block or the model block, or pass --override)."
            )
        return default

    def _log_summary(self, bundle: SimpleNamespace, clustering: Dict, content_note: str) -> None:
        kinds = bundle.clusters["kinds"]
        summary = (
            f"K={self.expert_num} = union of per-kind clusters {[kd['K'] for kd in kinds]} (stage 1: k-means split into "
            f"{clustering['kind_split']} persona kinds, sizes {[kd['size'] for kd in kinds]}; stage 2: silhouette over "
            f"K={clustering['k_min']}..{clustering['k_max']} inside each kind, per-kind silhouette "
            f"{[round(float(kd['silhouette'].get(kd['K'], float('nan'))), 4) for kd in kinds]}); "
            f"sizes={bundle.clusters.get('sizes')}; user personas={bundle.user_table.shape[0] - 1} from {self.persona_file} "
            f"(users without persona={int(bundle.user_missing.sum())}); items=RLMRec profile {ITEM_PROFILE_FILE}; "
            f"{bundle.dim}-d text embeddings; gate input={self.expert_num} cluster similarities; top_k={self.top_k}; "
            f"balance weight={self.balance_loss_weight:g}; align weight={self.persona_align_weight:g} tau={self.llm_align_temperature:g}; "
            f"graph CL w={self.graph_cl_weight:g} eps={self.graph_cl_eps:g} tau={self.graph_cl_temperature:g}"
            + (
                f"; content score b={self.content_score_weight:g} (fixed, prediction only, per-user standardised; {content_note})"
                if self.content_score_weight > 0
                else ""
            )
            + f"; cache={bundle.cache_path}"
        )
        configs["model"]["persona_clusters"] = summary  # picked up by the Logger's 'Model|' line
        key = (self.dataset, summary)
        if key in _SUMMARY_PRINTED:
            return
        _SUMMARY_PRINTED.add(key)
        print(f"[persona_mosaic] {self.dataset}: {summary}")
        for kd in kinds:
            curve = " ".join(f"{k}:{v:.4f}" for k, v in sorted(kd["silhouette"].items()))
            print(f"[persona_mosaic] silhouette curve (kind {kd['kind']}, n={kd['size']}, K={kd['K']}): {curve}")

    # ------------------------------------------------------------------ MoE forward

    def _prior(self, node_type: str, expert_idx: int, prior_cache: PriorCache) -> t.Tensor:
        """Projected persona prior x^k_v of every user / item under expert k (computed once per forward pass)."""
        key = (node_type, expert_idx)
        if key not in prior_cache:
            with t.no_grad():
                if node_type == "user":  # the user's persona closest to cluster k
                    rows = self.user_persona_table.index_select(0, self.user_persona_index[:, expert_idx])
                else:  # an item has one text vector
                    rows = self.item_text_table
                prior_input = rows * self.cluster_centroids[expert_idx].unsqueeze(0) * self._product_scale
            prior_cache[key] = self.prior_projectors[expert_idx](prior_input)
        return prior_cache[key]

    def _expert_outputs(self, keep_rate: float, prior_cache: PriorCache, cl_views: int = 0):
        """Per-expert outputs of every node: (user list, item list, user views, item views), views[v] = list over experts."""
        user_outputs, item_outputs = [], []
        user_views = [[] for _ in range(cl_views)]
        item_views = [[] for _ in range(cl_views)]
        for expert_idx, expert in enumerate(self.lightgcn_experts):
            user_out, item_out, views = expert(
                self.adj,
                keep_rate,
                self._prior("user", expert_idx, prior_cache),
                self._prior("item", expert_idx, prior_cache),
                self.layer_logits[expert_idx],
                cl_views=cl_views,
                cl_eps=self.graph_cl_eps,
            )
            user_outputs.append(user_out)
            item_outputs.append(item_out)
            for v, (user_view, item_view) in enumerate(views):
                user_views[v].append(user_view)
                item_views[v].append(item_view)
        return user_outputs, item_outputs, user_views, item_views

    def _route(self, expert_outputs: List[t.Tensor], weights: t.Tensor, node_ids: Optional[t.Tensor] = None) -> t.Tensor:
        """sum_k w^k_v * (output of expert k for v), for the nodes ``node_ids`` (every node when None)."""
        routed = expert_outputs[0].new_zeros((weights.shape[0], self.embedding_size))
        for expert_idx, output in enumerate(expert_outputs):
            if node_ids is not None:
                output = output.index_select(0, node_ids)
            routed = routed + weights[:, expert_idx : expert_idx + 1] * output
        return routed

    def _final_embeddings(self) -> Tuple[t.Tensor, t.Tensor]:
        """Routed embeddings of every user and item; cached in eval mode until the next training step."""
        if self._cached_embeds is not None and not self.training:
            return self._cached_embeds
        user_outputs, item_outputs, _, _ = self._expert_outputs(1.0, {})
        w_user, _ = self.gate(self.user_gate_features, self.top_k)
        w_item, _ = self.gate(self.item_gate_features, self.top_k)
        embeds = (self._route(user_outputs, w_user), self._route(item_outputs, w_item))
        if not self.training:
            self._cached_embeds = embeds
        return embeds

    # ------------------------------------------------------------------ losses

    def cal_loss(self, batch_data):
        ancs, poss, negs = (x.long() for x in batch_data[:3])
        self._cached_embeds = None
        prior_cache: PriorCache = {}
        cl_views = 2 if self.graph_cl_weight > 0 else 0
        user_outputs, item_outputs, user_views, item_views = self._expert_outputs(self.keep_rate, prior_cache, cl_views)
        w_user, b_user = self.gate(self.user_gate_features, self.top_k)
        w_item, b_item = self.gate(self.item_gate_features, self.top_k)

        # route only the nodes of the batch
        batch_users, user_inverse = t.unique(ancs, sorted=False, return_inverse=True)
        batch_items, item_inverse = t.unique(t.cat([poss, negs], dim=0), sorted=False, return_inverse=True)
        final_users = self._route(user_outputs, w_user.index_select(0, batch_users), batch_users)
        final_items = self._route(item_outputs, w_item.index_select(0, batch_items), batch_items)
        user_embeds = final_users.index_select(0, user_inverse)
        pos_embeds = final_items.index_select(0, item_inverse[: poss.shape[0]])
        neg_embeds = final_items.index_select(0, item_inverse[poss.shape[0] :])

        pos_scores = (user_embeds * pos_embeds).sum(dim=-1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1)
        bpr_loss = F.softplus(neg_scores - pos_scores).sum() / ancs.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        balance_loss = self.balance_loss_weight * (0.5 * (b_user + b_item))
        persona_align_loss = self._persona_alignment_loss(
            ancs, poss, negs, user_embeds, pos_embeds, neg_embeds, w_user.detach(), w_item.detach(), prior_cache
        )
        graph_cl_loss = user_embeds.new_tensor(0.0)
        if cl_views > 0:
            # Text-free graph contrast: the two noisy views of the batch users and of the positive items, routed with
            # constant (detached) gate weights; users and items separately, no cross term.
            cl_items = t.unique(poss, sorted=False)
            w_user_cl = w_user.detach().index_select(0, batch_users)
            w_item_cl = w_item.detach().index_select(0, cl_items)
            user_pair = [self._route(view, w_user_cl, batch_users) for view in user_views]
            item_pair = [self._route(view, w_item_cl, cl_items) for view in item_views]
            graph_cl_loss = self.graph_cl_weight * (
                graph_cl_loss
                + _view_infonce(user_pair[0], user_pair[1], self.graph_cl_temperature)
                + _view_infonce(item_pair[0], item_pair[1], self.graph_cl_temperature)
            )

        loss = bpr_loss + reg_loss + balance_loss + persona_align_loss + graph_cl_loss
        losses = {"bpr_loss": bpr_loss, "reg_loss": reg_loss, "balance_loss": balance_loss}
        if self.persona_align_weight > 0:
            losses["persona_align_loss"] = persona_align_loss
        if self.graph_cl_weight > 0:
            losses["graph_cl_loss"] = graph_cl_loss
        return loss, losses

    def _persona_alignment_loss(
        self,
        ancs: t.Tensor,
        poss: t.Tensor,
        negs: t.Tensor,
        user_embeds: t.Tensor,
        pos_embeds: t.Tensor,
        neg_embeds: t.Tensor,
        w_user: t.Tensor,
        w_item: t.Tensor,
        prior_cache: PriorCache,
    ) -> t.Tensor:
        """Gate-weighted InfoNCE between the routed output and the persona prior of each selected expert.

        For node v with gate weights w^k_v (top-k, summing to one, constants here) the term is
        sum_k w^k_v * InfoNCE(y_v, x^k_v | {x^k_u}), so the output stays aligned with the priors of the experts that
        produced it.  Summed over the batch users, positives and negatives and divided by the batch size.
        """
        if self.persona_align_weight <= 0:
            return user_embeds.new_tensor(0.0)
        temp = self.llm_align_temperature
        w_anc = w_user.index_select(0, ancs)
        w_pos = w_item.index_select(0, poss)
        w_neg = w_item.index_select(0, negs)
        loss = user_embeds.new_tensor(0.0)
        for expert_idx in range(self.expert_num):
            if bool((w_anc[:, expert_idx] > 0).any()):
                user_prior = self._prior("user", expert_idx, prior_cache)
                loss = loss + _weighted_infonce(user_embeds, ancs, user_prior, w_anc[:, expert_idx], temp)
            if bool((w_pos[:, expert_idx] > 0).any()) or bool((w_neg[:, expert_idx] > 0).any()):
                item_prior = self._prior("item", expert_idx, prior_cache)
                loss = loss + _weighted_infonce(pos_embeds, poss, item_prior, w_pos[:, expert_idx], temp)
                loss = loss + _weighted_infonce(neg_embeds, negs, item_prior, w_neg[:, expert_idx], temp)
        return self.persona_align_weight * loss / max(int(ancs.shape[0]), 1)

    # ------------------------------------------------------------------ inference

    @t.no_grad()
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        user_embeds, item_embeds = self._final_embeddings()
        scores = user_embeds.index_select(0, pck_users) @ item_embeds.T
        if self.content_score_weight > 0:
            # fixed content score: s + b * std(s_u.) * z(c_u.); a user without a content row (zero row) keeps the model ranking
            content = self.content_user_rows.index_select(0, pck_users) @ self.item_text_table.T
            content = (content - content.mean(dim=1, keepdim=True)) / content.std(dim=1, keepdim=True).clamp_min(1e-8)
            scores = scores + self.content_score_weight * scores.std(dim=1, keepdim=True) * content
        return self._mask_predict(scores, train_mask)


# models/bulid_model.py looks up a class whose lower-cased name equals the model name 'persona_mosaic'.
Persona_Mosaic = PersonaMosaic

__all__ = ["PersonaMosaic", "Persona_Mosaic"]
