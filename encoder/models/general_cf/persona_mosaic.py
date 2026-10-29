"""PERSONA-MOSAIC: MOSAIC whose experts are LLM persona clusters.

Differences from ``mosaic.py`` (the topology gate features, the shared softmax top-k gate, the
ExpertGCN = prior-modulated LightGCN expert, and the loss terms are identical):

* LLM response  - every user has 1..n LLM-written personas (``data/<dataset>/persona/persona_emb.pkl``,
                  ``record_ids`` 'u<uid>:p<j>', ``field_mean`` [records, dim]) instead of a global
                  intent / conformity description.  Items have no facets; their single text signal is set
                  by ``item_text``: ``profile`` = the RLMRec item profile embedding
                  (``data/<dataset>/itm_emb_np.pkl``, [items, dim], the same embedder as the personas) or
                  ``persona`` = one LLM-written persona per item (``data/<dataset>/persona/item_persona_emb.pkl``,
                  ``emb`` [items, dim] in the user persona text space, ``valid`` [items]).
* Expert number - the experts are persona clusters: k-means over every user persona, with K chosen
                  without validation data within [clustering_k_min, clustering_k_max] (``expert_num: 0``, rule
                  ``clustering_k_select``: silhouette maximum or SSE elbow) or fixed (``expert_num: K``).
                  One ExpertGCN per cluster.
* Prior         - expert k receives, for node v, the persona of v that is closest to cluster k composed
                  with the cluster centroid (``prior_composition``: persona (.) centroid, persona only,
                  or centroid only) and projects it with its own MLP^k.  An item's single text vector is its
                  closest persona for every cluster.  ``profile``: centred by the profile mean and mapped into
                  the clustering / prior spaces of the user personas.  ``persona``: centred, projected and
                  normalised exactly like the user personas; items whose persona generation failed (``valid``
                  false) use the zero fallback persona, as users without personas do.
* Gate input    - MOSAIC's 4 topology features, concatenated with the node's cosine similarity to each
                  cluster (max over its personas) when ``gate_persona_similarity`` is true.
* Text dimension- d_text, the space of the k-means, of each node's closest-persona choice and of the gate's
                  cluster similarities (``text_dim``, CLI ``--d_text``).  The default is the raw text embedding
                  dimension (4096; ``text_dim: 0`` or ``text_dim: 4096`` = no PCA).  A smaller value, e.g. 32,
                  maps the embeddings into d_text dimensions with a PCA fitted on the user persona pool.
* Control       - ``persona_shuffle`` (global / within_cluster) and ``item_shuffle`` permute which node owns which
                  text vector (facets stay together); the clustering is unchanged.  Trained from scratch, they
                  measure how much of the accuracy comes from the persona *content* (see _shuffle_user_owners).
* Prior space   - ``prior_space: raw`` (default) feeds MLP^k with the raw text embedding (centred, unit-
                  normalised, 4096-d) of the selected persona and the raw-space centroid of cluster k (unit
                  mean of its members), so MLP^k projects R^4096 -> R^d as in MOSAIC; ``prior_space: pca``
                  uses the d_text PCA vectors and centroids instead (MLP^k: R^{d_text} -> R^d).

Persona vectors are mean-centred, PCA-projected (when d_text is smaller than the raw dimension) and
re-normalised before clustering / similarity; centring removes the dominant common direction of LLM
embeddings (``persona_center`` only matters for the raw path).  The clustering (PCA projection, centroids, K,
silhouette and SSE curves) is cached in ``data/<dataset>/persona/persona_clusters_<spec>.pkl`` so that
every model build of a run, and every later run, uses the same experts.
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
from typing import Dict, Optional, Tuple

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
PriorCache = Dict[Tuple[str, int, str], t.Tensor]

PERSONA_FILE = "persona/persona_emb.pkl"
ITEM_PROFILE_FILE = "itm_emb_np.pkl"  # RLMRec item profile embeddings (configs['itmprf_embeds'])
ITEM_PERSONA_FILE = "persona/item_persona_emb.pkl"  # one LLM persona per item: {'emb', 'valid', 'view_names'}
ITEM_TEXTS = ("profile", "persona")
CLUSTER_CACHE_TEMPLATE = "persona/persona_clusters_{spec}.pkl"
PRIOR_COMPOSITIONS = ("product", "persona", "cluster")
PRIOR_SPACES = ("raw", "pca")
PERSONA_SHUFFLES = ("none", "global", "within_cluster")  # control experiment: who owns which persona set


# ---------------------------------------------------------------------------
# Data / feature helpers (identical to mosaic.py)
# ---------------------------------------------------------------------------


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _as_csr_binary(mat) -> sp.csr_matrix:
    if not sp.issparse(mat):
        mat = sp.csr_matrix(mat)
    mat = mat.tocsr()
    return (mat != 0).astype(np.float32)


def _load_train_matrix(data_handler, data_dir: Path) -> sp.csr_matrix:
    """Use only trn_mat.pkl / data_handler.trn_mat as the observed graph."""
    if hasattr(data_handler, "trn_mat"):
        return _as_csr_binary(data_handler.trn_mat)

    trn_path = data_dir / "trn_mat.pkl"
    if not trn_path.exists():
        raise FileNotFoundError(f"Missing training interaction matrix: {trn_path}")
    return _as_csr_binary(_load_pickle(trn_path))


def _load_community_assignments(
    data_dir: Path,
    dataset: str,
    user_num: int,
    item_num: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    cache_path = data_dir / "community_assignments.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing community assignments for {dataset}: {cache_path}. "
            "PersonaMosaic needs user_community and item_community for the topology gate features."
        )

    cached = _load_pickle(cache_path)
    if not isinstance(cached, dict):
        raise ValueError(f"Community assignment cache must be a dict: {cache_path}")

    required = {"user_community", "item_community"}
    missing = sorted(required - set(cached.keys()))
    if missing:
        raise ValueError(f"Community assignment cache is missing keys {missing}: {cache_path}")

    user_comm = np.asarray(cached["user_community"], dtype=np.int64).reshape(-1)
    item_comm = np.asarray(cached["item_community"], dtype=np.int64).reshape(-1)
    if user_comm.shape[0] != user_num or item_comm.shape[0] != item_num:
        raise ValueError(
            f"Community assignment shape mismatch for {dataset}: "
            f"user/item lengths={(user_comm.shape[0], item_comm.shape[0])}, "
            f"expected={(user_num, item_num)}, path={cache_path}"
        )
    if user_comm.size and int(user_comm.min()) < 0:
        raise ValueError(f"user_community contains negative labels in {cache_path}")
    if item_comm.size and int(item_comm.min()) < 0:
        raise ValueError(f"item_community contains negative labels in {cache_path}")

    inferred = int(max(user_comm.max(initial=0), item_comm.max(initial=0))) + 1
    declared = int(cached.get("num_communities") or inferred)
    if declared < inferred:
        raise ValueError(
            f"num_communities={declared} is smaller than max community label "
            f"{inferred - 1} in {cache_path}"
        )

    metadata = {
        **(cached.get("metadata") or {}),
        "dataset": cached.get("dataset", dataset),
        "num_communities": declared,
        "source": str(cache_path),
    }
    return user_comm, item_comm, metadata


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=np.float32)
    np.divide(numer, denom, out=out, where=denom > 0)
    return out.astype(np.float32)


def _combined_zscore(user_features: np.ndarray, item_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    all_features = np.vstack([user_features, item_features]).astype(np.float32)
    mean = all_features.mean(axis=0, keepdims=True)
    std = all_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    user_norm = (user_features - mean) / std
    item_norm = (item_features - mean) / std
    return user_norm.astype(np.float32), item_norm.astype(np.float32)


def _participation_from_neighbor_communities(
    node_ids: np.ndarray,
    neighbor_comms: np.ndarray,
    node_count: int,
    degree: np.ndarray,
) -> np.ndarray:
    """Participation coefficient: 1 - sum_c (k_ic / k_i)^2."""
    buckets = [dict() for _ in range(node_count)]
    for node, comm in zip(node_ids, neighbor_comms):
        node = int(node)
        comm = int(comm)
        buckets[node][comm] = buckets[node].get(comm, 0) + 1

    participation = np.zeros(node_count, dtype=np.float32)
    for node, counts in enumerate(buckets):
        if degree[node] <= 0 or not counts:
            continue
        probs = np.asarray(list(counts.values()), dtype=np.float32) / float(degree[node])
        participation[node] = 1.0 - float(np.square(probs).sum())
    return participation


def _build_gate_features(
    trn_mat: sp.csr_matrix,
    user_comm: np.ndarray,
    item_comm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build 4-dim user/item topology inputs for the single shared gate (as in MOSAIC).

    User features:
      1) degree
      2) popularity bias: mean popularity of interacted items
      3) within-community interaction ratio
      4) participation across item communities

    Item features:
      1) popularity
      2) audience activity: mean activity degree of interacting users
      3) within-community interaction ratio
      4) participation across user communities
    """
    trn_mat = _as_csr_binary(trn_mat)
    coo = trn_mat.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)

    user_degree = np.asarray(trn_mat.sum(axis=1)).reshape(-1).astype(np.float32)
    item_popularity = np.asarray(trn_mat.sum(axis=0)).reshape(-1).astype(np.float32)

    # User popularity bias: average item popularity over observed interactions.
    user_pop_sum = np.zeros(trn_mat.shape[0], dtype=np.float32)
    np.add.at(user_pop_sum, rows, item_popularity[cols])
    user_pop_bias = _safe_divide(user_pop_sum, user_degree)

    # Item audience activity: average user degree over observed audience.
    item_audience_sum = np.zeros(trn_mat.shape[1], dtype=np.float32)
    np.add.at(item_audience_sum, cols, user_degree[rows])
    item_audience_activity = _safe_divide(item_audience_sum, item_popularity)

    same_community = (user_comm[rows] == item_comm[cols]).astype(np.float32)
    user_same = np.zeros(trn_mat.shape[0], dtype=np.float32)
    item_same = np.zeros(trn_mat.shape[1], dtype=np.float32)
    np.add.at(user_same, rows, same_community)
    np.add.at(item_same, cols, same_community)
    user_within_ratio = _safe_divide(user_same, user_degree)
    item_within_ratio = _safe_divide(item_same, item_popularity)

    user_participation = _participation_from_neighbor_communities(
        rows,
        item_comm[cols],
        trn_mat.shape[0],
        user_degree,
    )
    item_participation = _participation_from_neighbor_communities(
        cols,
        user_comm[rows],
        trn_mat.shape[1],
        item_popularity,
    )

    user_features = np.stack(
        [
            np.log1p(user_degree),
            np.log1p(user_pop_bias),
            user_within_ratio,
            user_participation,
        ],
        axis=1,
    )
    item_features = np.stack(
        [
            np.log1p(item_popularity),
            np.log1p(item_audience_activity),
            item_within_ratio,
            item_participation,
        ],
        axis=1,
    )
    return _combined_zscore(user_features, item_features)


# ---------------------------------------------------------------------------
# Persona records
# ---------------------------------------------------------------------------

_RECORD_RE = re.compile(r"^([ui])(\d+):p(\d+)$")


class PersonaRecords:
    """Flat persona table of one node type: ``embeddings[r]`` belongs to node ``owner[r]``."""

    def __init__(self, embeddings: np.ndarray, owner: np.ndarray):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.owner = np.asarray(owner, dtype=np.int64).reshape(-1)
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != self.owner.shape[0]:
            raise ValueError(
                f"Persona embeddings must be [records, dim] with one owner per record, got "
                f"{self.embeddings.shape} and {self.owner.shape}."
            )

    @property
    def num_records(self) -> int:
        return int(self.owner.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


def _parse_persona_pickle(path: Path) -> Tuple[Optional[PersonaRecords], Optional[PersonaRecords], Dict]:
    """Read a persona generation pickle: {'record_ids': ['u0:p0', ...], 'field_mean': [records, dim], ...}."""
    obj = _load_pickle(path)
    if not isinstance(obj, dict) or "record_ids" not in obj or "field_mean" not in obj:
        raise ValueError(f"{path} must be a dict with 'record_ids' and 'field_mean' (persona generation output).")
    record_ids = [str(r) for r in obj["record_ids"]]
    emb = np.asarray(obj["field_mean"], dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(record_ids):
        raise ValueError(
            f"'field_mean' must be [records, dim] aligned with 'record_ids' in {path}: "
            f"{emb.shape} vs {len(record_ids)} ids."
        )
    if not np.isfinite(emb).all():
        raise ValueError(f"Persona embeddings contain NaN/inf: {path}")

    kinds = np.empty(len(record_ids), dtype="<U1")
    owners = np.empty(len(record_ids), dtype=np.int64)
    for r, rid in enumerate(record_ids):
        match = _RECORD_RE.match(rid)
        if match is None:
            raise ValueError(
                f"Unrecognised persona record id {rid!r} in {path}; expected 'u<uid>:p<j>' or 'i<iid>:p<j>'."
            )
        kinds[r] = match.group(1)
        owners[r] = int(match.group(2))

    user_mask = kinds == "u"
    item_mask = kinds == "i"
    if "uids" in obj and user_mask.any():
        uids = np.asarray(obj["uids"]).reshape(-1)
        if uids.shape[0] == len(record_ids) and not np.array_equal(uids[user_mask], owners[user_mask]):
            raise ValueError(f"'uids' disagree with the user ids parsed from 'record_ids' in {path}.")

    users = PersonaRecords(emb[user_mask], owners[user_mask]) if user_mask.any() else None
    items = PersonaRecords(emb[item_mask], owners[item_mask]) if item_mask.any() else None
    meta = {"view_names": list(obj.get("view_names", [])), "path": str(path)}
    return users, items, meta


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


def _load_item_personas(data_dir: Path, item_num: int, raw_dim: int) -> Tuple[np.ndarray, np.ndarray, Path]:
    """One LLM persona per item: (embeddings [item_num, raw_dim], valid [item_num] bool, path).

    ``item_persona_emb.pkl`` = {'emb': [items, dim] (row = item id), 'valid': [items] bool, 'view_names': [...]}.
    """
    path = data_dir / ITEM_PERSONA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Missing LLM item persona embeddings: {path}. PersonaMosaic needs persona/item_persona_emb.pkl."
        )
    obj = _load_pickle(path)
    if not isinstance(obj, dict) or "emb" not in obj:
        raise ValueError(f"{path} must be a dict with 'emb' [items, dim] (and optionally 'valid' [items]).")
    emb = np.asarray(obj["emb"], dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != item_num or emb.shape[1] != raw_dim:
        raise ValueError(
            f"Item persona embeddings must be [{item_num}, {raw_dim}] to live in the user persona text space, "
            f"got {emb.shape} from {path}."
        )
    valid = np.asarray(obj.get("valid", np.ones(item_num, dtype=bool)), dtype=bool).reshape(-1)
    if valid.shape[0] != item_num:
        raise ValueError(f"'valid' must have {item_num} entries, got {valid.shape[0]} in {path}.")
    if not np.isfinite(emb[valid]).all():
        raise ValueError(f"Item persona embeddings contain NaN/inf: {path}")
    emb = np.where(valid[:, None], emb, 0.0).astype(np.float32)  # invalid rows are never read (fallback row)
    return emb, valid, path


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


class TextReducer:
    """Linear map of LLM text embeddings into the d_text space used for clustering, priors and similarities.

    ``mean`` [1, raw_dim] is subtracted first (the user persona mean; zeros when centring is off and no PCA is
    used), then the vector is projected with ``components`` [raw_dim, d_text] (PCA fitted on the user persona
    pool; ``None`` keeps the raw dimension) and L2-normalised.
    """

    def __init__(self, mean: np.ndarray, components: Optional[np.ndarray]):
        self.mean = np.ascontiguousarray(mean, dtype=np.float32).reshape(1, -1)
        self.components = None if components is None else np.ascontiguousarray(components, dtype=np.float32)
        if self.components is not None and (
            self.components.ndim != 2 or self.components.shape[0] != self.mean.shape[1]
        ):
            raise ValueError(
                f"PCA components {None if self.components is None else self.components.shape} do not match "
                f"the text embedding dim {self.mean.shape[1]}."
            )

    @property
    def raw_dim(self) -> int:
        return int(self.mean.shape[1])

    @property
    def dim(self) -> int:
        return int(self.components.shape[1]) if self.components is not None else self.raw_dim

    def transform(self, x: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.raw_dim:
            raise ValueError(f"Expected [n, {self.raw_dim}] text embeddings, got {x.shape}.")
        y = x - (self.mean if mean is None else np.asarray(mean, dtype=np.float32).reshape(1, -1))
        if self.components is not None:
            y = y @ self.components
        return _l2_normalize_np(y)

    def state(self) -> Dict:
        return {"mean": self.mean, "components": self.components}

    @classmethod
    def from_state(cls, state: Dict) -> "TextReducer":
        return cls(state["mean"], state.get("components"))


def _fit_text_reducer(
    embeddings: np.ndarray, text_dim: int, center: bool, device: t.device
) -> Tuple[TextReducer, float]:
    """PCA of the user persona pool to ``text_dim`` dims (``text_dim`` <= 0 or == raw dim: identity, optional centring).

    Returns the reducer and the fraction of the pool variance kept by the ``text_dim`` leading components
    (1.0 for the identity).  The eigen-decomposition runs in float64 and every eigenvector sign is fixed
    (largest-magnitude entry positive) so that CPU and GPU fits agree.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Text embeddings must be [records, dim], got {emb.shape}.")
    n, raw_dim = emb.shape
    if text_dim > raw_dim:
        raise ValueError(f"text_dim (d_text) cannot exceed the text embedding dim {raw_dim}, got {text_dim}.")
    if text_dim <= 0 or text_dim == raw_dim:  # raw space: no PCA
        mean = emb.mean(axis=0, keepdims=True) if center else np.zeros((1, raw_dim), dtype=np.float32)
        return TextReducer(mean, None), 1.0
    if text_dim >= n:
        raise ValueError(f"text_dim (d_text) must be smaller than the persona pool size {n}, got {text_dim}.")
    mean = emb.mean(axis=0, keepdims=True)
    X = t.from_numpy(emb - mean).to(device=device, dtype=t.float64)
    cov = (X.T @ X) / float(max(n - 1, 1))
    del X
    evals, evecs = t.linalg.eigh(cov)  # ascending eigenvalues
    comps = evecs[:, -text_dim:].flip(1)  # leading text_dim components, descending variance
    evals = evals.clamp(min=0.0)
    total = float(evals.sum())
    explained = float(evals[-text_dim:].sum()) / total if total > 0 else 1.0
    anchor = comps.abs().argmax(dim=0)
    signs = t.sign(comps[anchor, t.arange(text_dim, device=comps.device)])
    comps = comps * t.where(signs == 0, t.ones_like(signs), signs)[None, :]
    return TextReducer(mean, comps.to(t.float32).cpu().numpy()), explained


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


def _kmeans(
    X: t.Tensor,
    K: int,
    n_init: int,
    max_iter: int,
    seed: int,
    tol: float = 1e-6,
) -> Tuple[t.Tensor, t.Tensor, float, int]:
    """Lloyd k-means on unit-norm rows; the restart with the lowest inertia is kept."""
    best = None
    for restart in range(max(1, int(n_init))):
        gen = t.Generator(device=X.device).manual_seed(int(seed) * 100003 + int(K) * 1009 + restart)
        centers = _kmeans_pp_init(X, K, gen)
        iters = 0
        for iters in range(1, max(1, int(max_iter)) + 1):
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
        if best is None or inertia < best[2]:
            best = (centers.clone(), labels.clone(), inertia, iters)
    return best


@t.no_grad()
def _silhouette_cosine(
    X: t.Tensor,
    labels: t.Tensor,
    K: int,
    sample_size: int,
    seed: int,
    chunk: int = 2048,
) -> float:
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


def _cluster_spec(params: Dict) -> str:
    if params["expert_num"] > 0:
        k_spec = f"k{params['expert_num']}"
    else:
        k_spec = f"auto-{params['select']}-k{params['k_min']}-{params['k_max']}"
    if int(params.get("text_dim", 0)) > 0:
        space = f"pca{int(params['text_dim'])}"
    else:
        space = "center" if params["center"] else "raw"
    return f"{k_spec}_{space}_seed{params['seed']}"


def _read_clustering_cache(cache_path: Path, params: Dict, fingerprint: Dict) -> Optional[Dict]:
    """Cached clustering if it exists and matches the persona file and clustering parameters."""
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
        and "centroids" in cached
        and "projection" in cached
        and "labels" in cached
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


def _elbow_k(ks, sses, allowed) -> int:
    """Kneedle-style elbow: the K whose SSE lies farthest below the chord from (ks[0], sses[0]) to (ks[-1], sses[-1])."""
    ks_arr = np.asarray(ks, dtype=np.float64)
    sse_arr = np.asarray(sses, dtype=np.float64)
    if len(ks) < 3 or sse_arr[0] <= sse_arr[-1]:
        return int(min(allowed))
    x = (ks_arr - ks_arr[0]) / (ks_arr[-1] - ks_arr[0])
    y = (sse_arr - sse_arr[-1]) / (sse_arr[0] - sse_arr[-1])
    distance = (1.0 - x) - y
    allowed_set = {int(k) for k in allowed}
    best, best_distance = int(min(allowed)), -np.inf
    for k, d in zip(ks, distance):
        if int(k) in allowed_set and d > best_distance:  # ties -> smallest K
            best, best_distance = int(k), float(d)
    return best


def _compute_clusters(embeddings: np.ndarray, params: Dict, device: t.device) -> Dict:
    """Fit the text reducer (centring + PCA to text_dim) on the user persona pool, then k-means in that space."""
    reducer, explained = _fit_text_reducer(
        embeddings, int(params.get("text_dim", 0)), bool(params["center"]), device
    )
    pool = reducer.transform(embeddings)
    X = t.from_numpy(np.ascontiguousarray(pool, dtype=np.float32)).to(device)
    N = int(X.shape[0])
    if params["expert_num"] > 0:
        candidates = [int(params["expert_num"])]
    else:
        k_max = min(int(params["k_max"]), max(N - 1, 2))
        candidates = list(range(int(params["k_min"]), k_max + 1))
    if not candidates or candidates[0] < 1 or candidates[-1] >= N:
        raise ValueError(f"Invalid persona cluster range {candidates} for {N} persona records.")

    sse_baseline = float(((X - X.mean(dim=0, keepdim=True)) ** 2).sum())  # K = 1, anchors the elbow chord
    results = {}
    for K in candidates:
        centers, labels, inertia, iters = _kmeans(
            X, K, params["n_init"], params["max_iter"], params["seed"]
        )
        score = _silhouette_cosine(X, labels, K, params["silhouette_sample"], params["seed"])
        results[K] = {
            "silhouette": score,
            "centroids": centers.cpu().numpy(),
            "labels": labels.cpu().numpy().astype(np.int64),
            "inertia": inertia,
            "iters": iters,
        }
    select = str(params.get("select", "silhouette"))
    if params["expert_num"] > 0:
        best_k, selected_by = candidates[0], "fixed"
    elif select == "elbow":
        best_k = _elbow_k(
            [1] + candidates, [sse_baseline] + [results[k]["inertia"] for k in candidates], candidates
        )
        selected_by = "elbow"
    else:
        best_k = max(results, key=lambda k: results[k]["silhouette"])  # ties -> smallest K
        selected_by = "silhouette"
    chosen = results[best_k]
    return {
        "K": int(best_k),
        "selected_by": selected_by,
        "sse_baseline": sse_baseline,
        "centroids": _l2_normalize_np(chosen["centroids"]),
        "labels": chosen["labels"],
        "sizes": np.bincount(chosen["labels"], minlength=best_k).tolist(),
        "silhouette": {int(k): float(v["silhouette"]) for k, v in results.items()},
        "inertia": {int(k): float(v["inertia"]) for k, v in results.items()},
        "device": str(device),
        "projection": reducer.state(),  # centring mean + PCA components: maps raw text embeddings into the pool space
        "text_dim": int(reducer.dim),
        "raw_dim": int(reducer.raw_dim),
        "explained_variance": float(explained),
        "params": params,
        "fingerprint": None,  # filled by _cluster_personas
    }


_CLUSTER_LOCK_TIMEOUT = 900.0  # seconds to wait for another process that is computing the same cache


def _cluster_personas(
    embeddings: np.ndarray,
    params: Dict,
    cache_path: Path,
    fingerprint: Dict,
    device: t.device,
) -> Dict:
    """Map the raw user persona embeddings into the d_text space and cluster them; K is fixed
    (expert_num > 0) or chosen by silhouette / SSE elbow.

    The result (projection + clustering) is cached on disk so every build of a run, and every later run,
    uses the same experts in the same space.
    A lock file lets runs launched at the same time (e.g. several seeds) share one computation instead
    of racing to write different clusterings.
    """
    use_cache = bool(params["cache"])
    if use_cache:
        cached = _read_clustering_cache(cache_path, params, fingerprint)
        if cached is not None:
            return cached

    lock_path = cache_path.with_name(cache_path.name + ".lock")
    have_lock = False
    if use_cache:
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
        if use_cache:
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


def _best_persona_per_cluster(
    table: np.ndarray,
    slots: np.ndarray,
    centroids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every node and cluster: the node persona with the largest cosine to the cluster centroid.

    ``table`` is [records + 1, dim] whose last row is the fallback (zero) persona used by nodes without
    records.  Returns (record index [nodes, K], similarity [nodes, K], missing mask [nodes]).
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


def _cluster_means(pool: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """[K, dim] unit-normalised mean of the ``pool`` rows of every cluster: the clusters' centroids in that space."""
    n = int(pool.shape[0])
    counts = np.bincount(labels, minlength=K).astype(np.float64)
    if (counts == 0).any():
        raise ValueError(f"Empty persona cluster(s): sizes={counts.astype(int).tolist()}")
    member = sp.csr_matrix((np.ones(n, dtype=np.float64), (labels, np.arange(n))), shape=(K, n))
    sums = np.asarray(member.dot(pool.astype(np.float64)))
    return _l2_normalize_np((sums / counts[:, None]).astype(np.float32))


def _shuffle_user_owners(
    owner: np.ndarray,
    labels: np.ndarray,
    user_num: int,
    mode: str,
    seed: int,
) -> Tuple[np.ndarray, int]:
    """Control experiment: permute which user owns which persona *set* (facets stay together), then train from
    scratch.  The persona pool, hence the clustering, is unchanged; only the user -> personas mapping moves.

    ``global``: permute among users with the same number of personas.  ``within_cluster``: additionally only
    among users with the same dominant cluster (majority label of their personas), so the expert routing a user
    would get from its own personas is kept and only the individual persona content is randomised.  Users
    without personas keep none.  Returns (new owner per record, number of users whose set changed).
    """
    counts = np.bincount(owner, minlength=user_num).astype(np.int64)
    strata = counts.copy()
    if mode == "within_cluster":
        K = int(labels.max()) + 1 if labels.size else 1
        per_user = np.zeros((user_num, K), dtype=np.int64)
        np.add.at(per_user, (owner, labels), 1)
        strata = strata * K + per_user.argmax(axis=1)
    rng = np.random.default_rng(seed)
    perm = np.arange(user_num)  # perm[u] = the user that receives u's persona set
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        if members.size > 1 and counts[members[0]] > 0:
            perm[members] = rng.permutation(members)
    moved = int(((perm != np.arange(user_num)) & (counts > 0)).sum())
    return perm[owner], moved


_BUNDLE_MEMO: Dict[str, SimpleNamespace] = {}


def _build_persona_bundle(
    data_dir: Path,
    dataset: str,
    user_num: int,
    item_num: int,
    params: Dict,
) -> SimpleNamespace:
    """Load the user personas and the item text (RLMRec profile or item persona), cluster the user personas
    into experts and derive the per-node persona tables and cluster similarities."""
    memo_key = f"{data_dir.resolve()}|{sorted(params.items())}"
    if memo_key in _BUNDLE_MEMO:
        return _BUNDLE_MEMO[memo_key]

    persona_path = data_dir / PERSONA_FILE
    if not persona_path.exists():
        raise FileNotFoundError(
            f"Missing LLM persona embeddings for {dataset}: {persona_path}. "
            "PersonaMosaic needs persona/persona_emb.pkl from the persona generation step."
        )
    users, _, meta = _parse_persona_pickle(persona_path)  # item persona records, if any, are not used
    if users is None:
        raise ValueError(f"No user persona records ('u<uid>:p<j>') found in {persona_path}.")
    if users.owner.size and (int(users.owner.min()) < 0 or int(users.owner.max()) >= user_num):
        raise ValueError(
            f"User persona ids in {persona_path} are outside [0, {user_num}): "
            f"min={int(users.owner.min())}, max={int(users.owner.max())}"
        )
    raw_dim = users.dim
    item_text = str(params["item_text"]).lower()
    if item_text not in ITEM_TEXTS:
        raise ValueError(f"item_text must be one of {ITEM_TEXTS}, got {item_text!r}.")
    if item_text == "persona":
        item_emb, item_valid, item_text_path = _load_item_personas(data_dir, item_num, raw_dim)
    else:
        profiles = _load_item_profiles(data_dir, item_num, raw_dim)
        item_text_path = data_dir / ITEM_PROFILE_FILE
    persona_shuffle = str(params.get("persona_shuffle", "none")).lower()
    if persona_shuffle not in PERSONA_SHUFFLES:
        raise ValueError(f"persona_shuffle must be one of {PERSONA_SHUFFLES}, got {persona_shuffle!r}.")
    item_shuffle = bool(params.get("item_shuffle", False))
    shuffle_seed = int(params.get("shuffle_seed", params["seed"]))
    shuffle_notes = []
    if item_shuffle:
        # Control experiment: item i receives the text vector of a random other item (trained from scratch).
        item_perm = np.random.default_rng(shuffle_seed + 1).permutation(item_num)
        if item_text == "persona":
            item_emb, item_valid = item_emb[item_perm], item_valid[item_perm]
        else:
            profiles = profiles[item_perm]
        shuffle_notes.append(f"item text shuffled across items (seed {shuffle_seed + 1})")
    text_dim = int(params["text_dim"])
    if text_dim > raw_dim:
        raise ValueError(f"text_dim (d_text) cannot exceed the text embedding dim {raw_dim}, got {text_dim}.")
    if text_dim == raw_dim:
        text_dim = 0  # the raw space; shares the cache of text_dim 0

    clustering_params = {
        "expert_num": int(params["expert_num"]),
        "k_min": int(params["k_min"]),
        "k_max": int(params["k_max"]),
        "seed": int(params["seed"]),
        "n_init": int(params["n_init"]),
        "max_iter": int(params["max_iter"]),
        "silhouette_sample": int(params["silhouette_sample"]),
        "center": bool(params["center"]),
        "cache": bool(params["cache"]),
        "select": str(params["select"]),
        "text_dim": text_dim,
    }
    stat = persona_path.stat()
    fingerprint = {
        "size": int(stat.st_size),
        "mtime": int(stat.st_mtime),
        "records": int(users.num_records),
        "dim": int(raw_dim),
    }
    cache_path = data_dir / CLUSTER_CACHE_TEMPLATE.format(spec=_cluster_spec(clustering_params))
    # The clustering step also fits the text reducer (centring + PCA to text_dim) on the user persona pool; the
    # projection is cached next to the centroids so that every process maps personas into the same space.
    clusters = _cluster_personas(
        users.embeddings, clustering_params, cache_path, fingerprint, _clustering_device(params.get("device", "auto"))
    )
    reducer = TextReducer.from_state(clusters["projection"])
    if reducer.raw_dim != raw_dim:
        raise ValueError(f"Cached projection expects {reducer.raw_dim}-d text embeddings, personas are {raw_dim}-d.")
    user_pool = reducer.transform(users.embeddings)  # [records, d_text] unit vectors: the clustering space
    clustering_dim = reducer.dim
    clustering_centroids = np.asarray(clusters["centroids"], dtype=np.float32)
    K = int(clusters["K"])
    if clustering_centroids.shape != (K, clustering_dim):
        raise ValueError(
            f"Cached centroids {clustering_centroids.shape} do not match K={K}, d_text={clustering_dim}."
        )
    labels = np.asarray(clusters["labels"], dtype=np.int64).reshape(-1)
    if labels.shape[0] != users.num_records or (labels.size and (labels.min() < 0 or labels.max() >= K)):
        raise ValueError(f"Cached cluster labels {labels.shape} do not match {users.num_records} personas / K={K}.")

    # Prior space: what the expert MLPs consume.  'raw' = the centred, unit-normalised raw text embeddings
    # (MLP^k: R^raw_dim -> R^d as in MOSAIC) with raw-space centroids of the d_text clusters; 'pca' = the
    # d_text clustering vectors and centroids.  Without PCA (text_dim 0) both spaces coincide.
    prior_space = str(params["prior_space"]).lower()
    if prior_space not in PRIOR_SPACES:
        raise ValueError(f"prior_space must be one of {PRIOR_SPACES}, got {prior_space!r}.")
    prior_is_raw = prior_space == "raw" and reducer.components is not None
    raw_mean = (
        users.embeddings.mean(axis=0, keepdims=True) if params["center"] else np.zeros((1, raw_dim), np.float32)
    )
    if prior_is_raw:
        dim = raw_dim
        user_prior_pool = _l2_normalize_np(users.embeddings - raw_mean)
        centroids = _cluster_means(user_prior_pool, labels, K)

        def to_prior(x: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
            return _l2_normalize_np(np.asarray(x, dtype=np.float32) - (raw_mean if mean is None else mean))

    else:
        prior_space = "pca" if reducer.components is not None else "raw"
        dim = clustering_dim
        user_prior_pool = user_pool
        centroids = clustering_centroids

        def to_prior(x: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
            return reducer.transform(x, mean=mean)

    fallback = np.zeros((1, dim), dtype=np.float32)  # "no persona": zero vector (prior space)
    cluster_fallback = np.zeros((1, clustering_dim), dtype=np.float32)
    user_table = np.vstack([user_prior_pool, fallback])  # prior-space rows, same record order as user_pool
    user_owner = users.owner
    if persona_shuffle != "none":
        user_owner, moved = _shuffle_user_owners(users.owner, labels, user_num, persona_shuffle, shuffle_seed)
        shuffle_notes.append(f"user persona sets shuffled {persona_shuffle} ({moved} users moved, seed {shuffle_seed})")
    user_slots = _node_slots(user_owner, user_num)
    user_index, user_sim, user_missing = _best_persona_per_cluster(
        np.vstack([user_pool, cluster_fallback]), user_slots, clustering_centroids
    )

    if item_text == "persona":
        # Items: one persona per item (no facets), so it is the item's closest persona for every cluster.  It is in
        # the same text space as the user personas and goes through the same maps: centring with the user persona
        # mean (+ the user persona PCA) into the clustering space, and to_prior into the prior space.  Items with
        # an invalid persona get the zero fallback row, like users without personas.
        item_pool = reducer.transform(item_emb)  # clustering space: similarity to every cluster
        item_table = np.vstack([to_prior(item_emb), fallback])
        item_slots = np.where(item_valid, np.arange(item_num, dtype=np.int64), -1)[:, None]
        item_index, item_sim, item_missing = _best_persona_per_cluster(
            np.vstack([item_pool, cluster_fallback]), item_slots, clustering_centroids
        )
    else:
        # Items: the RLMRec profile embedding stands in for a persona (one row per item).  Profiles are centred
        # by their own mean (a different text kind than the personas) and mapped with the persona PCA into the
        # clustering space for the cluster similarities, and into the prior space for the expert inputs.
        profile_mean = (
            profiles.mean(axis=0, keepdims=True) if params["center"] else np.zeros((1, raw_dim), dtype=np.float32)
        )
        pca_mean = profiles.mean(axis=0, keepdims=True) if reducer.components is not None else profile_mean
        profile_pool = reducer.transform(profiles, mean=pca_mean)  # clustering space: similarity to every cluster
        item_table = np.vstack([to_prior(profiles, mean=profile_mean if prior_is_raw else pca_mean), fallback])
        item_slots = np.arange(item_num, dtype=np.int64)[:, None]
        item_index, item_sim, item_missing = _best_persona_per_cluster(
            np.vstack([profile_pool, cluster_fallback]), item_slots, clustering_centroids
        )

    bundle = SimpleNamespace(
        K=K,
        dim=dim,  # prior-space dimension (what the expert MLPs consume)
        raw_dim=raw_dim,
        clustering_dim=clustering_dim,
        prior_space=prior_space,
        reducer=reducer,
        explained_variance=float(clusters.get("explained_variance", 1.0)),
        centroids=centroids,  # prior-space centroids
        clustering_centroids=clustering_centroids,
        labels=labels,
        clusters=clusters,
        user_table=user_table,
        user_index=user_index,
        user_sim=user_sim,
        user_missing=user_missing,
        user_records=users.num_records,
        item_table=item_table,  # prior-space item rows (persona or RLMRec profile, + zero fallback row)
        item_index=item_index,
        item_sim=np.asarray(item_sim, dtype=np.float32),
        item_missing=item_missing,
        view_names=meta.get("view_names", []),
        persona_path=str(persona_path),
        item_text=item_text,
        item_text_path=str(item_text_path),
        cache_path=str(cache_path),
        shuffle_note="; ".join(shuffle_notes),
    )
    _BUNDLE_MEMO[memo_key] = bundle
    return bundle


class PriorProjector(nn.Module):
    """Project a node LLM prior into the LightGCN embedding space."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or max(output_dim, (input_dim + output_dim) // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                init(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x.float())


class LayerWisePriorModulator(nn.Module):
    """Prior-conditioned gate used immediately before one LightGCN propagation hop."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        dropout: float,
        max_scale: float,
        lambda_init: float,
        init_std: float,
    ):
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.max_scale = float(max_scale)
        self.init_std = float(init_std)
        if self.max_scale < 0:
            raise ValueError("layer_modulation_scale must be non-negative.")
        lambda_init = float(lambda_init)
        if lambda_init < 0:
            raise ValueError("layer_modulation_lambda_init must be non-negative.")
        if self.max_scale == 0.0:
            lambda_init = 0.0
        elif lambda_init > self.max_scale:
            warnings.warn(
                "layer_modulation_lambda_init is larger than layer_modulation_scale; "
                "clamping the initial layer-wise modulation lambda to the scale cap."
            )
            lambda_init = self.max_scale
        self.lambda_init = lambda_init
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_size, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), self.embedding_size),
        )
        self.norm = nn.LayerNorm(self.embedding_size)
        self.lambda_param = nn.Parameter(t.tensor(self.lambda_init, dtype=t.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linears = [module for module in self.mlp if isinstance(module, nn.Linear)]
        for layer in linears:
            if self.init_std > 0:
                nn.init.normal_(layer.weight, mean=0.0, std=self.init_std)
            else:
                nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def modulation_lambda(self) -> t.Tensor:
        return t.clamp(self.lambda_param, min=0.0, max=self.max_scale)

    def forward(self, prior: t.Tensor) -> t.Tensor:
        if prior.shape[-1] != self.embedding_size:
            raise ValueError(
                f"LayerWisePriorModulator expected last dim {self.embedding_size}, got {prior.shape}."
            )
        delta = self.norm(self.mlp(prior.float()))
        lambda_l = self.modulation_lambda().to(device=delta.device, dtype=delta.dtype)
        return 1.0 + lambda_l * t.tanh(delta)


class SharedSoftmaxTopKGate(nn.Module):
    """One shared softmax router used by both user and item nodes."""

    def __init__(self, input_dim: int, hidden_dim: int, expert_num: int):
        super().__init__()
        self.expert_num = int(expert_num)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, expert_num),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.gate:
            if isinstance(module, nn.Linear):
                init(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: t.Tensor, topk: int) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        logits = self.gate(features.float())
        probs = F.softmax(logits, dim=-1)

        topk = max(1, min(int(topk), self.expert_num))
        top_probs, top_idx = t.topk(probs, k=topk, dim=-1)
        top_weights = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-12)

        sparse_weights = t.zeros_like(probs)
        sparse_weights.scatter_(1, top_idx, top_weights)

        selected = (sparse_weights > 0).float()
        importance = probs.mean(dim=0)
        load = sparse_weights.mean(dim=0) if topk == self.expert_num else selected.mean(dim=0)
        balance_loss = self.expert_num * t.sum(importance * load)
        return sparse_weights, top_idx, probs, balance_loss


class LightGCNExpert(nn.Module):
    """Layer-wise modulated LightGCN unit for one MoE expert (ExpertGCN).

    The expert uses learned user/item node parameters as its initial LightGCN
    embedding. At every layer it computes a prior-conditioned gate and applies
    it immediately before graph propagation, so priors modulate the graph
    filter without replacing the learned E^(0). The final readout is a
    learnable weighted sum of the modulated hop representations.
    """

    def __init__(
        self,
        user_num: int,
        item_num: int,
        embedding_size: int,
        layer_num: int,
        modulation_hidden_size: int,
        modulation_dropout: float,
        modulation_scale: float,
        modulation_lambda_init: float,
        modulation_init_std: float,
    ):
        super().__init__()
        self.user_num = int(user_num)
        self.item_num = int(item_num)
        self.embedding_size = int(embedding_size)
        self.layer_num = int(layer_num)
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.layer_modulators = nn.ModuleList(
            [
                LayerWisePriorModulator(
                    self.embedding_size,
                    modulation_hidden_size,
                    modulation_dropout,
                    modulation_scale,
                    modulation_lambda_init,
                    modulation_init_std,
                )
                for _ in range(self.layer_num + 1)
            ]
        )
        # Each hop has its own module; this assertion catches accidental module reuse.
        assert len({id(module) for module in self.layer_modulators}) == self.layer_num + 1
        self.edge_dropper = SpAdjEdgeDrop()

    def forward(
        self,
        adj: t.Tensor,
        keep_rate: float,
        user_prior: t.Tensor,
        item_prior: t.Tensor,
        layer_logits: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor]:
        expected_user = (self.user_num, self.embedding_size)
        expected_item = (self.item_num, self.embedding_size)
        for name, tensor, shape in (
            ("user_prior", user_prior, expected_user),
            ("item_prior", item_prior, expected_item),
        ):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}.")

        user_embeds = self.user_embeds
        item_embeds = self.item_embeds
        embed_device = user_embeds.device
        if item_embeds.device != embed_device:
            item_embeds = item_embeds.to(device=embed_device, non_blocking=True)
        if user_prior.device != embed_device:
            user_prior = user_prior.to(device=embed_device, non_blocking=True)
        if item_prior.device != embed_device:
            item_prior = item_prior.to(device=embed_device, non_blocking=True)
        if adj.device != embed_device:
            adj = adj.to(device=embed_device)
        if layer_logits.device != embed_device:
            layer_logits = layer_logits.to(device=embed_device, non_blocking=True)
        if layer_logits.numel() != self.layer_num + 1:
            raise ValueError(
                f"layer_logits length must be layer_num + 1 ({self.layer_num + 1}), got {layer_logits.numel()}."
            )

        all_embeds = t.cat([user_embeds, item_embeds], dim=0)
        all_priors = t.cat([user_prior, item_prior], dim=0)
        readout_terms = []
        prop_adj = self.edge_dropper(adj, keep_rate) if self.training else adj
        if prop_adj.device != embed_device:
            prop_adj = prop_adj.to(device=embed_device)

        for layer_idx in range(self.layer_num + 1):
            gate = self.layer_modulators[layer_idx](all_priors)
            if gate.shape != all_embeds.shape:
                raise ValueError(f"Layer-wise gate shape {gate.shape} must match embeddings {all_embeds.shape}.")
            modulated_embeds = gate * all_embeds
            readout_terms.append(modulated_embeds)
            if layer_idx < self.layer_num:
                all_embeds = t.spmm(prop_adj, modulated_embeds)

        assert len(readout_terms) == self.layer_num + 1
        alpha = F.softmax(layer_logits.reshape(-1), dim=0)
        all_embeds = sum(alpha[layer_idx] * readout_terms[layer_idx] for layer_idx in range(self.layer_num + 1))

        return all_embeds[: self.user_num], all_embeds[self.user_num :]


def _weighted_infonce(
    embeds: t.Tensor,
    ids: t.Tensor,
    prior_all: t.Tensor,
    weights: t.Tensor,
    temp: float,
) -> t.Tensor:
    """Sum over rows with weight > 0 of weight * InfoNCE(embed_row, prior[id_row] | all priors)."""
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
    """MOSAIC with K persona-cluster experts, persona-derived priors and persona-aware routing."""

    def __init__(self, data_handler):
        super().__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.dataset = configs["data"]["name"]
        self.data_dir = Path(configs["data"]["dir"])  # data/<dataset>, set by config/configurator.py

        self.keep_rate = float(configs["model"].get("keep_rate", 1.0))
        self.layer_num = int(self._hp("layer_num", 3))
        self.reg_weight = float(self._hp("reg_weight", 1e-6))
        self.balance_loss_weight = float(self._hp("balance_loss_weight", 0.0))

        self.llm_embedding_device = str(configs["model"].get("llm_embedding_device", "model")).lower()
        if self.llm_embedding_device in {"cuda", "gpu"}:
            self.llm_embedding_device = "model"
        if self.llm_embedding_device not in {"cpu", "model"}:
            raise ValueError("PersonaMosaic llm_embedding_device must be either 'cpu' or 'model'.")

        trn_mat = _load_train_matrix(data_handler, self.data_dir)
        user_comm, item_comm, community_meta = _load_community_assignments(
            self.data_dir,
            self.dataset,
            self.user_num,
            self.item_num,
        )
        self.num_communities = int(community_meta["num_communities"])

        # ----- Phase 1: persona clusters -> experts, persona priors, cluster similarities -----
        persona_params = {
            "expert_num": int(self._hp("expert_num", 0)),
            "k_min": int(self._hp("clustering_k_min", 2)),
            "k_max": int(self._hp("clustering_k_max", 16)),
            "seed": int(self._hp("clustering_seed", configs["train"].get("seed", 2025))),
            "n_init": int(self._hp("clustering_n_init", 4)),
            "max_iter": int(self._hp("clustering_max_iter", 300)),
            "silhouette_sample": int(self._hp("clustering_silhouette_sample", 10000)),
            "cache": bool(self._hp("clustering_cache", True)),
            "center": bool(self._hp("persona_center", True)),
            "device": str(self._hp("clustering_device", "auto")).lower(),
            "select": str(self._hp("clustering_k_select", "silhouette")).lower(),
            "text_dim": int(self._hp("text_dim", 0)),  # 0 / raw dim: no PCA
            "prior_space": str(self._hp("prior_space", "raw")).lower(),
            "item_text": str(self._hp("item_text", "profile")).lower(),  # profile: RLMRec item profile | persona: item persona
            # control experiments (see _shuffle_user_owners): none | global | within_cluster; item_shuffle: bool
            "persona_shuffle": str(self._hp("persona_shuffle", "none")).lower(),
            "item_shuffle": bool(self._hp("item_shuffle", False)),
            "shuffle_seed": int(self._hp("shuffle_seed", self._hp("clustering_seed", configs["train"].get("seed", 2025)))),
        }
        if persona_params["prior_space"] not in PRIOR_SPACES:
            raise ValueError(f"prior_space must be one of {PRIOR_SPACES}, got {persona_params['prior_space']!r}.")
        if persona_params["text_dim"] < 0:
            raise ValueError("text_dim (d_text) must be 0 / the raw text embedding dimension or a positive PCA dimension.")
        if persona_params["select"] not in ("silhouette", "elbow"):
            raise ValueError(f"clustering_k_select must be 'silhouette' or 'elbow', got {persona_params['select']!r}.")
        if persona_params["expert_num"] < 0:
            raise ValueError("expert_num must be 0 (silhouette-selected K) or a positive number of experts.")
        if persona_params["expert_num"] == 0 and not 1 <= persona_params["k_min"] <= persona_params["k_max"]:
            raise ValueError("clustering_k_min / clustering_k_max must satisfy 1 <= k_min <= k_max.")
        bundle = _build_persona_bundle(
            self.data_dir,
            self.dataset,
            self.user_num,
            self.item_num,
            persona_params,
        )
        self.persona_params = persona_params
        self.expert_num = int(bundle.K)
        self.all_expert_ids = tuple(range(self.expert_num))
        self.persona_dim = int(bundle.dim)  # prior-space dimension: input width of every MLP^k
        self.clustering_dim = int(bundle.clustering_dim)  # d_text: k-means, closest-persona choice, gate similarities
        self.prior_space = str(bundle.prior_space)
        self.raw_text_dim = int(bundle.raw_dim)
        self.top_k = int(self._hp("top_k", 2))
        if not 1 <= self.top_k <= self.expert_num:
            raise ValueError(
                f"PersonaMosaic flat gate requires top_k in [1, {self.expert_num}] (K persona clusters), got {self.top_k}."
            )
        self.prior_composition = str(self._hp("prior_composition", "product")).lower()
        if self.prior_composition not in PRIOR_COMPOSITIONS:
            raise ValueError(f"prior_composition must be one of {PRIOR_COMPOSITIONS}, got {self.prior_composition!r}.")
        # persona (.) centroid of unit vectors has norm ~ 1/sqrt(dim); rescale to unit order like the other inputs.
        self._product_scale = math.sqrt(float(self.persona_dim))

        self.register_buffer("cluster_centroids", t.from_numpy(bundle.centroids.astype(np.float32)), persistent=False)
        self._store_static_feature("user_persona_table", bundle.user_table)
        self.register_buffer("user_persona_index", t.from_numpy(bundle.user_index.astype(np.int64)), persistent=False)
        self.register_buffer("user_cluster_sim", t.from_numpy(bundle.user_sim.astype(np.float32)), persistent=False)
        self.register_buffer("item_cluster_sim", t.from_numpy(bundle.item_sim.astype(np.float32)), persistent=False)
        self._store_static_feature("item_persona_table", bundle.item_table)  # one text vector per item
        self.register_buffer("item_persona_index", t.from_numpy(bundle.item_index.astype(np.int64)), persistent=False)

        # ----- Gate: MOSAIC topology features (+ cluster similarities) -> shared softmax top-k gate -----
        user_gate_features, item_gate_features = _build_gate_features(trn_mat, user_comm, item_comm)
        self.gate_persona_similarity = bool(self._hp("gate_persona_similarity", True))
        if self.gate_persona_similarity:
            user_sim_z, item_sim_z = _combined_zscore(bundle.user_sim, bundle.item_sim)
            user_gate_features = np.hstack([user_gate_features, user_sim_z]).astype(np.float32)
            item_gate_features = np.hstack([item_gate_features, item_sim_z]).astype(np.float32)
        self.gate_input_dim = int(user_gate_features.shape[1])
        self.register_buffer("user_gate_features", t.from_numpy(user_gate_features), persistent=False)
        self.register_buffer("item_gate_features", t.from_numpy(item_gate_features), persistent=False)

        gate_hidden = int(self._hp("gate_hidden_size", 64))
        self.gate = SharedSoftmaxTopKGate(
            input_dim=self.gate_input_dim,
            hidden_dim=gate_hidden,
            expert_num=self.expert_num,
        )

        # ----- Phase 2: K ExpertGCNs with persona-prior diversifiers -----
        prior_hidden = self._hp("prior_hidden_size", None)
        prior_hidden = int(prior_hidden) if prior_hidden else None
        self.prior_projectors = nn.ModuleList(
            [PriorProjector(self.persona_dim, self.embedding_size, prior_hidden) for _ in range(self.expert_num)]
        )

        modulation_hidden = 4 * self.embedding_size
        modulation_dropout = 0.0
        self.layer_modulation_scale = float(self._hp("layer_modulation_scale", 0.5))
        modulation_lambda_init = float(
            self._hp("layer_modulation_lambda_init", min(self.layer_modulation_scale, 1.0e-2))
        )
        modulation_init_std = float(self._hp("layer_modulation_init_std", 1.0e-3))
        self.lightgcn_experts = nn.ModuleList(
            [
                LightGCNExpert(
                    self.user_num,
                    self.item_num,
                    self.embedding_size,
                    self.layer_num,
                    modulation_hidden,
                    modulation_dropout,
                    self.layer_modulation_scale,
                    modulation_lambda_init,
                    modulation_init_std,
                )
                for _ in range(self.expert_num)
            ]
        )
        self.layer_logits = nn.Parameter(t.zeros(self.expert_num, self.layer_num + 1))
        assert self.layer_logits.shape == (self.expert_num, self.layer_num + 1)
        self.llm_align_temperature = float(self._hp("llm_align_temperature", 0.2))
        self.persona_align_weight = float(self._hp("persona_align_weight", 0.0))

        self._cached_e_user = None
        self._cached_e_item = None
        self._log_summary(bundle)

    # ------------------------------------------------------------------ config / logging helpers

    def _hp(self, key: str, default):
        """Dataset block first (also where --override lands), then the model block, then the default."""
        return self.hyper_config.get(key, configs["model"].get(key, default))

    def _log_summary(self, bundle: SimpleNamespace) -> None:
        clusters = bundle.clusters
        if bundle.reducer.components is not None:
            clustering_space = (
                f"d_text={self.clustering_dim} (PCA of {bundle.raw_dim}-d text embeddings, "
                f"{bundle.explained_variance:.1%} variance kept)"
            )
        else:
            clustering_space = f"d_text={self.clustering_dim} (raw text embeddings)"
        prior_space = f"{self.persona_dim}-d {'raw text embeddings' if self.prior_space == 'raw' else 'PCA vectors'}"
        if clusters.get("selected_by") == "fixed":
            selection = f"fixed K={self.expert_num} (silhouette {clusters['silhouette'].get(self.expert_num, float('nan')):.4f})"
        else:
            selection = (
                f"K={self.expert_num} selected by {clusters.get('selected_by')} "
                f"(silhouette {clusters['silhouette'][self.expert_num]:.4f}) "
                f"over K={self.persona_params['k_min']}..{self.persona_params['k_max']}"
            )
        summary = (
            f"{selection}; sizes={clusters.get('sizes')}; user personas={bundle.user_records} "
            f"(users without persona={int(bundle.user_missing.sum())}); "
            + (
                f"item personas={self.item_num} (items without persona={int(bundle.item_missing.sum())}); "
                if bundle.item_text == "persona"
                else f"items=RLMRec profile {ITEM_PROFILE_FILE}; "
            )
            + f"prior={self.prior_composition} "
            f"on {prior_space}; clustering {clustering_space}; gate input={self.gate_input_dim} (topology 4"
            f"{' + cluster sim ' + str(self.expert_num) if self.gate_persona_similarity else ''}); "
            f"top_k={self.top_k}; cache={bundle.cache_path}"
            + (f"; CONTROL: {bundle.shuffle_note}" if bundle.shuffle_note else "")
        )
        configs["model"]["persona_clusters"] = summary  # picked up by Logger's 'Model|' line
        key = (self.dataset, summary)
        if key in _SUMMARY_PRINTED:
            return
        _SUMMARY_PRINTED.add(key)
        print(f"[persona_mosaic] {self.dataset}: {summary}")
        if clusters.get("selected_by") != "fixed":
            curve = " ".join(f"{k}:{v:.4f}" for k, v in sorted(clusters["silhouette"].items()))
            print(f"[persona_mosaic] silhouette curve: {curve}")
            sse = " ".join(f"{k}:{v:.0f}" for k, v in sorted(clusters["inertia"].items()))
            baseline = clusters.get("sse_baseline")
            prefix = f"1:{baseline:.0f} " if baseline is not None else ""
            print(f"[persona_mosaic] SSE curve: {prefix}{sse}")

    # ------------------------------------------------------------------ static tables

    def _store_static_feature(self, name: str, value: np.ndarray) -> None:
        tensor = t.from_numpy(np.ascontiguousarray(value.astype(np.float32, copy=False)))
        if self.llm_embedding_device == "model":
            self.register_buffer(name, tensor, persistent=False)
        else:
            setattr(self, name, tensor)

    def _feature_table(self, name: str, device: t.device) -> t.Tensor:
        feature = getattr(self, name)
        if feature.device == device:
            return feature
        return feature.to(device=device, non_blocking=True)

    def _embedding_device(self) -> t.device:
        return self.lightgcn_experts[0].user_embeds.device

    def _centroid(self, expert_idx: int, device: t.device) -> t.Tensor:
        centroids = self.cluster_centroids
        if centroids.device != device:
            centroids = centroids.to(device=device, non_blocking=True)
        return centroids[expert_idx]

    def _persona_rows(self, node_type: str, expert_idx: int, device: t.device) -> t.Tensor:
        """[nodes, dim] persona of every node closest to cluster ``expert_idx`` (an item's single persona)."""
        table = getattr(self, f"{node_type}_persona_table")
        index = getattr(self, f"{node_type}_persona_index")[:, expert_idx]
        if index.device != table.device:
            index = index.to(device=table.device)
        rows = table.index_select(0, index)
        if rows.device != device:
            rows = rows.to(device=device, non_blocking=True)
        return rows

    def _compose_prior_input(self, persona_rows: t.Tensor, expert_idx: int) -> t.Tensor:
        if self.prior_composition == "persona":
            return persona_rows
        centroid = self._centroid(expert_idx, persona_rows.device).unsqueeze(0)
        if self.prior_composition == "cluster":
            return centroid.expand(persona_rows.shape[0], -1)
        return persona_rows * centroid * self._product_scale

    def _raw_node_prior_table(
        self,
        node_type: str,
        expert_idx: int,
        device: t.device | None = None,
        prior_cache: PriorCache | None = None,
    ) -> t.Tensor:
        """Projected persona prior x^k_v for every node of ``node_type`` under expert ``expert_idx``."""
        expert_idx = int(expert_idx)
        if not 0 <= expert_idx < self.expert_num:
            raise ValueError(f"Selected expert index out of range: {expert_idx}")
        if device is None:
            device = self._embedding_device()
        device = t.device(device)
        cache_key = (node_type, expert_idx, str(device))
        if prior_cache is not None and cache_key in prior_cache:
            return prior_cache[cache_key]
        if node_type not in {"user", "item"}:
            raise ValueError(f"Unknown node_type: {node_type}")

        with t.no_grad():
            rows = self._persona_rows(node_type, expert_idx, device)
            prior_input = self._compose_prior_input(rows, expert_idx)
        prior = self.prior_projectors[expert_idx](prior_input)
        if prior_cache is not None:
            prior_cache[cache_key] = prior
        return prior

    def _gate_features(self, node_type: str, device: t.device | None = None) -> t.Tensor:
        if node_type not in {"user", "item"}:
            raise ValueError(f"Unknown node_type: {node_type}")
        features = self.user_gate_features if node_type == "user" else self.item_gate_features
        if device is not None and features.device != device:
            features = features.to(device=device, non_blocking=True)
        return features

    # ------------------------------------------------------------------ MoE forward

    def _compute_expert_outputs(
        self,
        keep_rate: float,
        prior_cache: PriorCache | None = None,
    ) -> Tuple[Dict[int, t.Tensor], Dict[int, t.Tensor]]:
        device = self._embedding_device()
        user_outputs = {}
        item_outputs = {}
        for global_idx in self.all_expert_ids:
            global_idx = int(global_idx)
            user_prior = self._raw_node_prior_table("user", global_idx, device=device, prior_cache=prior_cache)
            item_prior = self._raw_node_prior_table("item", global_idx, device=device, prior_cache=prior_cache)
            user_out, item_out = self.lightgcn_experts[global_idx](
                self.adj,
                keep_rate,
                user_prior=user_prior,
                item_prior=item_prior,
                layer_logits=self.layer_logits[global_idx],
            )
            user_outputs[global_idx] = user_out
            item_outputs[global_idx] = item_out
        return user_outputs, item_outputs

    def _route_expert_outputs(
        self,
        expert_outputs: Dict[int, t.Tensor],
        weights: t.Tensor,
        expert_ids: Tuple[int, ...],
        node_ids: t.Tensor | None = None,
    ) -> t.Tensor:
        first_output = expert_outputs[int(expert_ids[0])]
        if node_ids is not None:
            node_ids = node_ids.long()
            if node_ids.device != first_output.device:
                node_ids = node_ids.to(device=first_output.device, non_blocking=True)
            expected_rows = node_ids.shape[0]
        else:
            expected_rows = first_output.shape[0]
        if weights.device != first_output.device:
            weights = weights.to(device=first_output.device, non_blocking=True)
        if weights.shape[0] != expected_rows:
            raise ValueError(f"weights rows {weights.shape[0]} must match routed rows {expected_rows}.")

        routed = first_output.new_zeros((expected_rows, self.embedding_size))
        for local_idx, global_idx in enumerate(expert_ids):
            expert_output = expert_outputs[int(global_idx)]
            if node_ids is not None:
                expert_output = expert_output.index_select(0, node_ids)
            routed = routed + weights[:, local_idx : local_idx + 1] * expert_output
        return routed

    def _clear_embedding_cache(self) -> None:
        self._cached_e_user = None
        self._cached_e_item = None

    def _compute_all_embeddings(self, keep_rate: float, prior_cache: PriorCache | None = None):
        if self.training:
            self._clear_embedding_cache()
        if prior_cache is None:
            prior_cache = {}

        user_outputs, item_outputs = self._compute_expert_outputs(keep_rate, prior_cache=prior_cache)
        device = user_outputs[int(self.all_expert_ids[0])].device
        w_user, _, _, b_user = self.gate(self._gate_features("user", device), self.top_k)
        w_item, _, _, b_item = self.gate(self._gate_features("item", device), self.top_k)
        final_user = self._route_expert_outputs(user_outputs, w_user, self.all_expert_ids)
        final_item = self._route_expert_outputs(item_outputs, w_item, self.all_expert_ids)

        balance = 0.5 * (b_user + b_item)
        return final_user, final_item, balance

    def _compute_training_batch_embeddings(
        self,
        keep_rate: float,
        ancs: t.Tensor,
        poss: t.Tensor,
        negs: t.Tensor,
        prior_cache: PriorCache | None = None,
    ):
        self._clear_embedding_cache()
        if prior_cache is None:
            prior_cache = {}

        user_outputs, item_outputs = self._compute_expert_outputs(keep_rate, prior_cache=prior_cache)
        device = user_outputs[int(self.all_expert_ids[0])].device
        w_user, _, _, b_user = self.gate(self._gate_features("user", device), self.top_k)
        w_item, _, _, b_item = self.gate(self._gate_features("item", device), self.top_k)

        batch_users, user_inverse = t.unique(ancs, sorted=False, return_inverse=True)
        batch_item_ids = t.cat([poss, negs], dim=0)
        batch_items, item_inverse = t.unique(batch_item_ids, sorted=False, return_inverse=True)
        pos_inverse = item_inverse[: poss.shape[0]]
        neg_inverse = item_inverse[poss.shape[0] :]

        final_users = self._route_expert_outputs(
            user_outputs,
            w_user.index_select(0, batch_users),
            self.all_expert_ids,
            node_ids=batch_users,
        )
        final_items = self._route_expert_outputs(
            item_outputs,
            w_item.index_select(0, batch_items),
            self.all_expert_ids,
            node_ids=batch_items,
        )

        balance = 0.5 * (b_user + b_item)
        return (
            final_users.index_select(0, user_inverse),
            final_items.index_select(0, pos_inverse),
            final_items.index_select(0, neg_inverse),
            balance,
            w_user,
            w_item,
        )

    def _get_final_embeddings(self, keep_rate: float) -> Tuple[t.Tensor, t.Tensor]:
        if self.training:
            user_embeds, item_embeds, _ = self._compute_all_embeddings(keep_rate)
            return user_embeds, item_embeds
        if self._cached_e_user is None or self._cached_e_item is None:
            user_embeds, item_embeds, _ = self._compute_all_embeddings(keep_rate)
            self._cached_e_user = user_embeds
            self._cached_e_item = item_embeds
        return self._cached_e_user, self._cached_e_item

    def _score_pairs(
        self,
        users: t.Tensor,
        items: t.Tensor,
        return_balance: bool = False,
    ):
        if return_balance:
            all_user_embeds, all_item_embeds, balance = self._compute_all_embeddings(self.keep_rate)
        else:
            all_user_embeds, all_item_embeds = self._get_final_embeddings(self.keep_rate)
            balance = None
        users = users.long()
        items = items.long()
        if users.device != all_user_embeds.device:
            users = users.to(device=all_user_embeds.device, non_blocking=True)
        if items.device != all_item_embeds.device:
            items = items.to(device=all_item_embeds.device, non_blocking=True)
        user_embeds = all_user_embeds.index_select(0, users)
        item_embeds = all_item_embeds.index_select(0, items)
        scores = (user_embeds * item_embeds).sum(dim=-1)

        if return_balance:
            return scores, balance
        return scores

    def forward(self, users: t.Tensor, items: t.Tensor) -> t.Tensor:
        return self._score_pairs(users.long(), items.long(), return_balance=False)

    # ------------------------------------------------------------------ losses

    def cal_loss(self, batch_data):
        ancs, poss, negs = batch_data[:3]
        ancs = ancs.long()
        poss = poss.long()
        negs = negs.long()

        device = self._embedding_device()
        if ancs.device != device:
            ancs = ancs.to(device=device, non_blocking=True)
        if poss.device != device:
            poss = poss.to(device=device, non_blocking=True)
        if negs.device != device:
            negs = negs.to(device=device, non_blocking=True)

        prior_cache: PriorCache = {}
        (
            user_embeds,
            pos_item_embeds,
            neg_item_embeds,
            balance,
            w_user,
            w_item,
        ) = self._compute_training_batch_embeddings(
            self.keep_rate,
            ancs,
            poss,
            negs,
            prior_cache=prior_cache,
        )
        pos_scores = (user_embeds * pos_item_embeds).sum(dim=-1)
        neg_scores = (user_embeds * neg_item_embeds).sum(dim=-1)

        bpr_loss = F.softplus(neg_scores - pos_scores).sum() / ancs.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        balance_loss = self.balance_loss_weight * balance
        persona_align_loss = self._persona_alignment_loss(
            ancs,
            poss,
            negs,
            user_embeds,
            pos_item_embeds,
            neg_item_embeds,
            w_user.detach(),
            w_item.detach(),
            prior_cache=prior_cache,
        )
        loss = bpr_loss + reg_loss + balance_loss + persona_align_loss
        losses = {
            "bpr_loss": bpr_loss,
            "reg_loss": reg_loss,
            "balance_loss": balance_loss,
        }
        if self.persona_align_weight > 0:
            losses["persona_align_loss"] = persona_align_loss
        return loss, losses

    def _persona_alignment_loss(
        self,
        ancs: t.Tensor,
        poss: t.Tensor,
        negs: t.Tensor,
        final_user_embeds: t.Tensor,
        final_pos_item_embeds: t.Tensor,
        final_neg_item_embeds: t.Tensor,
        w_user: t.Tensor,
        w_item: t.Tensor,
        prior_cache: PriorCache | None = None,
    ) -> t.Tensor:
        """InfoNCE between the routed output and the persona prior of each selected expert, gate-weighted.

        For node v with gate weights w^k_v (top-k, summing to one, detached) the term is
        sum_k w^k_v * InfoNCE(y_v, x^k_v | {x^k_u}), so the output stays aligned with the LLM priors of
        the experts that produced it. Averaged over the batch for users, positives and negatives.
        """
        if self.persona_align_weight <= 0:
            return final_user_embeds.new_tensor(0.0)

        batch_size = max(int(ancs.shape[0]), 1)
        temp = self.llm_align_temperature
        device = final_user_embeds.device
        w_anc = w_user.index_select(0, ancs)
        w_pos = w_item.index_select(0, poss)
        w_neg = w_item.index_select(0, negs)

        loss = final_user_embeds.new_tensor(0.0)
        for expert_idx in self.all_expert_ids:
            uses_user = bool((w_anc[:, expert_idx] > 0).any())
            uses_item = bool((w_pos[:, expert_idx] > 0).any()) or bool((w_neg[:, expert_idx] > 0).any())
            if not uses_user and not uses_item:
                continue
            if uses_user:
                user_prior = self._raw_node_prior_table("user", expert_idx, device=device, prior_cache=prior_cache)
                loss = loss + _weighted_infonce(final_user_embeds, ancs, user_prior, w_anc[:, expert_idx], temp)
            if uses_item:
                item_prior = self._raw_node_prior_table("item", expert_idx, device=device, prior_cache=prior_cache)
                loss = loss + _weighted_infonce(final_pos_item_embeds, poss, item_prior, w_pos[:, expert_idx], temp)
                loss = loss + _weighted_infonce(final_neg_item_embeds, negs, item_prior, w_neg[:, expert_idx], temp)
        return self.persona_align_weight * loss / batch_size

    # ------------------------------------------------------------------ inference

    @t.no_grad()
    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        user_all, item_embeds = self._get_final_embeddings(1.0)
        if pck_users.device != user_all.device:
            pck_users = pck_users.to(device=user_all.device, non_blocking=True)
        user_embeds = user_all.index_select(0, pck_users)
        full_preds = user_embeds @ item_embeds.T
        if train_mask.device != full_preds.device:
            train_mask = train_mask.to(device=full_preds.device, non_blocking=True)
        return self._mask_predict(full_preds, train_mask)


# models/bulid_model.py looks up a class whose lower-cased name equals the model name 'persona_mosaic'.
Persona_Mosaic = PersonaMosaic

__all__ = ["PersonaMosaic", "Persona_Mosaic"]
