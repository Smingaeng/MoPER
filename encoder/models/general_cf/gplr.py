"""GPLR (SIGIR'25, "You Are What You Bought"): LLM personas in a low-pass GCN over users, items and personas.

Ported from https://github.com/Hanc1999/YouAreWhatYouBought (src/baselines/lgcn3: model_LGCN_tri.py,
model_LGCN_afd_tri.py, afd_utils.py, pretraining/_graph_embeddings_tri.py, read_data.py).  The method is unchanged;
what follows this repository instead of the original scripts is the data path, the (user, item, negative) batches
of the data handler, and the Trainer with its validation rule.

  * Graph: nodes [users, items, personas]; edges = the training interactions, the user-persona links and the
    item-persona links.  The persona links are LLM output: data/<dataset>/<persona_dir>/{persona2idx,
    tri_graph_uidx2pidx, tri_graph_tidx2pidx}.json, built by generation/gplr/generate_personas.py (GPLR: an LLM
    labels a sample of prototype users, a random walk passes their personas on to the other users).
  * LGCN3: every node has a trainable embedding, E_0.  Layer l filters the embeddings with the low-pass graph
    filter  E_l = U diag(k_l) U^T E_{l-1},  U = the `frequency` eigenvectors of the normalized Laplacian with the
    smallest eigenvalues, k_l = a trainable kernel.  Final embeddings  E_0 + sum_l E_l / (l + 1);  score = inner
    product of a user and an item;  loss = BPR summed over the batch + reg_weight * L2 of the batch's user / item
    embeddings and of all persona embeddings.
  * A-LGCN3 (afd_alpha > 0): the loss also holds afd_alpha * (c(users) + c(items) + c(personas)), c(X) = norm of
    the correlations between the columns of X (the de-correlation loss of AFDGCF) on the final embeddings.

The eigenvectors (the release's pretraining step) are computed once per graph and kept in save/checkpoint/gplr/.
The release runs A-LGCN3 with one layer only (its de-correlation code fails with more); here the de-correlation
term is always the one of the final embeddings, which is what the release computes with one layer.
"""
import hashlib
import json
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import torch as t
from torch import nn

from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import cal_bpr_loss
from trainer.trainer import SAVE_DIR


def _read_links(path, row_num, persona_num):
    """{row id: [persona ids]} of a GPLR file -> unique (row, persona) pairs.  A row that is not listed has no link."""
    with open(path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    pairs = sorted({(int(row), int(persona)) for row, personas in mapping.items() for persona in personas})
    links = np.array(pairs, dtype=np.int64).reshape(-1, 2)
    if len(links) and (links.min() < 0 or links[:, 0].max() >= row_num or links[:, 1].max() >= persona_num):
        raise ValueError("{} does not match this dataset ({} rows, {} personas)".format(path, row_num, persona_num))
    return links


def _persona_graph(persona_dir, user_num, item_num):
    """Persona count, user-persona links and item-persona links from the LLM persona files."""
    paths = {name: os.path.join(persona_dir, name + '.json') for name in ('persona2idx', 'tri_graph_uidx2pidx', 'tri_graph_tidx2pidx')}
    for path in paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(
                "GPLR needs the LLM persona file {}; build it with generation/gplr/generate_personas.py".format(path))
    with open(paths['persona2idx'], 'r', encoding='utf-8') as f:
        persona_num = len(json.load(f))
    user_links = _read_links(paths['tri_graph_uidx2pidx'], user_num, persona_num)
    item_links = _read_links(paths['tri_graph_tidx2pidx'], item_num, persona_num)
    if len(user_links) == 0 and len(item_links) == 0:
        raise ValueError("{} links no user and no item to a persona".format(persona_dir))
    return persona_num, user_links, item_links


def _normalized_adj(trn_mat, user_links, item_links, persona_num):
    """D^-1/2 A D^-1/2 of the tri-partite graph over [users, items, personas] (binary, symmetric A)."""
    user_num, item_num = trn_mat.shape
    size = user_num + item_num + persona_num
    rows = np.concatenate((trn_mat.row, user_links[:, 0], item_links[:, 0] + user_num))
    cols = np.concatenate((trn_mat.col + user_num, user_links[:, 1] + user_num + item_num, item_links[:, 1] + user_num + item_num))
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size)).tocsr()
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.sort_indices()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    inv_sqrt[degree > 0] = degree[degree > 0] ** -0.5
    return (sp.diags(inv_sqrt) @ adj @ sp.diags(inv_sqrt)).tocsr()


def _low_pass_basis(norm_adj, frequency):
    """The `frequency` eigenvectors of the normalized Laplacian I - D^-1/2 A D^-1/2 with the smallest eigenvalues,
    [nodes, frequency], smallest first (pretraining/_graph_embeddings_tri.py; computed as the largest eigenvalues
    of D^-1/2 A D^-1/2, which are the same vectors).  Cached per graph."""
    size = norm_adj.shape[0]
    if frequency >= size:
        raise ValueError("frequency ({}) must be smaller than the number of graph nodes ({})".format(frequency, size))
    digest = hashlib.sha1()
    for array in (norm_adj.indptr, norm_adj.indices):
        digest.update(np.ascontiguousarray(array, dtype=np.int64).tobytes())
    path = os.path.join(SAVE_DIR, 'checkpoint', 'gplr', 'basis-{}-n{}-f{}-{}.npy'.format(
        configs['data']['name'], size, frequency, digest.hexdigest()[:10]))
    if os.path.exists(path):
        return np.load(path), os.path.basename(path)

    print("[gplr] low-pass graph filter: {} eigenvectors of a graph with {} nodes -> {}".format(frequency, size, path))
    start = np.random.RandomState(0).uniform(-1.0, 1.0, size)  # fixed start: the basis depends on the graph only
    values, vectors = scipy.sparse.linalg.eigsh(norm_adj.astype(np.float64), k=frequency, which='LA', tol=1e-5, v0=start)
    basis = np.ascontiguousarray(vectors[:, np.argsort(-values)], dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = "{}.{}.tmp.npy".format(path, os.getpid())
    np.save(tmp_path, basis)
    os.replace(tmp_path, path)
    return basis, os.path.basename(path) + " (computed now)"


def correlation_norm(x):
    """Frobenius norm of the correlations between the columns of x, every pair once (afd_utils.calculate_correlation)."""
    centered = x - x.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (x.shape[0] - 1)
    std = t.sqrt(t.diagonal(cov))
    corr = cov / (std.unsqueeze(1) * std.unsqueeze(0))
    return t.triu(corr, diagonal=1).norm()


class GPLR(BaseModel):
    def __init__(self, data_handler):
        super(GPLR, self).__init__(data_handler)
        self.layer_num = self._hp('layer_num')
        self.reg_weight = self._hp('reg_weight')
        self.afd_alpha = self._hp('afd_alpha')
        frequency = self._hp('frequency')

        persona_dir = os.path.join(configs['data']['dir'], self._hp('persona_dir'))
        self.persona_num, user_links, item_links = _persona_graph(persona_dir, self.user_num, self.item_num)
        norm_adj = _normalized_adj(data_handler.trn_mat, user_links, item_links, self.persona_num)
        basis, basis_note = _low_pass_basis(norm_adj, frequency)
        self.basis = t.from_numpy(basis).to(configs['device'])  # fixed input: neither a parameter nor state

        def normal(*shape):  # the release initializes every variable from N(0.01, 0.02^2)
            return nn.Parameter(t.empty(*shape).normal_(mean=0.01, std=0.02))

        self.user_embeds = normal(self.user_num, self.embedding_size)
        self.item_embeds = normal(self.item_num, self.embedding_size)
        self.persona_embeds = normal(self.persona_num, self.embedding_size)
        self.kernels = nn.ParameterList([normal(frequency) for _ in range(self.layer_num)])
        self.final_embeds = None

        summary = "{} personas (dir {}), {} user-persona links ({} of {} users), {} item-persona links ({} of {} items), basis {}".format(
            self.persona_num, persona_dir, len(user_links), len(np.unique(user_links[:, 0])), self.user_num,
            len(item_links), len(np.unique(item_links[:, 0])), self.item_num, basis_note)
        configs['model']['gplr_summary'] = summary  # picked up by the Logger's 'Model|' line
        print("[gplr] {}: {}".format(configs['data']['name'], summary))

    def _hp(self, key):
        """Dataset block of gplr.yml first (--override lands there), then the model block."""
        return self.hyper_config[key] if key in self.hyper_config else configs['model'][key]

    def _propagate(self):
        """Final embeddings of [users, items, personas]: E_0 + sum_l E_l / (l + 1)."""
        embeds = t.cat((self.user_embeds, self.item_embeds, self.persona_embeds), dim=0)
        pooled = embeds
        for layer, kernel in enumerate(self.kernels):
            embeds = self.basis @ (kernel.unsqueeze(1) * (self.basis.T @ embeds))
            pooled = pooled + embeds / (layer + 2)
        return pooled

    def forward(self):
        if self.final_embeds is None or self.training:
            self.final_embeds = self._propagate()
        return self.final_embeds

    def cal_loss(self, batch_data):
        self.final_embeds = None
        ancs, poss, negs = batch_data
        user_embeds, item_embeds, persona_embeds = t.split(
            self._propagate(), [self.user_num, self.item_num, self.persona_num], dim=0)

        bpr_loss = cal_bpr_loss(user_embeds[ancs], item_embeds[poss], item_embeds[negs])  # -sum log sigmoid(pos - neg)
        reg_loss = self.reg_weight * 0.5 * (
            self.user_embeds[ancs].square().sum() + self.item_embeds[poss].square().sum()
            + self.item_embeds[negs].square().sum() + self.persona_embeds.square().sum())
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        if self.afd_alpha > 0:
            afd_loss = self.afd_alpha * (
                correlation_norm(user_embeds) + correlation_norm(item_embeds) + correlation_norm(persona_embeds))
            loss = loss + afd_loss
            losses['afd_loss'] = afd_loss
        return loss, losses

    def full_predict(self, batch_data):
        embeds = self.forward()
        pck_users, train_mask = batch_data
        full_preds = embeds[pck_users.long()] @ embeds[self.user_num:self.user_num + self.item_num].T
        return self._mask_predict(full_preds, train_mask)
