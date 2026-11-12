"""HyperLLM (SIGIR'25) on HGCF: LLM-enhanced collaborative filtering in hyperbolic space.

Ported from https://github.com/Qin-lab-code/HyperLLM (phase1.py, phase1_model.py, main.py, model.py, sampler.py);
the Lorentz arithmetic and the Riemannian SGD are in models/lorentz.py.  The method is unchanged; what follows this
repository instead of the original scripts is the data path, the (user, item, negative) batches of the data handler,
and the Trainer with its validation rule.

Phase 1 (semantic hierarchy): the frozen LLM embeddings of users and items (usr_emb_np.pkl / itm_emb_np.pkl) pass a
mixture of linear experts that is trained with the margin loss of the HGCF encoder.  The best-epoch output is stored
in save/checkpoint/hyperllm/ and reused by every run with the same phase-1 settings.

Phase 2 (structural hierarchy): user and item embeddings start from the phase-1 output and are trained on the
hyperboloid together with tag embeddings.  Loss = margin loss on the user-item graph + margin loss on the tag graph
(tag-tag and tag-item edges) + contrast between the item representations of the two graphs.

The tag hierarchy is LLM output: data/<dataset>/<tag_dir>/{id_to_tag,tag_to_tag,tag_to_item}.json, built by
generation/hyperllm/generate_tags.py.
"""
import hashlib
import json
import os

import numpy as np
import scipy.sparse as sp
import torch as t
from torch import nn

from config.configurator import configs
from models import lorentz as L
from models.base_model import BaseModel
from trainer.trainer import SAVE_DIR, Trainer


def _row_normalized_adj(mat):
    """D^-1 (A + I) as a torch sparse tensor on the run device."""
    mat = mat + sp.eye(mat.shape[0])
    inv_degree = np.power(np.array(mat.sum(1)), -1.0).flatten()
    inv_degree[np.isinf(inv_degree)] = 0.0
    mat = sp.diags(inv_degree).dot(mat).tocoo()
    idxs = t.from_numpy(np.vstack((mat.row, mat.col)).astype(np.int64))
    vals = t.from_numpy(mat.data.astype(np.float32))
    return t.sparse_coo_tensor(idxs, vals, mat.shape).coalesce().to(configs['device'])


def _user_item_adj(trn_mat):
    """Bipartite adjacency over the nodes [users, items]."""
    user_num = trn_mat.shape[0]
    rows = np.concatenate((trn_mat.row, trn_mat.col + user_num))
    cols = np.concatenate((trn_mat.col + user_num, trn_mat.row))
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(sum(trn_mat.shape),) * 2).tocsr().astype(np.float32)


def _tag_graph(tag_dir, item_num):
    """Tag graph over the nodes [tags, items], from the LLM tag files.

    Returns the tag count, the tag-tag adjacency [tags, tags], the tag-item adjacency and the joint adjacency (both
    [tags + items, tags + items]).  As in main.py, every listed relation is inserted in both directions and repeated
    entries add up in the joint adjacency: tag_to_tag.json lists an edge under both of its tags, so a tag-tag edge
    weighs 2 and a tag-item edge 1.
    """
    files = {}
    for name in ('id_to_tag', 'tag_to_tag', 'tag_to_item'):
        path = os.path.join(tag_dir, name + '.json')
        if not os.path.exists(path):
            raise FileNotFoundError(
                "HyperLLM needs the LLM tag hierarchy {}; build it with generation/hyperllm/generate_tags.py".format(path))
        with open(path, 'r', encoding='utf-8') as f:
            files[name] = json.load(f)
    tag_num = len(files['id_to_tag'])

    def pairs(mapping):
        left = np.array([int(key) for key, values in mapping.items() for _ in values], dtype=np.int64)
        right = np.array([value for values in mapping.values() for value in values], dtype=np.int64)
        return left, right

    tag_a, tag_b = pairs(files['tag_to_tag'])
    tag_c, item = pairs(files['tag_to_item'])
    if len(tag_a) == 0 or len(tag_c) == 0:
        raise ValueError("{} holds no tag-tag or no tag-item relation".format(tag_dir))
    if max(tag_a.max(), tag_b.max(), tag_c.max()) >= tag_num or item.max() >= item_num or min(tag_b.min(), item.min()) < 0:
        raise ValueError("{} does not match this dataset ({} tags, {} items)".format(tag_dir, tag_num, item_num))

    def symmetric(rows, cols, size):
        rows, cols = np.concatenate((rows, cols)), np.concatenate((cols, rows))
        return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size)).tocsr().astype(np.float32)

    size = tag_num + item_num
    tag_tag = symmetric(tag_a, tag_b, tag_num)
    tag_tag.setdiag(0)
    tag_tag.eliminate_zeros()
    tag_item = symmetric(tag_c, tag_num + item, size)
    joint = symmetric(np.concatenate((tag_a, tag_c)), np.concatenate((tag_b, tag_num + item)), size)
    return tag_num, tag_tag, tag_item, joint


class TripleSampler:
    """(anchor, positive, negative) triples of a graph, as sampler.py draws them.

    The positives walk through a shuffled list of the first half of the adjacency entries in row order: for a
    bipartite adjacency over [anchors, others] that is every edge once, seen from its anchor; for the symmetric
    tag-tag adjacency it is, as in the original, the rows of the low tag ids.  The negative is uniform over the
    candidate nodes [cand_low, cand_high) that are not neighbours of the anchor.
    """

    def __init__(self, adj, cand_low, cand_high, batch_size):
        adj = adj.tocsr()
        adj.sort_indices()
        coo = adj.tocoo()
        pairs = np.stack((coo.row, coo.col), axis=1)[:adj.nnz // 2].astype(np.int64)
        candidates = adj[:, cand_low:cand_high].tocsr()
        candidates.sort_indices()
        self.cand_low, self.cand_num = cand_low, cand_high - cand_low
        self.indptr, self.indices = candidates.indptr.astype(np.int64), candidates.indices.astype(np.int64)
        self.degree = np.diff(self.indptr)
        # an anchor linked to every candidate has no negative: its edges cannot form a triple
        self.pairs = pairs[self.degree[pairs[:, 0]] < self.cand_num]
        self.dropped = len(pairs) - len(self.pairs)
        if len(self.pairs) == 0:
            raise ValueError("TripleSampler: no edge with a valid negative")
        self.batch_size = batch_size
        self.order, self.cursor = np.random.permutation(len(self.pairs)), 0

    def _non_neighbours(self, anchors):
        """One uniform draw per anchor among the candidates it is not linked to (exact, no rejection loop): the
        r-th non-neighbour is r + (number of neighbours before it), found by bisection on neighbour[j] - j <= r."""
        start, degree = self.indptr[anchors], self.degree[anchors]
        rank = np.floor(np.random.random(len(anchors)) * (self.cand_num - degree)).astype(np.int64)
        low, high = np.zeros_like(degree), degree.copy()
        while True:
            active = low < high
            if not active.any():
                break
            mid = (low + high) // 2
            before = self.indices[np.minimum(start + mid, len(self.indices) - 1)] - mid
            right = active & (before <= rank)
            high = np.where(active & ~right, mid, high)
            low = np.where(right, mid + 1, low)
        return rank + low

    def next_batch(self):
        size = self.batch_size
        if size >= len(self.pairs):
            idx = np.random.randint(len(self.pairs), size=size)
        else:
            if self.cursor + size > len(self.order):
                self.order, self.cursor = np.random.permutation(len(self.pairs)), 0
            idx = self.order[self.cursor:self.cursor + size]
            self.cursor += size
        anchors, positives = self.pairs[idx, 0], self.pairs[idx, 1]
        return np.stack((anchors, positives, self.cand_low + self._non_neighbours(anchors)), axis=1)


def propagate(adj, embeds, layer_num):
    """HGCF encoder: average the neighbours in the tangent space of the origin, sum the 1..L-hop results and map
    the sum onto the hyperboloid."""
    layers = [embeds]
    for _ in range(layer_num):
        layers.append(t.spmm(adj, layers[-1]))
    return L.expmap0(sum(layers[1:]))


def margin_loss(out, triples, margin):
    """sum of max(d^2(anchor, positive) - d^2(anchor, negative) + margin, 0)."""
    anchors, positives, negatives = out[triples[:, 0]], out[triples[:, 1]], out[triples[:, 2]]
    loss = L.sqdist(anchors, positives) - L.sqdist(anchors, negatives) + margin
    return loss.masked_fill(loss < 0, 0).sum()


def _node_triples(batch_data, user_num):
    """(user, positive item, negative item) of the data handler -> node ids of the graph [users, items]."""
    ancs, poss, negs = batch_data
    return t.stack((ancs, poss + user_num, negs + user_num), dim=1)


def _check_finite(loss):
    if not bool(t.isfinite(loss)):
        raise FloatingPointError("HyperLLM: the training loss is {} (the original run stops here)".format(float(loss)))


class MoE(nn.Module):
    """Mixture of linear experts with a linear softmax gate."""

    def __init__(self, input_dim, output_dim, expert_num):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(expert_num)])
        self.gating_network = nn.Linear(input_dim, expert_num)

    def forward(self, x):
        weights = t.softmax(self.gating_network(x), dim=-1)
        outputs = t.stack([expert(x) for expert in self.experts], dim=1)
        return t.sum(outputs * weights.unsqueeze(2), dim=1)


class SemanticAdapter(nn.Module):
    """Phase 1: the mixture of experts maps the frozen LLM embeddings of all users and items to the embedding size;
    only the mixture is trained, through the HGCF encoder and its margin loss on the user-item graph."""

    def __init__(self, semantic, adj, user_num, layer_num, margin, embedding_size, expert_num):
        super(SemanticAdapter, self).__init__()
        self.semantic, self.adj = semantic, adj  # fixed inputs: neither parameters nor state
        self.user_num, self.layer_num, self.margin = user_num, layer_num, margin
        self.transform = MoE(semantic.shape[1], embedding_size, expert_num)
        self.final_embeds = None

    def embeddings(self):
        return self.transform(self.semantic)

    def cal_loss(self, batch_data):
        self.final_embeds = None
        out = propagate(self.adj, self.embeddings(), self.layer_num)
        return margin_loss(out, _node_triples(batch_data, self.user_num), self.margin)

    def full_predict(self, batch_data):
        if self.final_embeds is None:
            self.final_embeds = propagate(self.adj, self.embeddings(), self.layer_num)
        pck_users, train_mask = batch_data
        preds = -L.sqdist_multi(self.final_embeds[pck_users.long()], self.final_embeds[self.user_num:])
        return preds * (1 - train_mask) - 1e8 * train_mask


class HyperLLM(BaseModel):
    def __init__(self, data_handler):
        super(HyperLLM, self).__init__(data_handler)
        device = configs['device']
        self.layer_num = self._hp('layer_num')
        self.margin = self._hp('margin')
        self.tag_margin = self._hp('tag_margin')
        self.cl_weight = self._hp('cl_weight')
        self.cl_temperature = self._hp('cl_temperature')

        self.adj = _row_normalized_adj(_user_item_adj(data_handler.trn_mat))
        # read first: a missing tag hierarchy stops the run before phase 1 is trained
        tag_dir = os.path.join(configs['data']['dir'], self._hp('tag_dir'))
        self.tag_num, tag_tag, tag_item, joint = _tag_graph(tag_dir, self.item_num)

        # ---- phase 1: semantic embeddings -> initial user / item embeddings (cached) ----
        init = self._phase1_embeddings(data_handler)
        self.user_embeds = L.ManifoldParameter(L.expmap0(init['user'].to(device)))
        self.item_embeds = L.ManifoldParameter(L.expmap0(init['item'].to(device)))

        # ---- phase 2: tag hierarchy ----
        self.tag_adj = _row_normalized_adj(joint)
        scale = self._hp('tag_init_scale')
        tag_init = t.empty(self.tag_num, self.embedding_size, device=device).uniform_(-scale, scale)
        self.tag_embeds = L.ManifoldParameter(L.expmap0(tag_init))
        # per training step: as many tag triples as spread every tag-graph adjacency entry over one epoch
        batch_num = len(data_handler.train_dataloader)
        self.tag_tag_sampler = TripleSampler(tag_tag, 0, self.tag_num, -(-tag_tag.nnz // batch_num))
        self.tag_item_sampler = TripleSampler(tag_item, self.tag_num, self.tag_num + self.item_num, -(-tag_item.nnz // batch_num))

        self.final_embeds = None
        summary = "tags={} (dir {}), tag-tag entries={}, tag-item entries={}, tag triples per step={}+{}, phase 1: {}".format(
            self.tag_num, tag_dir, tag_tag.nnz, tag_item.nnz, self.tag_tag_sampler.batch_size,
            self.tag_item_sampler.batch_size, init['note'])
        dropped = self.tag_tag_sampler.dropped + self.tag_item_sampler.dropped
        if dropped:
            summary += "; {} edges of tags linked to every candidate are not sampled".format(dropped)
        configs['model']['hyperllm_summary'] = summary  # picked up by the Logger's 'Model|' line
        print("[hyperllm] {}: {}".format(configs['data']['name'], summary))

    def _hp(self, key):
        """Dataset block of hyperllm.yml first (--override lands there), then the model block."""
        return self.hyper_config[key] if key in self.hyper_config else configs['model'][key]

    # ------------------------------------------------------------------ phase 1
    def _semantic_embeds(self):
        """LLM embeddings of [users, items], through the exp and log maps at the origin as the original stores and
        reads them (the first coordinate is the time-like one and becomes 0)."""
        if 'usrprf_embeds' not in configs or 'itmprf_embeds' not in configs:
            raise FileNotFoundError("HyperLLM needs usr_emb_np.pkl and itm_emb_np.pkl in {}".format(configs['data']['dir']))
        raw = np.concatenate((configs['usrprf_embeds'], configs['itmprf_embeds']), axis=0)
        if raw.shape[0] != self.user_num + self.item_num:
            raise ValueError("usr_emb_np.pkl / itm_emb_np.pkl do not match the interaction matrix")
        with t.no_grad():
            return L.logmap0(L.expmap0(t.from_numpy(raw).float().to(configs['device'])))

    def _phase1_embeddings(self, data_handler):
        train, optim = configs['train'], configs['optimizer']
        spec = {
            'dataset': configs['data']['name'], 'users': self.user_num, 'items': self.item_num,
            'semantic': [int(configs['usrprf_embeds'].shape[1]) if 'usrprf_embeds' in configs else 0,
                         round(float(np.abs(configs['itmprf_embeds'][:64]).sum()), 4) if 'itmprf_embeds' in configs else 0],
            'embedding_size': self.embedding_size, 'layer_num': self.layer_num,
            'expert_num': self._hp('phase1_expert_num'), 'margin': self._hp('phase1_margin'),
            'optimizer': [optim['lr'], optim['momentum'], optim['weight_decay']],
            'train': [train['seed'], train['batch_size'], train['epoch'], train['test_step'], train['patience'],
                      train.get('early_stop_metric', 'recall_last')],
        }
        digest = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
        path = os.path.join(SAVE_DIR, 'checkpoint', 'hyperllm', 'phase1-{}-k{}-d{}-m{}-seed{}-{}.pt'.format(
            spec['dataset'], spec['expert_num'], spec['embedding_size'], spec['margin'], train['seed'], digest))
        if os.path.exists(path):
            cached = t.load(path, map_location='cpu')
            cached['note'] = "{} (best epoch {}, validation score {:.4f})".format(os.path.basename(path), cached['epoch'], cached['score'])
            return cached

        print("[hyperllm] phase 1: training the semantic adapter ({} experts, margin {}) -> {}".format(
            spec['expert_num'], spec['margin'], path))
        # phase 1 has its own random stream (train.seed): phase 2 starts from the same state whether phase 1 was
        # trained now or read from the cache
        rng = (t.random.get_rng_state(), t.cuda.get_rng_state_all() if t.cuda.is_available() else None, np.random.get_state())
        t.manual_seed(train['seed'])
        t.cuda.manual_seed_all(train['seed'])
        np.random.seed(train['seed'])

        net = SemanticAdapter(self._semantic_embeds(), self.adj, self.user_num, self.layer_num, spec['margin'],
                              self.embedding_size, spec['expert_num']).to(configs['device'])
        optimizer = L.RiemannianSGD(net.parameters(), lr=optim['lr'], momentum=optim['momentum'], weight_decay=optim['weight_decay'])
        selector = Trainer(data_handler, None)  # for its validation metric and early-stopping score
        loader = data_handler.train_dataloader
        best, waited = {'score': -1e9}, 0
        for epoch in range(train['epoch']):
            loader.dataset.sample_negs()
            net.train()
            ep_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                loss = net.cal_loss([x.long().to(configs['device']) for x in batch])
                _check_finite(loss)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
            if epoch % train['test_step'] == 0:
                net.eval()
                with t.no_grad():
                    score = float(selector._early_stop_score(selector.metric.eval(net, data_handler.valid_dataloader)))
                    print("[hyperllm] phase 1 epoch {:3d}: loss {:.1f}, validation score {:.4f}".format(epoch, ep_loss, score))
                    if score > best['score']:
                        embeds = net.embeddings().cpu()
                        best = {'user': embeds[:self.user_num].clone(), 'item': embeds[self.user_num:].clone(),
                                'epoch': epoch, 'score': score, 'spec': spec}
                        waited = 0
                    else:
                        waited += 1
                if waited == train['patience']:
                    break

        t.random.set_rng_state(rng[0])
        if rng[1] is not None:
            t.cuda.set_rng_state_all(rng[1])
        np.random.set_state(rng[2])

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = "{}.{}.tmp".format(path, os.getpid())
        t.save(best, tmp_path)
        os.replace(tmp_path, path)
        best['note'] = "{} (trained now: best epoch {}, validation score {:.4f})".format(os.path.basename(path), best['epoch'], best['score'])
        return best

    # ------------------------------------------------------------------ phase 2
    def forward(self):
        """Hyperboloid representations of [users, items] from the user-item graph."""
        if self.final_embeds is None or self.training:
            tangent = t.cat((L.logmap0(self.user_embeds), L.logmap0(self.item_embeds)), dim=0)
            self.final_embeds = propagate(self.adj, tangent, self.layer_num)
        return self.final_embeds

    def _contrast_loss(self, h1, h2):
        """InfoNCE over all items between the two graphs' item representations, similarity = -d^2 / temperature."""
        pos_score = t.exp(-L.sqdist(h1, h2) / self.cl_temperature)
        tot_score = t.exp(-L.sqdist_multi(h1, h2) / self.cl_temperature).sum(dim=1)
        return -t.log(pos_score / tot_score).sum()

    def cal_loss(self, batch_data):
        self.final_embeds = None
        triples = _node_triples(batch_data, self.user_num)
        tag_triples = np.vstack((self.tag_tag_sampler.next_batch(), self.tag_item_sampler.next_batch()))
        tag_triples = t.from_numpy(tag_triples).to(triples.device)

        item_tangent = L.logmap0(self.item_embeds)
        out = propagate(self.adj, t.cat((L.logmap0(self.user_embeds), item_tangent), dim=0), self.layer_num)
        tag_out = propagate(self.tag_adj, t.cat((L.logmap0(self.tag_embeds), item_tangent), dim=0), self.layer_num)

        ui_loss = margin_loss(out, triples, self.margin)
        tag_loss = margin_loss(tag_out, tag_triples, self.tag_margin)
        cl_loss = self.cl_weight * self._contrast_loss(out[self.user_num:], tag_out[self.tag_num:])
        loss = ui_loss + tag_loss + cl_loss
        _check_finite(loss)
        losses = {'ui_loss': ui_loss, 'tag_loss': tag_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        out = self.forward()
        pck_users, train_mask = batch_data
        full_preds = -L.sqdist_multi(out[pck_users.long()], out[self.user_num:])
        return self._mask_predict(full_preds, train_mask)
