"""SimGCL (SIGIR'22, "Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation").

Re-implemented from the authors' release, https://github.com/Coder-Yu/SELFRec (model/graph/SimGCL.py,
util/loss_torch.py).  The method is unchanged; what follows this repository instead of the release is the data path,
the (user, item, negative) batches of the data handler, and the Trainer with its validation rule.

  * Encoder (SimGCL_Encoder.forward): LightGCN propagation of the free user / item embeddings over the symmetrically
    normalised bipartite graph (no self-loops, the same matrix as the data handler's).  The final embedding is the
    MEAN of the n_layer propagated layers; the input layer is not part of it.
  * Views: the same propagation with the noise  sign(E) * unit(U(0, 1)) * eps  added after every layer.
  * Loss = BPR (mean of -log(1e-5 + sigmoid(pos - neg)) over the batch)
           + L2 (reg_form)
           + cl_weight * (InfoNCE over the batch's distinct users + InfoNCE over its distinct positive items) between
             two noisy views, cosine / temperature, the other rows of the batch as negatives.
  * L2, reg_form 'release': reg_weight * (||U_batch|| + ||I_batch||) / batch size, on the final embeddings of the batch
    users and positive items (l2_reg_loss of the release: the norm, not its square).
    reg_form 'params': reg_weight * (||U||^2 + ||I||^2) over the two embedding tables, the L2 term of the other models
    of this repository (loss_utils.reg_params).  Not in the release; the encoder, the views and the other two terms
    are the same under both forms.
"""
import torch as t
from torch import nn
import torch.nn.functional as F

from models.base_model import BaseModel
from models.loss_utils import reg_params

init = nn.init.xavier_uniform_


class SimGCL(BaseModel):
    def __init__(self, data_handler):
        super(SimGCL, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.reg_form = self.hyper_config['reg_form']
        if self.reg_form not in ('release', 'params'):
            raise ValueError("reg_form must be 'release' or 'params', got {!r}".format(self.reg_form))
        self.cl_weight = self.hyper_config['cl_weight']
        self.eps = self.hyper_config['eps']
        self.temperature = self.hyper_config['temperature']

        self.final_embeds = None
        self.is_training = False

    def forward(self, perturbed=False):
        ego_embeddings = t.cat([self.user_embeds, self.item_embeds], 0)
        all_embeddings = []
        for _ in range(self.layer_num):
            ego_embeddings = t.spmm(self.adj, ego_embeddings)
            if perturbed:
                random_noise = t.rand_like(ego_embeddings)
                ego_embeddings = ego_embeddings + t.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
        all_embeddings = t.mean(t.stack(all_embeddings, dim=1), dim=1)
        return all_embeddings[:self.user_num], all_embeddings[self.user_num:]

    def _infonce(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / self.temperature
        score = t.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def cal_loss(self, batch_data):
        self.is_training = True
        ancs, poss, negs = batch_data
        rec_user_emb, rec_item_emb = self.forward()
        user_emb, pos_item_emb, neg_item_emb = rec_user_emb[ancs], rec_item_emb[poss], rec_item_emb[negs]

        pos_score = (user_emb * pos_item_emb).sum(dim=1)
        neg_score = (user_emb * neg_item_emb).sum(dim=1)
        bpr_loss = t.mean(-t.log(10e-6 + t.sigmoid(pos_score - neg_score)))
        if self.reg_form == 'params':
            reg_loss = self.reg_weight * reg_params(self)
        else:
            reg_loss = self.reg_weight * (t.norm(user_emb, p=2) / user_emb.shape[0] + t.norm(pos_item_emb, p=2) / pos_item_emb.shape[0])

        u_idx, i_idx = t.unique(ancs), t.unique(poss)
        user_view_1, item_view_1 = self.forward(perturbed=True)
        user_view_2, item_view_2 = self.forward(perturbed=True)
        cl_loss = self.cl_weight * (self._infonce(user_view_1[u_idx], user_view_2[u_idx])
                                    + self._infonce(item_view_1[i_idx], item_view_2[i_idx]))

        loss = bpr_loss + reg_loss + cl_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss}
        return loss, losses

    def full_predict(self, batch_data):
        if self.is_training or self.final_embeds is None:   # once per evaluation pass
            self.final_embeds = self.forward()
            self.is_training = False
        user_embeds, item_embeds = self.final_embeds
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        full_preds = user_embeds[pck_users] @ item_embeds.T
        return self._mask_predict(full_preds, train_mask)
