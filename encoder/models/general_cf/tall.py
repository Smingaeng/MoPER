"""TALL (ECIR'24): end-to-end adaptive local learning, a mixture of MultVAE experts against mainstream bias.

Ported from https://github.com/JP-25/end-To-end-Adaptive-Local-Leanring-TALL- (model/MOE.py).  The method is
unchanged; what follows this repository instead of the original scripts is the data path, the Trainer with its
validation rule, and where the weight decay is applied (see reg_weight).

  * Experts: `expert_num` MultVAE auto-encoders over the user's input row.
  * Adaptive loss-driven gate: no gate network.  An expert's weight for a user is the softmax over experts of
    1 / (that expert's loss on the user's held-out items), in training and in prediction alike.
  * Adaptive weight module: after the first `gap_window` epochs, a user's weight in the reconstruction loss follows
    how fast the user's held-out loss has been falling over the last `loss_window` epochs.

The held-out items are interactions the experts neither read nor reconstruct (`gate_source`):

  * holdout: `holdout_ratio` of every user's training interactions; the rest is the input row.  The model then uses
    the training set only, like the other models here, and its validation and test scores are comparable with theirs.
  * valid: the validation interactions, as TALL is published (its 70/10/20 split uses the 10% both for the gate and
    for early stopping).  The input row is the whole training row, but the gate also reads interactions no other
    model sees at prediction time, and validation scores are optimistic.  Test items are never read in either case.

TALL trains on user batches: the model replaces the data handler's (user, item, negative) loader by a loader of
user ids (train.batch_size users per step).
"""
import numpy as np
import torch as t
import torch.nn.functional as F
import torch.utils.data as data
from torch import nn

from config.configurator import configs
from models.base_model import BaseModel
from models.loss_utils import reg_params


class UserData(data.Dataset):
    """One training sample = one user id."""

    def __init__(self, user_num):
        self.user_num = user_num

    def sample_negs(self):  # called by the Trainer every epoch; TALL samples no negatives
        pass

    def __len__(self):
        return self.user_num

    def __getitem__(self, idx):
        return (idx,)


class MultVAEExpert(nn.Module):
    """MultVAE: encoder -> (mean, log-variance) -> decoder -> item logits."""

    def __init__(self, item_num, enc_dims, dropout):
        super(MultVAEExpert, self).__init__()
        enc_dims = [item_num] + list(enc_dims)
        dec_dims = enc_dims[::-1]
        self.latent_size = enc_dims[-1]
        self.dropout = dropout

        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            last = i == len(enc_dims) - 2
            self.encoder.append(nn.Linear(d_in, d_out * 2 if last else d_out))
            if not last:
                self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(dec_dims) - 2:
                self.decoder.append(nn.Tanh())

    def forward(self, x):
        """Item logits and, in training, the batch-mean KL term (None otherwise)."""
        h = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)
        mu_q, logvar_q = h[:, :self.latent_size], h[:, self.latent_size:]
        std_q = t.exp(0.5 * logvar_q)
        epsilon = t.zeros_like(std_q).normal_(mean=0, std=0.01)
        output = mu_q + self.training * epsilon * std_q
        for layer in self.decoder:
            output = layer(output)

        kl_loss = None
        if self.training:
            kl_loss = (0.5 * (-logvar_q + t.exp(logvar_q) + t.pow(mu_q, 2) - 1)).sum(1).mean()
        return output, kl_loss


def split_holdout(trn_mat, ratio, seed):
    """(users, items) of the held-out training interactions: round(ratio * n) of a user's n items, at least one and
    at most n - 1, so that the input row is never empty (a user with one item keeps it and has no held-out item).
    The draw has its own generator: every build of the model in a run gets the same split."""
    rng = np.random.RandomState(seed)
    csr = trn_mat.tocsr()
    users, items = [], []
    for user in range(csr.shape[0]):
        row = csr.indices[csr.indptr[user]:csr.indptr[user + 1]]
        num = min(max(int(ratio * len(row) + 0.5), 1), len(row) - 1)
        if num > 0:
            users.append(np.full(num, user, dtype=np.int64))
            items.append(rng.choice(row, num, replace=False).astype(np.int64))
    return np.concatenate(users), np.concatenate(items)


class TALL(BaseModel):
    def __init__(self, data_handler):
        super(TALL, self).__init__(data_handler)
        device = configs['device']
        self.expert_num = self._hp('expert_num')
        self.reg_weight = self._hp('reg_weight')
        self.anneal_cap = self._hp('anneal_cap')
        self.total_anneal_steps = self._hp('total_anneal_steps')
        self.gap_window = self._hp('gap_window')
        self.alpha = self._hp('alpha')
        self.loss_window = self._hp('loss_window')

        self.experts = nn.ModuleList(
            [MultVAEExpert(self.item_num, self._hp('enc_dims'), self._hp('dropout')) for _ in range(self.expert_num)])

        # dense interaction rows [user_num, item_num]: inputs of the experts and the gate (fixed data, not state)
        self.inp_mat = t.from_numpy(data_handler.trn_mat.toarray()).float().to(device)
        gate_source = self._hp('gate_source')
        if gate_source == 'holdout':
            held_users, held_items = split_holdout(data_handler.trn_mat, self._hp('holdout_ratio'), configs['train']['seed'])
        elif gate_source == 'valid':
            val_lists = data_handler.valid_dataloader.dataset.user_pos_lists
            held_users = np.repeat(np.arange(self.user_num), [len(items) for items in val_lists])
            held_items = np.concatenate([np.asarray(items, dtype=np.int64) for items in val_lists])
        else:
            raise ValueError("gate_source must be 'holdout' or 'valid', got {!r}".format(gate_source))
        held_users, held_items = t.from_numpy(held_users).to(device), t.from_numpy(held_items).to(device)
        if gate_source == 'holdout':
            self.inp_mat[held_users, held_items] = 0.0
        self.held_mat = t.zeros(self.user_num, self.item_num, device=device)
        self.held_mat[held_users, held_items] = 1.0

        data_handler.train_dataloader = data.DataLoader(
            UserData(self.user_num), batch_size=configs['train']['batch_size'], shuffle=True, num_workers=0)

        # training state of the adaptive weight module and of the KL annealing
        self.epoch, self.seen_users, self.update_count = 0, 0, 0
        self.user_loss = t.ones(self.user_num, device=device)    # latest held-out loss of every user
        self.user_weight = t.ones(self.user_num, device=device)  # weights of the current epoch
        self.prev_loss, self.loss_change = None, None

    def _hp(self, key):
        """Dataset block of tall.yml first (--override lands there), then the model block."""
        return self.hyper_config[key] if key in self.hyper_config else configs['model'][key]

    @staticmethod
    def _held_loss(logits, x, held):
        """Loss of the logits on the held-out items, per user: sum over held-out items j of
        -log(exp(o_j) / sum_k exp(o_k * (1 - x_k))), written with logsumexp (same value, no overflow)."""
        return held.sum(1) * t.logsumexp(logits * (1 - x), dim=1) - (logits * held).sum(1)

    def forward(self, x, held):
        """Gate-weighted item logits of the batch users, and the gate-weighted KL term in training."""
        outputs, held_losses, kl_losses = [], [], []
        for expert in self.experts:
            output, kl_loss = expert(x)
            outputs.append(output)
            held_losses.append(self._held_loss(output, x, held))
            kl_losses.append(kl_loss)
        held_losses = t.stack(held_losses, dim=1)  # [batch, experts]
        # A user without held-out items has no loss to compare the experts by: uniform gate.  (TALL's own split
        # gives every user a validation item; here a user may have none, or a single training item.)
        has_held = (held.sum(1) > 0).unsqueeze(1)
        scores = t.where(has_held, 1 / t.where(has_held, held_losses, t.ones_like(held_losses)), t.zeros_like(held_losses))
        gate = F.softmax(scores, dim=1)
        output = t.matmul(gate.unsqueeze(1), t.stack(outputs, dim=1)).squeeze(1)

        kl_loss = t.matmul(gate, t.stack(kl_losses)).sum() if self.training else None
        return output, kl_loss

    def _start_epoch(self):
        """Adaptive weight module: weights of the coming epoch from the users' recent held-out-loss decrease."""
        self.epoch += 1
        if self.epoch <= self.gap_window or self.loss_change is None:
            self.user_weight = t.ones_like(self.user_loss)
        else:
            change = self.loss_change.mean(1)
            self.user_weight = ((change - change.mean() + 2 * self.alpha) / (2 * self.alpha)).clamp(min=1e-8)

    def _end_epoch(self):
        if self.prev_loss is not None:
            change = (self.prev_loss - self.user_loss).unsqueeze(1)
            self.loss_change = change if self.loss_change is None else t.cat((self.loss_change, change), dim=1)[:, -self.loss_window:]
        self.prev_loss = self.user_loss.clone()

    def cal_loss(self, batch_data):
        users = batch_data[0]
        if self.seen_users == 0:
            self._start_epoch()
        x, held = self.inp_mat[users], self.held_mat[users]
        anneal = self.anneal_cap
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)

        output, kl_loss = self.forward(x, held)
        self.user_loss[users] = self._held_loss(output, x, held).detach()

        weight = self.user_weight[users]
        ce_loss = -((F.log_softmax(output, 1) * x) * weight.view(-1, 1)).sum(1).sum() / weight.sum()
        kl_loss = anneal * kl_loss
        # TALL: Adam(weight_decay = reg / experts).  Adam adds weight_decay * p to the gradient, which is the
        # gradient of this term, so the optimizer block keeps weight_decay 0 as for the other models.
        reg_loss = 0.5 * self.reg_weight / self.expert_num * reg_params(self)
        loss = ce_loss + kl_loss + reg_loss

        self.update_count += 1
        self.seen_users += users.numel()
        if self.seen_users >= self.user_num:  # every user has been seen once: the epoch is over
            self._end_epoch()
            self.seen_users = 0
        losses = {'ce_loss': ce_loss, 'kl_loss': kl_loss, 'reg_loss': reg_loss}
        return loss, losses

    def full_predict(self, batch_data):
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        full_preds, _ = self.forward(self.inp_mat[pck_users], self.held_mat[pck_users])
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds
