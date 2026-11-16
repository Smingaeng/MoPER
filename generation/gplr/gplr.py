import heapq

import numpy as np
import scipy.sparse as sp
import scipy.stats

FLOOR = 1e-12


class _MaxHeap:
    REMOVED = '<removed-task>'

    def __init__(self):
        self.data, self.entry_finder = [], {}

    def push(self, nid, priority):
        if nid in self.entry_finder:
            self.entry_finder.pop(nid)[-1] = self.REMOVED
        entry = [-priority, nid]
        self.entry_finder[nid] = entry
        heapq.heappush(self.data, entry)

    def pop(self):
        while self.data:
            minus_p, nid = heapq.heappop(self.data)
            if nid != self.REMOVED:
                del self.entry_finder[nid]
                return -minus_p, nid
        return None

    def max(self):
        while self.data:
            minus_p, nid = self.data[0]
            if nid == self.REMOVED:
                heapq.heappop(self.data)
            else:
                return -minus_p
        return -1


class _LayeredGraph:
    def __init__(self, trn, unlabeled, labeled, labels):
        trn = trn.tocsr()
        self.unlabeled, self.labeled, self.labels = np.asarray(unlabeled, dtype=np.int64), list(labeled), labels
        in_graph = np.concatenate((self.unlabeled, np.asarray(self.labeled, dtype=np.int64))).astype(np.int64)
        self.user_items = {int(u): trn.indices[trn.indptr[u]:trn.indptr[u + 1]] for u in in_graph}
        self.user_degree = np.diff(trn.indptr)
        self.item_out_degree = np.asarray(trn[in_graph].sum(axis=0)).ravel()
        self.item_unlabeled = trn[self.unlabeled].tocsc()
        self.persona_users = {}
        for user in self.labeled:
            for persona in set(labels[user]):
                self.persona_users.setdefault(persona, []).append(user)

    def reverse_push(self, persona, epsilon):
        threshold = epsilon / 3
        reached = {}
        residue = {'p': 1.0}
        heap = _MaxHeap()
        heap.push('p', 1.0)
        unlabeled_csc = self.item_unlabeled
        while heap.max() >= threshold:
            mass, nid = heap.pop()
            if nid == 'p':
                for user in self.persona_users.get(persona, ()):
                    key = 'l' + str(user)
                    residue[key] = residue.get(key, 0.0) + mass / len(set(self.labels[user]))
                    heap.push(key, residue[key])
            elif nid[0] == 'l':
                for item in self.user_items[int(nid[1:])]:
                    key = 't' + str(item)
                    residue[key] = residue.get(key, 0.0) + mass / self.item_out_degree[item]
                    heap.push(key, residue[key])
            else:
                item = int(nid[1:])
                for pos in unlabeled_csc.indices[unlabeled_csc.indptr[item]:unlabeled_csc.indptr[item + 1]]:
                    reached[pos] = reached.get(pos, 0.0) + mass / self.user_degree[self.unlabeled[pos]]
            residue[nid] = 0.0
        return reached


def affinity(trn, unlabeled, labeled, labels, personas, epsilon):
    unlabeled = np.asarray(unlabeled, dtype=np.int64)
    if epsilon > 0:
        graph = _LayeredGraph(trn, unlabeled, labeled, labels)
        psi = np.full((len(unlabeled), len(personas)), FLOOR)
        for col, persona in enumerate(personas):
            reached = graph.reverse_push(persona, epsilon)
            if reached:
                psi[list(reached.keys()), col] = list(reached.values())
        return psi

    trn = trn.tocsr().astype(np.float64)
    labeled = np.asarray(labeled, dtype=np.int64)
    column = {persona: col for col, persona in enumerate(personas)}
    phi = np.zeros((len(labeled), len(personas)))
    for row, user in enumerate(labeled):
        own = set(labels[int(user)])
        for persona in own:
            phi[row, column[persona]] = 1.0 / len(own)
    item_degree = np.asarray(trn[np.concatenate((unlabeled, labeled))].sum(axis=0)).ravel()
    inv_item = np.divide(1.0, item_degree, out=np.zeros_like(item_degree), where=item_degree > 0)
    item_scores = sp.diags(inv_item) @ (trn[labeled].T @ phi)
    user_degree = np.asarray(trn[unlabeled].sum(axis=1)).ravel()
    inv_user = np.divide(1.0, user_degree, out=np.zeros_like(user_degree), where=user_degree > 0)
    psi = sp.diags(inv_user) @ (trn[unlabeled] @ item_scores)
    return np.where(psi > 0, psi, FLOOR)


def _seen_personas(labels, labeled):
    order, count = [], {}
    for user in labeled:
        for persona in dict.fromkeys(labels[user]):
            if persona not in count:
                order.append(persona)
                count[persona] = 0
            count[persona] += 1
    return order, np.array([count[persona] for persona in order], dtype=np.float64)


class GPLR:
    def __init__(self, trn, persona_num, epsilon=0.0, seed=0):
        self.trn = sp.csr_matrix(trn)
        self.user_num = self.trn.shape[0]
        self.persona_num = persona_num
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        self.labels = {}
        self.unrepresentable = set()
        self.asked = set()

    def _scope(self):
        taken = np.fromiter(set(self.labels) | self.unrepresentable | self.asked, dtype=np.int64)
        return np.setdiff1d(np.arange(self.user_num), taken)

    def next_users(self, amount):
        users = self._draw(amount)
        self.asked.update(users)
        return users

    def _draw(self, amount):
        candidates = self._scope()
        if not self.labels:
            return self.rng.permutation(candidates)[:amount].tolist()
        labeled = sorted(self.labels)
        seen, count = _seen_personas(self.labels, labeled)
        current = np.zeros(self.persona_num)
        current[seen] = count
        current /= current.sum()
        scores = np.zeros((len(candidates), self.persona_num))
        scores[:, seen] = affinity(self.trn, candidates, labeled, self.labels, seen, self.epsilon)
        scores /= scores.sum(axis=1, keepdims=True)
        du = scipy.stats.entropy(scores, axis=1) + scipy.stats.entropy(
            np.broadcast_to(current, scores.shape), scores + 1e-16, axis=1)
        return candidates[np.argsort(du)[::-1][:amount]].tolist()

    def add_labels(self, answers):
        for user, personas in answers.items():
            personas = list(dict.fromkeys(int(p) for p in personas))
            if personas:
                self.labels[int(user)] = personas
            else:
                self.unrepresentable.add(int(user))

    def predict(self, top_k=5, beta=0.5):
        if not self.labels:
            raise ValueError("GPLR.predict: no prototype user has a persona label")
        labeled = sorted(self.labels)
        seen, count = _seen_personas(self.labels, labeled)
        unlabeled = np.setdiff1d(np.arange(self.user_num), labeled)
        scores = affinity(self.trn, unlabeled, labeled, self.labels, seen, self.epsilon)
        isolated = (scores <= FLOOR).all(axis=1)
        if beta > 0:
            scores = scores * (count.max() / count) ** beta
        ranked = np.argsort(-scores, axis=1, kind='stable')[:, :top_k]
        seen = np.asarray(seen)
        keep = np.array([user not in self.unrepresentable for user in unlabeled], dtype=bool)
        predicted = {int(user): seen[ranked[row]].tolist() for row, user in enumerate(unlabeled) if keep[row]}
        return predicted, int((isolated & keep).sum())
