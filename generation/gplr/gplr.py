"""GPLR (SIGIR'25, "You Are What You Bought"): personas for every user from the LLM labels of a few prototype users.

Ported from https://github.com/Hanc1999/YouAreWhatYouBought (src/random_walker/ReverseRW.py, PageRank.py,
Evaluation.py, src/utils/GraphMaker.py and the driver cells of src/gplr.ipynb).  The method is unchanged; the node
objects of the release became arrays, and the oracle that looks answers up became the caller's LLM round.

  * Affinity of an unlabeled user u to a persona p (one user -> item -> user step of a random walk, then the
    persona of the prototype reached):
        psi[u, p] = sum_{t in N(u)} 1/|N(u)| * sum_{l in N(t), l prototype} 1/|N(t)| * phi[l, p] / |phi[l]|
    N(t) counts the users of the graph (prototypes and unlabeled users).  `epsilon = 0` evaluates the sum exactly
    (the paper's setting for all but its largest dataset); `epsilon > 0` runs RevAff, the reverse push from the
    persona that stops when no node holds a residue of epsilon / 3 or more.
  * DUSample: the next users to label are those with the largest
        entropy(Q_u) + KL(Q_hat || Q_u),
    Q_u = the user's normalized affinities, Q_hat = the persona distribution of the labels collected so far.
  * Prediction: the affinities are multiplied by (max_q n_q / n_p) ** beta (n_p = prototypes labeled p), and an
    unlabeled user receives the top_k personas.

Kept from the release: an affinity that no push reached counts as 1e-12 (so a user with no prototype among the
co-buyers of its items receives the top_k rarest personas); the sampling scores use no de-bias factor; a persona
no prototype was labeled with cannot be predicted; users the LLM called "Unrepresentable" are no prototypes,
leave the sampling scope, stay in the graph of the final prediction as unlabeled users and receive no persona.

The release looks its labels up in a table that holds an answer for every user and samples among the users known
to be representable.  With a real LLM budget that is not known in advance: here every user can be drawn, a drawn
user counts against the budget whatever the LLM answers, and is not drawn again.
"""
import heapq

import numpy as np
import scipy.sparse as sp
import scipy.stats

FLOOR = 1e-12  # affinity of a (user, persona) pair that received no mass


class _MaxHeap:
    """Max-heap with updatable priorities (ReverseRW.max_heapq): an update removes the entry lazily and pushes a
    new one.  Entries are [-priority, node id]; equal priorities are ordered by the node id string as in the release."""
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
    """The graph of ReverseRW.construct_graph, for a walk  unlabeled user -> item -> user -> persona.

    Layer 3: the unlabeled users, each pointing to its items.  Layer 2: the items, each pointing to every graph
    user that bought it (prototypes, and the unlabeled users as layer-1 copies that lead nowhere).  Layer 1: the
    prototypes, each pointing to its personas.  Layer 0: the personas.
    """

    def __init__(self, trn, unlabeled, labeled, labels):
        trn = trn.tocsr()
        self.unlabeled, self.labeled, self.labels = np.asarray(unlabeled, dtype=np.int64), list(labeled), labels
        in_graph = np.concatenate((self.unlabeled, np.asarray(self.labeled, dtype=np.int64))).astype(np.int64)
        self.user_items = {int(u): trn.indices[trn.indptr[u]:trn.indptr[u + 1]] for u in in_graph}
        self.user_degree = np.diff(trn.indptr)
        # out-degree of an item: its buyers among the graph users
        self.item_out_degree = np.asarray(trn[in_graph].sum(axis=0)).ravel()
        # in-neighbours of an item: the unlabeled users that bought it
        self.item_unlabeled = trn[self.unlabeled].tocsc()
        self.persona_users = {}
        for user in self.labeled:
            for persona in set(labels[user]):
                self.persona_users.setdefault(persona, []).append(user)

    def reverse_push(self, persona, epsilon):
        """ReverseRW.reverse_rw_3.reverse_random_walk_single: the mass 1 placed on the persona is pushed backwards
        along the edges, always from the node holding most; a node pushes residue / out-degree(neighbour) to each
        in-neighbour.  Returns {position of the unlabeled user: affinity} for the users that were reached."""
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
    """[len(unlabeled), len(personas)] affinities of the unlabeled users to the given personas
    (ReverseRW.reverse_rw_3.reverse_random_walk_all); pairs without mass hold FLOOR."""
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
    item_scores = sp.diags(inv_item) @ (trn[labeled].T @ phi)  # [items, personas]
    user_degree = np.asarray(trn[unlabeled].sum(axis=1)).ravel()
    inv_user = np.divide(1.0, user_degree, out=np.zeros_like(user_degree), where=user_degree > 0)
    psi = sp.diags(inv_user) @ (trn[unlabeled] @ item_scores)
    return np.where(psi > 0, psi, FLOOR)


def _seen_personas(labels, labeled):
    """Personas of the prototypes in order of first appearance (PageRank.assemble_label_matrix) and the number of
    prototypes labeled with each."""
    order, count = [], {}
    for user in labeled:
        for persona in dict.fromkeys(labels[user]):
            if persona not in count:
                order.append(persona)
                count[persona] = 0
            count[persona] += 1
    return order, np.array([count[persona] for persona in order], dtype=np.float64)


class GPLR:
    """Algorithm 1 of the paper around the caller's LLM: `next_users` -> label them -> `add_labels`, for every
    round; then `predict`.

    trn: [users, items] training interactions.  Labels are lists of persona ids in [0, persona_num).
    """

    def __init__(self, trn, persona_num, epsilon=0.0, seed=0):
        self.trn = sp.csr_matrix(trn)
        self.user_num = self.trn.shape[0]
        self.persona_num = persona_num
        self.epsilon = epsilon
        self.rng = np.random.RandomState(seed)
        self.labels = {}              # prototype user -> persona ids
        self.unrepresentable = set()  # users the LLM found no persona for
        self.asked = set()            # users drawn in an earlier round, answered or not

    def _scope(self):
        """Users that can still be sampled (Iterative_Sampler_rev.sample_scope minus the labeled users)."""
        taken = np.fromiter(set(self.labels) | self.unrepresentable | self.asked, dtype=np.int64)
        return np.setdiff1d(np.arange(self.user_num), taken)

    def next_users(self, amount):
        """DUSample: `amount` users to label next.  Random while there is no prototype (trivial_rank), afterwards
        the users with the largest diversity-uncertainty score (criteria_rank_3)."""
        users = self._draw(amount)
        self.asked.update(users)  # a drawn user is not drawn again, also when the LLM gives no answer for it
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
        """answers: {user: persona ids}; an empty list marks the user as unrepresentable."""
        for user, personas in answers.items():
            personas = list(dict.fromkeys(int(p) for p in personas))
            if personas:
                self.labels[int(user)] = personas
            else:
                self.unrepresentable.add(int(user))

    def predict(self, top_k=5, beta=0.5):
        """{user: persona ids} for every user that is neither a prototype nor unrepresentable
        (ReverseRW.persona_prediction_rev + GraphMaker.make_app), and the number of those users that no
        prototype reaches."""
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
