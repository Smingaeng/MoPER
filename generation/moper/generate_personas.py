import argparse
import collections
import json
import math
import os
import pickle
import random
import re
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

API_KEY = ""
MODEL = "claude-haiku-5-5"
EFFORT = "medium"
MAX_TOKENS = 6000
BATCH_LIMIT = 5000
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(HERE))
DATA_ROOT = os.path.join(REPO_DIR, "data")

DATASETS = {
    "amazon_book": dict(
        domain="Amazon Books",
        interaction="the user wrote a review for the book (treated as implicit feedback)",
        cat_field="genres",
        person_word="reader",
    ),
    "amazon_movie": dict(
        domain="Amazon Movies & TV",
        interaction="the user wrote a review for the movie or show (treated as implicit feedback)",
        cat_field="genres",
        person_word="viewer",
    ),
    "yelp": dict(
        domain="Yelp local businesses (restaurants, bars, cafes, shops and services)",
        interaction="the user wrote a review for the business (treated as implicit feedback)",
        cat_field="categories",
        person_word="diner",
    ),
}

TIER_TOP = 90
TIER_POPULAR = 70
TIER_MIDDLE = 40
MEDIAN_CUT = 50
OVERALL_ITEMS = 12
MODE_ITEMS = 8
MIN_MODE_ITEMS = 3
PROFILE_CHARS = 200
REVIEW_CHARS = 220
TITLE_CHARS = 90
MIN_COMMUNITY_USERS = 5


def load_raw(ds, raw_root, communities=None):
    root = os.path.join(raw_root, ds)
    trn = sp.csr_matrix(pickle.load(open(os.path.join(root, "trn_mat.pkl"), "rb")))
    items = {}
    cat_field = DATASETS[ds]["cat_field"]
    for line in open(os.path.join(root, f"{ds}_item.jsonl")):
        r = json.loads(line)
        iid = int(r["iid"])
        cats = [c.strip() for c in str(r.get(cat_field) or r.get("categories") or "").split(",") if c.strip()]
        rec = dict(name=(r.get("name") or r.get("title") or "").strip(), cats=cats)
        if r.get("city"):
            rec["city"] = r["city"]
        if r.get("average_stars") not in (None, ""):
            try:
                rec["avg_stars"] = float(r["average_stars"])
            except ValueError:
                pass
        items[iid] = rec
    prf = pickle.load(open(os.path.join(root, "itm_prf.pkl"), "rb"))
    for iid, v in prf.items():
        if int(iid) in items:
            items[int(iid)]["profile"] = (v.get("profile") if isinstance(v, dict) else str(v)) or ""
    users = {}
    for line in open(os.path.join(root, f"{ds}_user.jsonl")):
        r = json.loads(line)
        users[int(r["uid"])] = r["interacted_items"]
    feats = np.load(os.path.join(root, "persona", "behavior_features.npz"))
    com = pickle.load(open(communities or os.path.join(DATA_ROOT, ds, "community_assignments.pkl"), "rb"))
    user_comm = np.asarray(com["user_community"]).astype(int)
    return trn, items, users, feats, user_comm


def load_groups(ds, communities=None):
    if ds != "yelp":
        return None
    path = communities or os.path.join(DATA_ROOT, ds, "community_assignments.pkl")
    com = pickle.load(open(path, "rb"))
    home = com.get("home_city")
    if home is None:
        return None
    labels = {c: j for j, c in enumerate(sorted(set(home.tolist())))}
    return np.array([labels[c] for c in home.tolist()], dtype=int)


class Signals:
    def __init__(self, trn, feats, user_comm, baseline_groups=None):
        self.trn = trn
        n_users, n_items = trn.shape
        coo = trn.tocoo()
        order = np.lexsort((coo.col, coo.row))
        rows, cols = coo.row[order], coo.col[order]
        assert np.array_equal(rows, feats["edge_users"]) and np.array_equal(cols, feats["edge_items"]), \
            "behaviour features are not aligned with trn_mat"
        self.edge_index = {}
        for k, (u, i) in enumerate(zip(rows.tolist(), cols.tolist())):
            self.edge_index[(u, i)] = k
        self.feats = {k: feats[k] for k in feats.files if k.startswith("user_") or feats[k].shape[0] == len(rows)}
        self.user_level = {k: feats[k] for k in feats.files if feats[k].shape[0] == n_users and not k.startswith("edge")}
        cnt = np.asarray(trn.sum(0)).ravel()
        self.item_count = cnt
        self.pop_pct = (rankdata(cnt, method="min") - 1) / max(n_items - 1, 1) * 100.0
        n_comm = int(user_comm.max()) + 1
        M = sp.csr_matrix((np.ones(n_users), (user_comm, np.arange(n_users))), shape=(n_comm, n_users))
        self.comm_item = (M @ trn).toarray()
        self.comm_size = np.asarray(M.sum(1)).ravel()
        self.user_comm = user_comm
        self.n_users = n_users
        self.user_group = baseline_groups
        if baseline_groups is not None:
            n_g = int(baseline_groups.max()) + 1
            G = sp.csr_matrix((np.ones(n_users), (baseline_groups, np.arange(n_users))), shape=(n_g, n_users))
            self.group_item = (G @ trn).toarray()
            self.group_size = np.asarray(G.sum(1)).ravel()

    def local_counts_loo(self, u):
        c = self.user_comm[u]
        counts = self.comm_item[c].copy()
        counts[self.trn[u].indices] -= 1
        return counts

    def local_pop_pct(self, u, iids):
        c = self.user_comm[u]
        if self.comm_size[c] - 1 < MIN_COMMUNITY_USERS:
            return {i: None for i in iids}
        counts = self.local_counts_loo(u)
        consumed = np.sort(counts[counts > 0])
        out = {}
        for i in iids:
            k = counts[i]
            out[i] = 0.0 if k <= 0 else float(np.searchsorted(consumed, k, side="left") / max(len(consumed) - 1, 1) * 100.0)
        return out

    def community_profile(self, c, items, global_cat_share):
        counts = self.comm_item[c]
        idx = np.where(counts > 0)[0]
        tot = counts[idx].sum()
        w = collections.Counter()
        for i in idx:
            cats = items.get(int(i), {}).get("cats") or ["(uncategorised)"]
            for cat in cats:
                w[cat] += counts[i] / len(cats)
        shares = sorted(((cat, v / tot) for cat, v in w.items()), key=lambda t: -t[1])
        top_cats = [[cat, round(sh, 2), round(sh / max(global_cat_share.get(cat, 1e-9), 1e-9), 1)] for cat, sh in shares[:5]]
        pop = self.pop_pct[idx]
        wts = counts[idx] / tot
        profile = dict(mean_popularity_percentile=round(float((pop * wts).sum()), 1),
                       share_top10=round(float(((pop >= TIER_TOP) * wts).sum()), 2),
                       share_long_tail_below_median=round(float(((pop < MEDIAN_CUT) * wts).sum()), 2))
        top = idx[np.argsort(-counts[idx])][:5]
        most = [dict(title=truncate(items.get(int(i), {}).get("name") or "(untitled)", 70), community_users=int(counts[i]),
                     global_popularity_percentile=int(round(self.pop_pct[i]))) for i in top]
        return dict(id=int(c), users=int(self.comm_size[c]), share_of_all_users=round(float(self.comm_size[c] / self.n_users), 3),
                    top_categories_vs_all_users=top_cats, popularity_profile=profile, most_consumed_in_community=most)

    def edge(self, name, u, i):
        arr = self.feats.get(name)
        if arr is None:
            return float("nan")
        return float(arr[self.edge_index[(u, i)]])

    def lift(self, u, i):
        c = self.user_comm[u]
        size = self.comm_size[c]
        if size <= 1:
            return float("nan")
        local = (self.comm_item[c, i] - 1) / (size - 1)
        if self.user_group is not None:
            g = self.user_group[u]
            if self.group_size[g] - 1 < MIN_COMMUNITY_USERS:
                return float("nan")
            base = (self.group_item[g, i] - 1) / (self.group_size[g] - 1)
        else:
            base = (self.item_count[i] - 1) / (self.n_users - 1)
        if base <= 0:
            return float("nan")
        return float(local / base)


def tier(p):
    if p >= TIER_TOP:
        return "top 10%"
    if p >= TIER_POPULAR:
        return "popular (top 30%)"
    if p >= TIER_MIDDLE:
        return "middle"
    return "long tail"


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def rnd(x, nd=2):
    return None if x != x else round(float(x), nd)


def category_stats(recs, items, n_cat_dataset):
    w = collections.Counter()
    for r in recs:
        cats = items.get(r["iid"], {}).get("cats") or ["(uncategorised)"]
        for c in cats:
            w[c] += 1.0 / len(cats)
    tot = sum(w.values()) or 1.0
    shares = sorted(((c, v / tot) for c, v in w.items()), key=lambda t: -t[1])
    H = -sum(s * math.log(s) for _, s in shares if s > 0)
    norm = H / math.log(max(n_cat_dataset, 2))
    return shares, norm


def facet_stats(recs, items, sig, user_level, n_cat_dataset):
    u = recs[0]["uid"]
    pops = [sig.pop_pct[r["iid"]] for r in recs]
    lifts = [sig.lift(u, r["iid"]) for r in recs]
    lifts_ok = [x for x in lifts if x == x]
    price = [sig.edge("price_level", u, r["iid"]) for r in recs]
    price_ok = [x for x in price if x == x]
    dev = [sig.edge("rating_deviation", u, r["iid"]) for r in recs]
    adopt = [sig.edge("adoption_percentile", u, r["iid"]) for r in recs]
    rchars = [sig.edge("review_chars", u, r["iid"]) for r in recs]
    shares, breadth = category_stats(recs, items, n_cat_dataset)
    local = sig.local_pop_pct(u, [r["iid"] for r in recs])
    loc = [v for v in local.values() if v is not None]
    st = dict(
        training_interactions=len(recs),
        distinct_categories=len(shares),
        top_categories=[[c, round(s, 2)] for c, s in shares[:4]],
        category_breadth_index=rnd(breadth, 3),
        mean_popularity_percentile=rnd(np.mean(pops), 1),
        share_top10=rnd(np.mean([p >= TIER_TOP for p in pops])),
        share_popular_top30=rnd(np.mean([p >= TIER_POPULAR for p in pops])),
        share_long_tail_below_median=rnd(np.mean([p < MEDIAN_CUT for p in pops])),
        community_concentration_share=rnd(np.mean([x >= 2.0 for x in lifts_ok])) if lifts_ok else None,
        mean_community_lift=rnd(np.mean(lifts_ok), 1) if lifts_ok else None,
        mean_rating=rnd(np.mean([float(r["rating"]) for r in recs if r.get("rating") is not None]), 2),
        rating_offset_vs_item_average=rnd(nanmean(dev), 2),
        adoption_percentile_mean=rnd(nanmean(adopt) * 100, 0),
        median_review_chars=rnd(np.nanmedian(rchars), 0) if any(x == x for x in rchars) else None,
        price_level_mean=rnd(np.mean(price_ok), 2) if price_ok else None,
        price_level_spread=rnd(np.std(price_ok), 2) if price_ok else None,
        price_unknown_share=rnd(1 - len(price_ok) / len(price)),
        mean_local_popularity_percentile=rnd(np.mean(loc), 1) if loc else None,
        share_community_top10=rnd(np.mean([v >= TIER_TOP for v in loc])) if loc else None,
        share_community_top30=rnd(np.mean([v >= TIER_POPULAR for v in loc])) if loc else None,
        local_minus_global_popularity=rnd(np.mean(loc) - np.mean(pops), 1) if loc else None,
    )
    if "user_lifetime_reviews" in user_level:
        st["lifetime_reviews_on_platform"] = int(user_level["user_lifetime_reviews"][u])
    if "home_city" in sig.feats:
        st["home_city_share"] = rnd(nanmean([sig.edge("home_city", u, r["iid"]) for r in recs]))
        st["friend_reviewed_before_share"] = rnd(nanmean([sig.edge("friend_before", u, r["iid"]) for r in recs]))
    if "user_elite_years" in user_level:
        st["elite_years"] = int(user_level["user_elite_years"][u])
        st["fans"] = int(user_level["user_fans"][u])
    return st


def rule_labels(st, ref):
    T = ref["cutpoints"]
    lab = {}
    if (st["share_popular_top30"] or 0) >= 0.35 and (st["share_long_tail_below_median"] or 0) >= 0.35:
        lab["conformity"] = "mixed"
    elif st["mean_popularity_percentile"] >= T["mean_popularity_percentile"][1]:
        lab["conformity"] = "mainstream"
    elif st["mean_popularity_percentile"] <= T["mean_popularity_percentile"][0]:
        lab["conformity"] = "niche"
    else:
        lab["conformity"] = "mixed"
    cc = st.get("community_concentration_share")
    if cc is None:
        lab["social_scope"] = "mixed"
    elif cc >= T["community_concentration_share"][1]:
        lab["social_scope"] = "community_driven"
    elif cc <= T["community_concentration_share"][0]:
        lab["social_scope"] = "broad"
    else:
        lab["social_scope"] = "mixed"
    if st["training_interactions"] < 4:
        lab["exploration"] = "unclear"
    elif st["category_breadth_index"] >= T["category_breadth_index"][1]:
        lab["exploration"] = "eclectic"
    elif st["category_breadth_index"] <= T["category_breadth_index"][0]:
        lab["exploration"] = "focused"
    else:
        lab["exploration"] = "moderate"
    a = st["adoption_percentile_mean"]
    lab["adoption"] = "typical" if a is None else ("early" if a <= T["adoption_percentile_mean"][0] else "late" if a >= T["adoption_percentile_mean"][1] else "typical")
    d = st["rating_offset_vs_item_average"]
    lab["rating_stance"] = "balanced" if d is None else ("generous" if d >= T["rating_offset_vs_item_average"][1] else "critical" if d <= T["rating_offset_vs_item_average"][0] else "balanced")
    if st["price_unknown_share"] is None or st["price_unknown_share"] > 0.5 or st["price_level_mean"] is None:
        lab["price"] = "unknown"
    elif st["price_level_spread"] >= 1.0:
        lab["price"] = "mixed"
    elif st["price_level_mean"] <= 1.75:
        lab["price"] = "budget"
    elif st["price_level_mean"] >= 2.75:
        lab["price"] = "premium"
    else:
        lab["price"] = "mid"
    m = st["median_review_chars"]
    lab["review_style"] = "standard" if m is None else ("terse" if m <= T["median_review_chars"][0] else "detailed" if m >= T["median_review_chars"][1] else "standard")
    return lab


def truncate(s, n):
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def item_record(r, items, sig, u, local_pct=None):
    it = items.get(r["iid"], {})
    p = sig.pop_pct[r["iid"]]
    rec = dict(
        id=r["iid"],
        title=truncate(it.get("name") or r.get("title") or "(untitled)", TITLE_CHARS),
        categories=", ".join(it.get("cats") or []) or "(none)",
        popularity_percentile=int(round(p)),
        popularity_tier=tier(p),
        user_rating=r.get("rating"),
    )
    if "city" in it:
        rec["city"] = it["city"]
    if "avg_stars" in it:
        rec["item_average_stars"] = round(it["avg_stars"], 1)
    dev = sig.edge("rating_deviation", u, r["iid"])
    if dev == dev:
        rec["rating_vs_item_average"] = round(dev, 1)
    pr = sig.edge("price_level", u, r["iid"])
    if pr == pr:
        rec["price_level_1to4"] = int(pr)
    ad = sig.edge("adoption_percentile", u, r["iid"])
    if ad == ad:
        rec["reviewed_before_pct_of_item_reviewers"] = int(round((1 - ad) * 100))
    lf = sig.lift(u, r["iid"])
    if lf == lf:
        rec["community_lift"] = round(lf, 1)
    if local_pct and local_pct.get(r["iid"]) is not None:
        rec["local_popularity_percentile"] = int(round(local_pct[r["iid"]]))
    if r.get("review"):
        rec["review_excerpt"] = truncate(r["review"], REVIEW_CHARS)
    if it.get("profile"):
        rec["about"] = truncate(it["profile"], PROFILE_CHARS)
    return rec


def sample_items(recs, sig, k, rng):
    if len(recs) <= k:
        return sorted(recs, key=lambda r: -sig.pop_pct[r["iid"]])
    buckets = collections.defaultdict(list)
    for r in recs:
        buckets[tier(sig.pop_pct[r["iid"]])].append(r)
    order = ["top 10%", "popular (top 30%)", "middle", "long tail"]
    for t in order:
        rng.shuffle(buckets[t])
        buckets[t].sort(key=lambda r: -len(r.get("review") or ""))
    chosen, seen_cats = [], collections.Counter()
    while len(chosen) < k and any(buckets[t] for t in order):
        for t in order:
            if len(chosen) >= k or not buckets[t]:
                continue
            best = min(range(len(buckets[t])), key=lambda j: (seen_cats[buckets[t][j].get("_cat")], j))
            r = buckets[t].pop(best)
            chosen.append(r)
            seen_cats[r.get("_cat")] += 1
    return sorted(chosen, key=lambda r: -sig.pop_pct[r["iid"]])


def build(ds, uids, out_dir, rng, raw_root, reference_only=False, communities=None, suffix=""):
    trn, items, users, feats, user_comm = load_raw(ds, raw_root, communities)
    sig = Signals(trn, feats, user_comm, baseline_groups=load_groups(ds, communities))
    n_cat_dataset = len({c for it in items.values() for c in it["cats"]})
    user_level = sig.user_level
    all_stats = {}
    for u, recs0 in users.items():
        recs = [dict(uid=u, iid=int(r["iid"]), rating=r.get("rating"), review=r.get("review"), title=r.get("title")) for r in recs0]
        if not recs:
            continue
        all_stats[u] = facet_stats(recs, items, sig, user_level, n_cat_dataset)
    keys = ["mean_popularity_percentile", "community_concentration_share", "category_breadth_index",
            "adoption_percentile_mean", "rating_offset_vs_item_average", "median_review_chars"]
    extra = ["training_interactions", "distinct_categories", "share_top10", "share_popular_top30",
             "share_long_tail_below_median", "mean_community_lift", "mean_rating", "price_level_mean",
             "lifetime_reviews_on_platform", "home_city_share", "friend_reviewed_before_share", "elite_years", "fans",
             "mean_local_popularity_percentile", "share_community_top10", "share_community_top30", "local_minus_global_popularity"]
    quantiles = {}
    for k in keys + extra:
        vals = np.array([s[k] for s in all_stats.values() if s.get(k) is not None], dtype=float)
        if len(vals) == 0:
            continue
        quantiles[k] = [round(float(np.percentile(vals, q)), 3) for q in (100 / 3, 50, 200 / 3)]
    cut = {}
    for k in keys:
        vals = np.array([s[k] for s in all_stats.values() if s.get(k) is not None], dtype=float)
        if k == "category_breadth_index":
            vals = np.array([s[k] for s in all_stats.values() if s.get(k) is not None and s["training_interactions"] >= 4], dtype=float)
        cut[k] = [round(float(np.percentile(vals, 100 / 3)), 3), round(float(np.percentile(vals, 200 / 3)), 3)]
    ref = dict(dataset=ds, users=len(all_stats), items=trn.shape[1], distinct_categories=n_cat_dataset,
               cutpoints=cut, quantiles=quantiles,
               medians={k: round(float(np.nanmedian([s[k] for s in all_stats.values() if s.get(k) is not None])), 3) for k in keys})
    os.makedirs(out_dir, exist_ok=True)
    ref["communities"] = communities or "community_assignments.pkl"
    ref["lift_baseline"] = "home city" if sig.user_group is not None else "all users"
    json.dump(ref, open(os.path.join(out_dir, f"reference{suffix}.json"), "w"), indent=1)
    if reference_only:
        print(f"{ds}: reference.json rewritten with {len(quantiles)} quantile keys")
        return

    gw = collections.Counter()
    for i, cnt in enumerate(sig.item_count):
        if cnt <= 0:
            continue
        cats = items.get(i, {}).get("cats") or ["(uncategorised)"]
        for cat in cats:
            gw[cat] += cnt / len(cats)
    gtot = sum(gw.values()) or 1.0
    global_cat_share = {cat: v / gtot for cat, v in gw.items()}
    comm_cache = {}
    n_rec, n_split = 0, 0
    with open(os.path.join(out_dir, f"evidence{suffix}.jsonl"), "w") as f:
        for u in uids:
            recs0 = users.get(u) or []
            recs = []
            for r in recs0:
                iid = int(r["iid"])
                cats = items.get(iid, {}).get("cats") or []
                recs.append(dict(uid=u, iid=iid, rating=r.get("rating"), review=r.get("review"), title=r.get("title"),
                                 _cat=cats[0] if cats else None))
            if not recs:
                continue
            overall = all_stats[u]
            local_pct = sig.local_pop_pct(u, [r["iid"] for r in recs])
            c = int(sig.user_comm[u])
            if c not in comm_cache:
                comm_cache[c] = sig.community_profile(c, items, global_cat_share)
            pops = np.array([sig.pop_pct[r["iid"]] for r in recs])
            main = [r for r, p in zip(recs, pops) if p >= TIER_POPULAR]
            tail = [r for r, p in zip(recs, pops) if p < MEDIAN_CUT]
            facets = [("overall", recs, None)]
            if len(main) >= MIN_MODE_ITEMS and len(tail) >= MIN_MODE_ITEMS:
                facets += [("mainstream_mode", main, tail), ("longtail_mode", tail, main)]
                n_split += 1
            for name, frecs, other in facets:
                st = overall if name == "overall" else facet_stats(frecs, items, sig, user_level, n_cat_dataset)
                k = OVERALL_ITEMS if name == "overall" else MODE_ITEMS
                shown = [item_record(r, items, sig, u, local_pct) for r in sample_items(frecs, sig, k, rng)]
                rec = dict(record_id=f"u{u}:{name}", uid=u, facet=name, dataset=ds,
                           facet_definition={
                               "overall": "the user's whole training history",
                               "mainstream_mode": f"only the user's items in the catalog's top 30% by popularity ({len(frecs)} of {len(recs)} interactions)",
                               "longtail_mode": f"only the user's items below the catalog's median popularity ({len(frecs)} of {len(recs)} interactions)",
                           }[name],
                           stats=st, items=shown, community=comm_cache[c])
                if other is not None:
                    osh, _ = category_stats(other, items, n_cat_dataset)
                    rec["other_mode_brief"] = dict(
                        mode="longtail_mode" if name == "mainstream_mode" else "mainstream_mode",
                        interactions=len(other),
                        top_categories=[[c, round(s, 2)] for c, s in osh[:3]],
                        example_titles=[truncate(items.get(r["iid"], {}).get("name") or r.get("title") or "", 60) for r in other[:3]],
                    )
                rec["rule_labels"] = rule_labels(st, ref)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1
    print(f"{ds}: {len(uids)} users -> {n_rec} evidence records ({n_split} users split into modes); "
          f"cutpoints {json.dumps(cut)}")


MAX_PERSONAS = 5
ABOUT_TASTE = "what they pick"
ABOUT_OTHER = "how or where they pick"
ROLE_NOUNS = ["enthusiast", "seeker", "lover", "fan", "fans", "explorer", "aficionado", "connoisseur",
              "buff", "devotee", "junkie", "addict"]

STAT_KEYS = ["training_interactions", "distinct_categories", "top_categories", "category_breadth_index",
             "mean_popularity_percentile", "share_top10", "share_popular_top30", "share_long_tail_below_median",
             "community_concentration_share", "mean_community_lift",
             "mean_local_popularity_percentile", "share_community_top10", "share_community_top30", "local_minus_global_popularity",
             "price_level_mean", "price_level_spread", "price_unknown_share",
             "home_city_share", "friend_reviewed_before_share"]
ITEM_KEYS = ["id", "title", "categories", "city", "popularity_percentile", "popularity_tier", "local_popularity_percentile",
             "community_lift", "price_level_1to4", "user_rating", "review_excerpt", "about"]
REF_KEYS = [k for k in STAT_KEYS if k not in ("top_categories", "price_level_spread", "price_unknown_share")]

REFERENCE = {
    "amazon_book": {
        "training_interactions": [6.0, 7.0, 9.0],
        "distinct_categories": [6.0, 7.0, 9.0],
        "category_breadth_index": [0.386, 0.427, 0.465],
        "mean_popularity_percentile": [60.2, 67.5, 75.1],
        "share_top10": [0.17, 0.27, 0.4],
        "share_popular_top30": [0.43, 0.56, 0.67],
        "share_long_tail_below_median": [0.14, 0.25, 0.33],
        "community_concentration_share": [0.6, 0.75, 0.89],
        "mean_community_lift": [3.8, 4.6, 6.9],
        "mean_local_popularity_percentile": [53.3, 66.2, 76.8],
        "share_community_top10": [0.22, 0.33, 0.44],
        "share_community_top30": [0.48, 0.6, 0.74],
        "local_minus_global_popularity": [-10.7, -1.1, 4.6],
        "price_level_mean": [2.0, 2.33, 2.75],
    },
    "amazon_movie": {
        "training_interactions": [4.0, 4.0, 6.0],
        "distinct_categories": [6.0, 7.0, 8.0],
        "category_breadth_index": [0.415, 0.457, 0.498],
        "mean_popularity_percentile": [64.2, 72.1, 79.6],
        "share_top10": [0.2, 0.33, 0.5],
        "share_popular_top30": [0.5, 0.67, 0.75],
        "share_long_tail_below_median": [0.0, 0.18, 0.29],
        "community_concentration_share": [0.35, 0.5, 0.6],
        "mean_community_lift": [3.3, 4.9, 7.9],
        "mean_local_popularity_percentile": [21.2, 33.2, 50.6],
        "share_community_top10": [0.01, 0.23, 0.33],
        "share_community_top30": [0.25, 0.33, 0.5],
        "local_minus_global_popularity": [-43.8, -31.3, -18.8],
        "price_level_mean": [2.33, 2.67, 3.0],
    },
    "yelp": {
        "training_interactions": [8.0, 10.0, 13.0],
        "distinct_categories": [27.0, 32.0, 40.0],
        "category_breadth_index": [0.452, 0.477, 0.5],
        "mean_popularity_percentile": [63.5, 68.9, 74.2],
        "share_top10": [0.22, 0.31, 0.4],
        "share_popular_top30": [0.5, 0.6, 0.67],
        "share_long_tail_below_median": [0.2, 0.26, 0.33],
        "community_concentration_share": [0.16, 0.25, 0.35],
        "mean_community_lift": [1.3, 1.5, 1.8],
        "mean_local_popularity_percentile": [49.1, 58.8, 66.8],
        "share_community_top10": [0.21, 0.28, 0.35],
        "share_community_top30": [0.41, 0.5, 0.6],
        "local_minus_global_popularity": [-19.9, -9.4, -0.9],
        "price_level_mean": [1.85, 1.94, 2.0],
        "home_city_share": [0.6, 0.75, 0.89],
        "friend_reviewed_before_share": [0.0, 0.0, 0.0],
    },
}

DEFINITIONS = """STATISTIC DEFINITIONS (all computed on this user's training interactions only)
- popularity percentile of an item: share of catalog items with fewer interactions; 100 = the most consumed item. Tiers: top 10% (>=90), popular top 30% (>=70), middle (40-70), long tail (<40).
- mean_popularity_percentile, share_top10, share_popular_top30, share_long_tail_below_median: over the user's items.
- community_lift of an item: how much more often the user's community consumes it than {lift_base}, leave-one-out; community_concentration_share = share of the user's items with lift >= 2.
- category_breadth_index: normalised entropy of the categories of the user's items (0 = one category, 1 = spread over every category).
- price_level: 1 (cheapest quartile) to 4 (most expensive); price_level_spread = standard deviation.
- user_rating on an item: the stars the user gave it (evidence of what they liked or disliked)."""

LOCAL_DEFS = """- community: {community_def}. The request carries a summary of the user's community: size, its top categories with lift versus all users, its popularity profile, and the items it consumes most.
- local_popularity_percentile of an item: rank of the item by consumption inside the user's community (the user excluded); 100 = the item the community consumes most. mean_local_popularity_percentile, share_community_top10 / share_community_top30: over the user's items. local_minus_global_popularity = mean local percentile minus mean global percentile: positive means their picks rank higher inside their community than in the whole catalog, negative the opposite."""

COMMUNITY_DEF = {
    "amazon_book": "users clustered by similar training histories (co-consumption clusters)",
    "amazon_movie": "users clustered by similar training histories (co-consumption clusters)",
    "yelp": "users of the same home city who consume similar businesses (co-consumption clusters within a city)",
}
LIFT_BASE = {"amazon_book": "the whole user base", "amazon_movie": "the whole user base", "yelp": "users of the same home city"}
YELP_DEFS = """- home_city_share: share of the user's items located in their home city; friend_reviewed_before_share: share of their items that a friend had reviewed before them."""


def reference_block(ds):
    q = REFERENCE[ds]
    lines = ["REFERENCE DISTRIBUTION (all users of this dataset; lower tertile / median / upper tertile)"]
    for k in REF_KEYS:
        if k in q:
            a, b, c = q[k]
            lines.append(f"- {k}: {a:g} / {b:g} / {c:g}")
    return "\n".join(lines)


def system_prompt(ds):
    d = DATASETS[ds]
    return f"""You write personas for a top-N recommender trained on {d['domain']}. An interaction means {d['interaction']}. Each request holds the evidence for ONE user: statistics over their whole training history, a summary of the community they belong to, and a sample of their items. Return one JSON object matching the required schema and nothing else.

WHAT A PERSONA IS HERE
A persona is one trait that characterises this {d['person_word']}: one thing a recommender should know about them. A user gets 1 to {MAX_PERSONAS} personas, each about a DIFFERENT aspect. You decide how many there are and which aspects they cover. Nothing in the evidence has to become a persona and no kind of trait is required, with one exception: at least one persona is about WHAT they pick (about = "{ABOUT_TASTE}"), because that is what a recommender needs first; a user with two unrelated tastes may get two. Any other persona (about = "{ABOUT_OTHER}") covers how, where or with whom they pick, or anything else that stands out in the evidence.

"{ABOUT_TASTE}" means the CONTENT of their picks: genres, themes, settings, authors, series, cuisines, dishes, venue types, places. How popular, how community-favoured, how priced or how varied their picks are is NOT content; such a persona is "{ABOUT_OTHER}".

Personas are embedded and clustered across all users, and users whose traits are alike should land together. So each persona must be self-contained, about one aspect only, and written so that two users with the same trait would receive similar text.

HOW TO DECIDE WHAT IS NOTABLE
- Judge numbers against the REFERENCE DISTRIBUTION below: a statistic outside the tertile boundaries is a candidate trait; one near the median is not. Judge taste from the items, their categories, notes, ratings and reviews; the community summary tells you what is ordinary within their community.
- Write everything in absolute terms: quote the user's own numbers, titles, places. Never write comparisons such as "unlike typical users", "more than most", "than average", "mainstream" or "niche" without the number that backs them.
- Keep taste words and behaviour words in separate personas. Do not split one taste into near-duplicates; merge neighbouring sub-genres unless the evidence really separates them.

FIELDS
- persona_name: 3-6 words, a noun phrase naming the trait itself, not the person, and stating its direction ("Long-tail picks within their community", not "Popularity"; "Budget-priced picks", not "Price level"). No role nouns ({', '.join(ROLE_NOUNS[:8])}).
- about: "{ABOUT_TASTE}" (content) or "{ABOUT_OTHER}" (everything else).
- aspect: 1-3 plain words for the kind of trait, in your own words; use the same wording for the same kind of trait.
- summary: 1-2 sentences, third person, "they". Concrete nouns for taste, numbers for behaviour. Vary how sentences open.
- strength: 3 = you would bet on it: a statistic clearly beyond a tertile boundary, or a taste repeated across several items; 2 = supported but moderate; 1 = suggested but thin.
- evidence_item_ids: 0-6 ids from the request's items that support the persona (0 only when it rests on statistics alone).
- not_notable: 0-5 short phrases for aspects you considered and found ordinary for this user.

GENERAL
- Use only the evidence; never invent titles, places, people or facts. Plain English, no markdown.

{DEFINITIONS.format(lift_base=LIFT_BASE[ds])}
{LOCAL_DEFS.format(community_def=COMMUNITY_DEF[ds])}{chr(10) + YELP_DEFS if ds == 'yelp' else ''}

{reference_block(ds)}"""


def user_message(ev):
    stats = {k: ev["stats"][k] for k in STAT_KEYS if ev["stats"].get(k) is not None}
    items = [{k: it[k] for k in ITEM_KEYS if k in it} for it in ev["items"]]
    body = {"user_stats": stats, "community": ev.get("community"), "items": items}
    return json.dumps(body, ensure_ascii=False, separators=(",", ": "))


SCHEMA = {
    "type": "object",
    "properties": {
        "personas": {
            "type": "array",
            "description": "1-5 personas, each about a DIFFERENT aspect of the user",
            "items": {
                "type": "object",
                "properties": {
                    "persona_name": {"type": "string", "description": "3-6 words naming the trait itself"},
                    "aspect": {"type": "string", "description": "1-3 plain words: what kind of trait this is, in your own words"},
                    "summary": {"type": "string", "description": "1-2 sentences, absolute wording, numbers or titles as evidence"},
                    "strength": {"type": "integer", "enum": [1, 2, 3]},
                    "evidence_item_ids": {"type": "array", "items": {"type": "integer"}},
                    "about": {"type": "string", "enum": [ABOUT_TASTE, ABOUT_OTHER]},
                },
                "required": ["persona_name", "about", "aspect", "summary", "strength", "evidence_item_ids"],
                "additionalProperties": False,
            },
        },
        "not_notable": {
            "type": "array",
            "items": {"type": "string"},
            "description": "0-5 short phrases: aspects considered and found ordinary for this user",
        },
    },
    "required": ["personas", "not_notable"],
    "additionalProperties": False,
}

POPULARITY_WORDS = ["mainstream", "niche", "popular", "popularity", "obscure", "hidden gem", "cult", "trend", "trending",
                    "bestseller", "best-seller", "well-known", "lesser-known", "less-known", "widely", "broad appeal",
                    "concentrated", "under-the-radar", "blockbuster"]
COMPARISON_WORDS = ["typical users", "most users", "most readers", "most viewers", "most diners", "unlike", "than average", "than others"]
BEHAVIOUR_RX = re.compile(r"popular|community|local|price|spending|breadth|spread|variety|range|rating|review|timing|adoption|platform|activity")


def banned_hits(text, words):
    t = text.lower()
    return [w for w in words if re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", t)]


def is_taste(p):
    return p.get("about") == ABOUT_TASTE and not BEHAVIOUR_RX.search((p.get("aspect") or "").lower())


def validate(obj, ev):
    errs = []
    ps = obj.get("personas") if isinstance(obj, dict) else None
    if not isinstance(ps, list):
        return ["no personas list"]
    if not (1 <= len(ps) <= MAX_PERSONAS):
        errs.append("persona count")
    ids = {it["id"] for it in ev["items"]}
    names = [(p.get("persona_name") or "").lower() for p in ps]
    aspects = [(p.get("aspect") or "").strip().lower() for p in ps]
    if len(set(names)) < len(names):
        errs.append("duplicate persona_name")
    if len(set(aspects)) < len(aspects):
        errs.append("repeated aspect")
    for p, name in zip(ps, names):
        if p.get("about") not in (ABOUT_TASTE, ABOUT_OTHER):
            errs.append("about value")
        if p.get("strength") not in (1, 2, 3):
            errs.append("strength value")
        if not set(p.get("evidence_item_ids") or []) <= ids:
            errs.append("evidence ids not in prompt")
        if any(re.search(rf"\b{n}\b", name) for n in ROLE_NOUNS):
            errs.append("role noun in persona_name")
        s = p.get("summary") or ""
        if banned_hits(s + " " + name, COMPARISON_WORDS):
            errs.append("comparison wording")
        if banned_hits(s + " " + name, POPULARITY_WORDS) and not re.search(r"\d", s):
            errs.append("popularity word without a number")
        if p.get("about") == ABOUT_TASTE and BEHAVIOUR_RX.search((p.get("aspect") or "").lower()):
            errs.append("about mislabel")
    if not any(is_taste(p) for p in ps):
        errs.append("no content taste persona")
    return sorted(set(errs))


def select_personas(personas, min_strength=3):
    ps = [p for p in personas if isinstance(p.get("strength"), int)]
    if not ps:
        return []
    keep = [p for p in ps if p["strength"] >= min_strength]
    if not keep:
        top = max(p["strength"] for p in ps)
        keep = [p for p in ps if p["strength"] == top]
    if not any(is_taste(p) for p in keep):
        tastes = [p for p in ps if is_taste(p)]
        if tastes:
            keep.append(max(tastes, key=lambda p: p["strength"]))
    return keep


def persona_text(p):
    return " ".join(x.strip() for x in (p.get("persona_name", ""), p.get("summary", "")) if x and x.strip())


def parse_json(text):
    t = text.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise
        return json.loads(t[i:j + 1])


def make_row(e, cid, result_type, stop, text):
    row = {"custom_id": cid, "record_id": f"u{e['uid']}", "uid": e["uid"], "facet": None,
           "result_type": result_type, "stop_reason": stop, "persona": None, "validation_errors": [], "valid": False,
           "model": MODEL}
    if result_type != "succeeded":
        return row
    try:
        obj = parse_json(text)
        row["persona"] = obj
        row["validation_errors"] = validate(obj, e)
        row["valid"] = not row["validation_errors"]
    except (ValueError, TypeError, AttributeError) as ex:
        row["result_type"] = "errored"
        row["error"] = f"{type(ex).__name__}: {ex}"[:200]
    return row


def request_params(system, e):
    return dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user_message(e)}],
    )


def run_batches(system, todo, lookup, ds, out_dir, out_path, raw_path, poll):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic(api_key=API_KEY or None)
    state_path = os.path.join(out_dir, "batches.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            batch_ids = json.load(f)
        print(f"resuming the submitted batches {batch_ids}", flush=True)
    else:
        batch_ids = []
        for start in range(0, len(todo), BATCH_LIMIT):
            requests = [Request(custom_id=f"{ds}__u{e['uid']}", params=MessageCreateParamsNonStreaming(**request_params(system, e)))
                        for e in todo[start:start + BATCH_LIMIT]]
            try:
                batch = client.messages.batches.create(requests=requests)
            except anthropic.AuthenticationError:
                sys.exit("Anthropic credentials are missing or invalid: set API_KEY at the top of this file.")
            except anthropic.RateLimitError as ex:
                sys.exit(f"Rate limited while creating the batch; run the same command again later. ({ex.message})")
            except anthropic.APIStatusError as ex:
                sys.exit(f"The batch was not created ({ex.status_code}): {ex.message}")
            except anthropic.APIConnectionError as ex:
                sys.exit(f"Could not reach the API: {ex}")
            batch_ids.append(batch.id)
            with open(state_path, "w") as f:
                json.dump(batch_ids, f)
            print(f"submitted batch {batch.id} ({len(requests)} users)", flush=True)

    rows = []
    for batch_id in batch_ids:
        while True:
            try:
                batch = client.messages.batches.retrieve(batch_id)
            except anthropic.APIConnectionError:
                time.sleep(poll)
                continue
            if batch.processing_status == "ended":
                break
            c = batch.request_counts
            print(f"  {batch_id}: {c.processing} processing, {c.succeeded} succeeded, {c.errored} errored", flush=True)
            time.sleep(poll)
        with open(out_path, "a") as fo, open(raw_path, "a") as fr:
            for result in client.messages.batches.results(batch_id):
                e = lookup.get(result.custom_id)
                if e is None:
                    continue
                text, stop = "", None
                if result.result.type == "succeeded":
                    message = result.result.message
                    text = next((b.text for b in message.content if b.type == "text"), "")
                    stop = message.stop_reason
                fr.write(json.dumps({"custom_id": result.custom_id, "uid": e["uid"], "text": text}, ensure_ascii=False) + "\n")
                row = make_row(e, result.custom_id, result.result.type, stop, text)
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
    os.remove(state_path)
    return rows


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def evidence_path(out_root, ds, suffix=""):
    if not suffix:
        v = os.path.join(out_root, ds, "evidence_v22.jsonl")
        if os.path.exists(v):
            return v
    return os.path.join(out_root, ds, f"evidence{suffix}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--raw-root", default=os.environ.get("PERSONA_RAW_ROOT", DATA_ROOT))
    ap.add_argument("--out-root", default=os.environ.get("PERSONA_OUT_ROOT", os.path.join(HERE, "out_full")))
    ap.add_argument("--users", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--communities", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--reference-only", action="store_true")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--tag", default="v22")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--uids", default=None)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ds = a.dataset

    ev_path = a.evidence or evidence_path(a.out_root, ds, a.suffix)
    build_only = a.build_only or a.reference_only
    if build_only or not os.path.exists(ev_path):
        if a.evidence and not build_only:
            sys.exit(f"{a.evidence} does not exist")
        rng = random.Random(a.seed)
        uids = [int(json.loads(l)["uid"]) for l in open(os.path.join(a.raw_root, ds, f"{ds}_user.jsonl"))]
        if a.users:
            uids = sorted(rng.sample(uids, min(a.users, len(uids))))
        build(ds, uids, os.path.join(a.out_root, ds), rng, a.raw_root, reference_only=a.reference_only,
              communities=a.communities, suffix=a.suffix)
        if build_only:
            return

    all_records = [e for e in load_jsonl(ev_path) if e.get("facet", "overall") == "overall"]
    records = all_records
    if a.uids:
        want = {int(x) for x in a.uids.split(",")}
        records = [e for e in records if e["uid"] in want]
    if a.limit:
        records = records[:a.limit]
    system = system_prompt(ds)

    if a.dry_run:
        print(system)
        print("\n" + "=" * 100 + "\nUSER MESSAGE (first record, uid %s):\n" % (records[0]["uid"] if records else "-"))
        if records:
            print(user_message(records[0]))
        print("\n" + "=" * 100 + "\nOUTPUT SCHEMA:\n" + json.dumps(SCHEMA, ensure_ascii=False, indent=1))
        return

    out_dir = a.out_dir or os.path.join(a.out_root, ds, a.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path, raw_path = os.path.join(out_dir, "personas.jsonl"), os.path.join(out_dir, "results_raw.jsonl")

    done = load_jsonl(out_path) if os.path.exists(out_path) else []
    if a.retry_errors:
        done = [r for r in done if r.get("persona")]
        with open(out_path, "w") as f:
            for r in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    skip = {r["uid"] for r in done}
    todo = [e for e in records if e["uid"] not in skip]
    pending = os.path.exists(os.path.join(out_dir, "batches.json"))
    print(f"{ds}: evidence {ev_path} ({len(records)} users) | done {len(skip)} | to run {len(todo)} | "
          f"model {MODEL} | out {out_dir}", flush=True)
    if todo or pending:
        lookup = {f"{ds}__u{e['uid']}": e for e in all_records}
        t0 = time.time()
        rows = run_batches(system, todo, lookup, ds, out_dir, out_path, raw_path, a.poll)
        done += rows
        n_ok = sum(r["persona"] is not None for r in rows)
        n_valid = sum(r["valid"] for r in rows)
        dt = time.time() - t0
        print(f"{ds}: {len(rows)} users in {dt/60:.1f} min | parsed {n_ok} | passed v2.2 checks {n_valid}", flush=True)
    write_texts(out_dir, done)


def write_texts(out_dir, rows):
    path = os.path.join(out_dir, "persona_texts.jsonl")
    n_users = n_all = n_kept = 0
    with open(path, "w") as f:
        for r in sorted(rows, key=lambda r: r["uid"]):
            obj = r.get("persona")
            if not obj or not isinstance(obj.get("personas"), list):
                continue
            n_users += 1
            n_all += len(obj["personas"])
            for j, p in enumerate(select_personas(obj["personas"])):
                n_kept += 1
                f.write(json.dumps({"record_id": f"u{r['uid']}:p{j}", "uid": r["uid"], "persona_index": j, "persona": p,
                                    "text": persona_text(p), "valid": r.get("valid")}, ensure_ascii=False) + "\n")
    print(f"{path}: {n_users} users, {n_all} personas -> {n_kept} kept ({n_kept/max(n_users,1):.2f}/user)", flush=True)


if __name__ == "__main__":
    main()
