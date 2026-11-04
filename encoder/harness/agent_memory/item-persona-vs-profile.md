---
name: item-persona-vs-profile
description: "Item-side text for PERSONA-MOSAIC - the existing one-per-item LLM item persona (user format) vs the RLMRec profile used today; offline comparison of 2026-10-28 and the user's proposal to use item personas"
metadata:
  node_type: memory
  type: project
  originSessionId: f7f1597f-a902-4037-94c5-2c3c48376061
  modified: 2026-10-29T13:34:22.649Z
---

On 2026-10-28 the user proposed making item personas "like the user ones, but one per item since items
have no facets". That data already exists under the symlinked data dirs:
- /home/kyung/PersonaMosaic/data/<ds>/persona/item_persona_pool.jsonl: one persona per item, SAME JSON
  schema as user personas (persona_name, summary, core_tastes, liked/disliked_aspects, behavior_traits,
  behavior_style, distinguishing_note), same LLM (Qwen3-4B-Instruct-2507), written as the persona of the
  item's typical consumer; evidence = description, RLMRec profile, <=3 reviews, category lifts, co-consumer
  categories and "N of this item's training interactions" statements (training split only).
  item_persona_emb.pkl = mean of the 4 views (identity/tastes/aspects/behavior); per-view pickles exist.
- user "facet" = the sub-history (category profile + items) each of the <=3 user personas is written from
  (persona_evidence.jsonl 'facet'); items have a single evidence record.
- persona_emb_one.pkl (created 2026-10-28, users only) = single view 'raw_response' embedding of the same
  28146/31801/15046 records; runs on 10-28 20:52 (Book) / 21:06 (Yelp) used it (cache prefix 'one_',
  Book K=2, Yelp K=16; produced by a code version with a persona-file knob that the 22:28
  persona_mosaic.py no longer has). No one-view item counterpart exists yet.

Offline comparison (scratch script, default 4-view clustering cache, user-mean centred item persona IP vs
own-mean centred RLMRec profile PR = current code): IP lives in the persona space (cos to user mean dir
0.86-0.89 vs PR 0.42-0.47), cluster-similarity margin 1.4-1.7x larger, cosine to the audience's mean
persona 0.33-0.50 vs 0.22-0.35 (shuffled ~0); agreement with the audience's dominant cluster: Book 0.849
vs 0.814 (majority 0.663), Yelp 0.348 vs 0.350, Movie 0.545 vs 0.546; user-retrieval AUC a tie
(0.884/0.740/0.885 vs 0.896/0.746/0.884). Conclusion given to the user: same information content, better
consistency with the expert/cluster space; expect a cleaner method, not a big accuracy jump.

**Why:** the user is deciding whether the item side should carry a real persona instead of the RLMRec stand-in.
Implemented 2026-10-28 (_load_item_personas in persona_mosaic.py); later turned into the `item_text` knob
(profile | persona). As of 2026-10-29 the yml and code default are `profile`, and all 10-29 Book runs used profile.
**How to apply:** profile-based results from 10-27/10-28 are the comparison baseline; update the Section 4 sentence that calls the RLMRec profile the item persona
([[persona-mosaic-paper-notation]], [[persona-mosaic-design]]).
