---
name: clustering-not-segment-terminology
description: User forbids the word "segment" for the persona k-means groups; say clustering / cluster (in prose, code, config, Korean messages)
metadata:
  type: feedback
---

Never call the persona k-means groups "segments" (Korean: 세그먼트). Use "clustering" for the procedure
and "cluster" for a group, e.g. config keys `clustering_k_min`, buffers `cluster_centroids`,
cache files `persona_clusters_<spec>.pkl`, Korean 클러스터/클러스터링.

**Why:** the user said on 2026-10-26 "앞으로 segment 라는 단어는 쓰지말고 clustering이라고 해 (k-means니까)";
the groups come from k-means, so "cluster" is the accurate term. The whole persona_mosaic code base was
renamed accordingly the same day (no `segment` left in persona_mosaic.py / persona_mosaic.yml).
**How to apply:** keep new code, config keys, docstrings and replies in cluster/clustering wording;
if an old `segment_*` key or file name shows up, treat it as stale. Related: [[persona-mosaic-design]].
