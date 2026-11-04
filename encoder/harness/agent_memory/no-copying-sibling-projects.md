---
name: no-copying-sibling-projects
description: User wants new models implemented from scratch per their instructions, not by referencing sibling project folders such as /home/kyung/PersonaMosaic
metadata:
  type: feedback
---

When asked to implement a model in /home/kyung/mosaic_v2, do not read or copy the implementation in
other project folders (e.g. /home/kyung/PersonaMosaic, /home/kyung/CoMoE, /home/kyung/MOSAIC); build it
from the user's instructions, the paper, and the issue. Data files symlinked into mosaic_v2/data may be
inspected as data.

**Why:** the user said so explicitly on 2026-10-26 ("다른 폴더의 persona mosaic 를 참고하지말고 ... 처음부터
다시 구현해"); they want an independent implementation.
**How to apply:** inspect only this repo's code plus symlinked data; state assumptions instead of
borrowing design details from sibling folders. Related: [[persona-mosaic-design]].
