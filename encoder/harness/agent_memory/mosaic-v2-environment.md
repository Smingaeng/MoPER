---
name: mosaic-v2-environment
description: How to run and test code in /home/kyung/mosaic_v2 (conda env, GPUs, missing tools)
metadata:
  type: user
---

- Python for this repo: /home/kyung/miniconda3/envs/mosaic/bin/python (torch 2.8.0+cu128, numpy 2.0,
  scipy 1.13, sklearn 1.6, yaml). System python3 has no numpy; `conda env list` shows other envs too.
- Run from the encoder dir: `cd encoder && python train_encoder.py --model <name> --dataset <ds> --cuda <n>`;
  `--override k=v` lands in the dataset block of the model yml. Datasets: yelp, amazon_book, amazon_movie.
- Two H100 80GB GPUs; GPU 0 was fully occupied by the user's jobs on 2026-10-26 -> use `--cuda 1` for tests.
- `gh` CLI is not installed; fetch GitHub issues with curl against api.github.com.
- encoder/save/ (logs, results) was empty before 2026-10-26; delete smoke-test logs/results after testing.
