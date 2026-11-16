# MoPER

<!-- TODO: paper title, authors, venue and link -->

This is the PyTorch implementation of **MoPER**, a mixture-of-experts recommender whose experts are built from
LLM-written user personas.

Every user is described by one or more personas (an LLM-written name and summary, embedded with Qwen3-Embedding-8B),
and every item by its profile embedding in the same text space. MoPER works in two phases:

- **Phase 1: persona experts.** The persona pool is clustered in two stages: k-means first splits it into persona
  kinds, then each kind is clustered with the number of clusters chosen by the silhouette score. Every cluster becomes
  one expert. No validation data is used.
- **Phase 2: expert diversification.** Every expert is a LightGCN whose layers are modulated by its persona prior (the
  diversifier). A shared gate sends each user and item to its top-2 experts, based on its similarity to the persona
  clusters. Training uses BPR with L2 regularisation, a gate balance loss, a persona alignment loss and a graph
  contrastive loss. At prediction time, a fixed content score between the user's taste personas and the item profile
  is added to the model score.

## 📝 Environment

Clone the repository:

```bash
git clone --depth 1 https://github.com/Smingaeng/MoPER.git
cd MoPER
```

Then create a conda environment:

```bash
conda create -y -n moper python=3.10
conda activate moper
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.0.2 scipy==1.13.1 pyyaml
```

All experiments were run on a single NVIDIA A100 (80GB) GPU.

😉 The code is built on the encoder framework of [RLMRec](https://github.com/HKUDS/RLMRec), which is based on
[SSLRec](https://github.com/HKUDS/SSLRec).

## 📂 Code Structure

```
MoPER
|--- encoder/                     # training and evaluation
|    |--- train_encoder.py        # entry point
|    |--- config/
|    |    |--- configurator.py    # command-line arguments
|    |    |--- modelconf/         # hyperparameters: moper.yml, promax.yml, tall.yml, hyperllm.yml, gplr.yml
|    |--- data_utils/             # data loading
|    |--- models/
|    |    |--- general_cf/        # moper.py, promax.py, tall.py, hyperllm.py, gplr.py
|    |--- trainer/                # training loop, metrics and logger
|--- generation/                  # LLM-generated inputs of the models
|    |--- moper/                  # user personas: generate_personas.py
|    |--- promax/                 # latent interactions: generate_rag.py
|    |--- hyperllm/               # tag hierarchy: generate_tags.py
|    |--- gplr/                   # personas: generate_personas.py, gplr.py
|--- data/                        # datasets (not in the repository, see below)
```

## 📚 Datasets

We use three datasets: *Amazon-Book*, *Yelp* and *Amazon-Movie*. Amazon-Book and Yelp are the processed datasets
released by RLMRec.

| Dataset      | #Users | #Items | #Train  | #Valid | #Test  |
|--------------|-------:|-------:|--------:|-------:|-------:|
| Amazon-Book  | 11,000 |  9,332 | 120,464 | 40,290 | 40,106 |
| Yelp         | 11,091 | 11,010 | 166,620 | 55,479 | 55,436 |
| Amazon-Movie |  7,351 |  7,002 |  57,283 | 19,028 | 18,863 |

Download the interaction matrices (about 8 MB) from [Google Drive](TODO) and put each dataset folder under `data/`
(`data/amazon_book`, `data/yelp`, `data/amazon_movie`). Only the training, validation and test matrices
(`trn_mat.pkl`, `val_mat.pkl`, `tst_mat.pkl`) are shared. The other files have to be built first (see
[Generation](#-generation)).

```
- amazon_book (yelp / amazon_movie)
|--- trn_mat.pkl            # training set (sparse matrix)                                   [download]
|--- val_mat.pkl            # validation set (sparse matrix)                                 [download]
|--- tst_mat.pkl            # test set (sparse matrix)                                       [download]
|--- new_trn_rag_mat.pkl    # training set with LLM-selected latent interactions (ProMax)    [generation]
|--- usr_emb_np.pkl         # user profile embeddings                                        [generation]
|--- itm_emb_np.pkl         # item profile embeddings                                        [generation]
|--- persona/
|    |--- persona_emb_v22.pkl           # user persona embeddings (MoPER)                    [generation]
|    |--- persona_clusters_v22_*.pkl    # Phase 1 clustering, written by MoPER on its first run
|--- hyperllm/              # LLM tag hierarchy (HyperLLM): id_to_tag / tag_to_tag / tag_to_item .json      [generation]
|--- gplr/                  # LLM personas (GPLR): persona2idx / tri_graph_uidx2pidx / tri_graph_tidx2pidx .json   [generation]
```

Each dataset has a training set, a validation set and a test set. The validation set decides when training stops.

Besides the three splits, each model reads the following files:

| Model      | Files                                                              |
|------------|--------------------------------------------------------------------|
| `moper`    | `itm_emb_np.pkl`, `persona/persona_emb_v22.pkl`                    |
| `promax`   | `usr_emb_np.pkl`, `itm_emb_np.pkl`, `new_trn_rag_mat.pkl`          |
| `tall`     | none                                                               |
| `hyperllm` | `usr_emb_np.pkl`, `itm_emb_np.pkl`, `hyperllm/*.json`              |
| `gplr`     | `gplr/*.json`                                                      |


## 🚀 Examples to run the codes

Run the commands from the root of the repository.

- MoPER:

  ```bash
  python encoder/train_encoder.py --model moper --dataset {dataset} --cuda 0
  ```

- Baselines:

  ```bash
  python encoder/train_encoder.py --model {model_name} --dataset {dataset} --cuda 0
  ```

Supported models/datasets:

* MoPER: `moper`
* model_name: `promax` ([ProMax](https://github.com/BlueGhostYi/ProRec)),
  `tall` ([TALL](https://github.com/JP-25/end-To-end-Adaptive-Local-Leanring-TALL-)),
  `hyperllm` ([HyperLLM](https://github.com/Qin-lab-code/HyperLLM)),
  `gplr` ([GPLR](https://github.com/Hanc1999/YouAreWhatYouBought))
* dataset: `amazon_book`, `yelp`, `amazon_movie`

Hyperparameters:

* The hyperparameters of each model are stored in `encoder/config/modelconf/{model_name}.yml`; dataset-specific values
  are in the `model.{dataset}` block. Running a model without `--override` uses the setting reported in the paper.
* All models share the same training protocol: the validation set is evaluated every 3 epochs, training stops when
  the mean of validation Recall@5, Recall@10, NDCG@5 and NDCG@10 has not improved for 30 evaluations, and the test
  Recall@K / NDCG@K (K = 5, 10, 20) of the best validation epoch are reported.
* `tall` follows the published TALL protocol (`gate_source: valid`, its gate reads the validation interactions);
  `gate_source: holdout` makes it use the training interactions only.

Outputs:

* Training logs: `encoder/save/log/{model_name}/{dataset}_{time}.log`
* Final test results: appended to `encoder/save/result/{model_name}_{dataset}.txt`
* HyperLLM (phase 1) and GPLR (graph eigenvectors) cache intermediate results in `encoder/save/checkpoint/` on their
  first run.

**For the other arguments (`--seed`, `--override key=value`, `--save-model`, ...), run the code with `--help`.**

## 🌟 Citation

The citation will be added when the paper is published.
