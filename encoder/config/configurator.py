import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

ENCODER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(ENCODER_DIR), 'data')

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='mosaic_v2')
    parser.add_argument('--model', type=str, default='lightgcn', help='Model name')
    parser.add_argument('--dataset', type=str, default='yelp', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='Model hyper-parameters as key=value (YAML values), e.g. --override top_k=1 segment_num=16')
    parser.add_argument('--d_text', type=int, default=None,
                        help='PERSONA-MOSAIC: d_text, the PCA dimension of the LLM text embeddings used for k-means '
                             '(e.g. 32). Default: the raw text embedding dimension (4096, no PCA); same as --override text_dim=<d>')
    parser.add_argument('--tag', type=str, default='', help='Run label written to the log and result files')
    parser.add_argument('--save-model', action='store_true', help='Save the best-epoch parameters to save/checkpoint/<model>/')
    args, _ = parser.parse_known_args()

    # cuda
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    yml_path = os.path.join(ENCODER_DIR, 'config', 'modelconf', '{}.yml'.format(model_name))
    if not os.path.exists(yml_path):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open(yml_path, encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()

        if args.model is not None:
            if configs['model']['name'] != args.model:
                raise Exception("Please check whether the model names are consistent.")

        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
        configs['device'] = args.device
        if args.dataset is not None:
            configs['data']['name'] = args.dataset
        if args.seed is not None:
            configs['train']['seed'] = args.seed
        # Hyper-parameter overrides go to the dataset-specific block when it exists (BaseModel reads that block).
        target = configs['model'].get(configs['data']['name'], configs['model'])
        for item in args.override:
            key, value = item.split('=', 1)
            target[key] = yaml.safe_load(value)
        auto_tag = list(args.override)
        if args.d_text is not None:
            target['text_dim'] = int(args.d_text)
            auto_tag.append('d_text={}'.format(args.d_text))
        configs['train']['tag'] = args.tag or ' '.join(auto_tag)
        if args.save_model:
            configs['train']['save_model'] = True

        # semantic embeddings
        configs['data']['dir'] = os.path.join(DATA_DIR, configs['data']['name'])
        usrprf_embeds_path = os.path.join(configs['data']['dir'], 'usr_emb_np.pkl')
        itmprf_embeds_path = os.path.join(configs['data']['dir'], 'itm_emb_np.pkl')
        if os.path.exists(usrprf_embeds_path) and os.path.exists(itmprf_embeds_path):
            with open(usrprf_embeds_path, 'rb') as f:
                configs['usrprf_embeds'] = pickle.load(f)
            with open(itmprf_embeds_path, 'rb') as f:
                configs['itmprf_embeds'] = pickle.load(f)

        # DisCoRec / Mosaic: LLM intent / conformity embeddings (CoMoE generation, data/<dataset>/)
        for key, fname in [('usrint_embeds', 'user_intent_emb_3.pkl'), ('itmint_embeds', 'item_intent_emb_3.pkl'),
                           ('usrconf_embeds', 'user_conf_emb.pkl'), ('itmconf_embeds', 'item_conf_emb.pkl'),
                           ('commint_embeds', 'community_intent_emb.pkl'), ('commconf_embeds', 'community_conf_emb.pkl')]:
            path = os.path.join(configs['data']['dir'], fname)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    configs[key] = pickle.load(f)

        return configs

configs = parse_configure()
