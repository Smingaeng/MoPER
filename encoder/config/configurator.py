import os
import yaml
import pickle
import argparse

ENCODER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(ENCODER_DIR), 'data')

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='MoPER')
    parser.add_argument('--model', type=str, default='moper', help='Model name: moper | promax | tall | hyperllm | gplr')
    parser.add_argument('--dataset', type=str, default='yelp', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: train.seed of the model yml)')
    parser.add_argument('--cuda', type=str, default='0', help='Device number')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='Model hyper-parameters as key=value (YAML values), e.g. --override top_k=1 reg_weight=1.0e-5')
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
        configs['train']['tag'] = args.tag or ' '.join(args.override)
        if args.save_model:
            configs['train']['save_model'] = True

        # RLMRec profile embeddings (ProMax: user and item profiles; MoPER: item profiles)
        configs['data']['dir'] = os.path.join(DATA_DIR, configs['data']['name'])
        usrprf_embeds_path = os.path.join(configs['data']['dir'], 'usr_emb_np.pkl')
        itmprf_embeds_path = os.path.join(configs['data']['dir'], 'itm_emb_np.pkl')
        if os.path.exists(usrprf_embeds_path) and os.path.exists(itmprf_embeds_path):
            with open(usrprf_embeds_path, 'rb') as f:
                configs['usrprf_embeds'] = pickle.load(f)
            with open(itmprf_embeds_path, 'rb') as f:
                configs['itmprf_embeds'] = pickle.load(f)

        return configs

configs = parse_configure()
