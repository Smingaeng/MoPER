import os
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim
from trainer.metrics import Metric
from models.bulid_model import build_model
from config.configurator import configs
from .utils import log_exceptions

SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'save')


def init_seed():
    if 'reproducible' in configs['train']:
        if configs['train']['reproducible']:
            seed = configs['train']['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    def create_optimizer(self, model):
        optim_config = configs['optimizer']
        if optim_config['name'] == 'adam':
            self.optimizer = optim.Adam([q for q in model.parameters() if q.requires_grad], lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])

    def train_epoch(self, model, epoch_idx):
        # prepare training data
        train_dataloader = self.data_handler.train_dataloader
        train_dataloader.dataset.sample_negs()

        # for recording loss
        loss_log_dict = {}
        ep_loss = 0
        # start this epoch
        model.train()
        for _, tem in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            batch_data = list(map(lambda x: x.long().to(configs['device']), tem))
            loss, loss_dict = model.cal_loss(batch_data)
            ep_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # record loss
            for loss_name in loss_dict:
                _loss_val = float(loss_dict[loss_name]) / len(train_dataloader)
                if loss_name not in loss_log_dict:
                    loss_log_dict[loss_name] = _loss_val
                else:
                    loss_log_dict[loss_name] += _loss_val

        if 'log_loss' in configs['train'] and configs['train']['log_loss']:
            self.logger.log(loss_log_dict, save_to_log=False, print_to_console=True)
        return ep_loss / max(len(train_dataloader), 1)

    def _early_stop_score(self, eval_result):
        """Validation score watched by early stopping and best-epoch selection (train.early_stop_metric):
          goal_mean   = mean of recall@5, recall@10, ndcg@5, ndcg@10 (the reported metrics)
          recall_last = recall at the largest k"""
        mode = str(configs['train'].get('early_stop_metric', 'recall_last')).lower()
        if mode == 'recall_last':
            return eval_result['recall'][-1]
        if mode == 'goal_mean':
            ks = list(configs['test']['k'])
            vals = [float(eval_result[m][ks.index(k)]) for m in ('recall', 'ndcg') for k in (5, 10)]
            return sum(vals) / len(vals)
        raise ValueError("early_stop_metric must be 'recall_last' or 'goal_mean', got {!r}".format(mode))

    @log_exceptions
    def train(self, model):
        now_patience = 0
        best_epoch = 0
        best_score = -1e9
        self.create_optimizer(model)
        train_config = configs['train']

        for epoch_idx in range(train_config['epoch']):
            # train
            self.train_epoch(model, epoch_idx)
            # evaluate
            if epoch_idx % train_config['test_step'] == 0:
                eval_result = self.evaluate(model, epoch_idx)
                stop_score = self._early_stop_score(eval_result)
                if stop_score > best_score:
                    now_patience = 0
                    best_epoch = epoch_idx
                    best_score = stop_score
                    best_state_dict = deepcopy(model.state_dict())
                else:
                    now_patience += 1

                # early stop
                if now_patience == train_config['patience']:
                    break

        # evaluation again
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        self.evaluate(model)

        # final test
        self.logger.log("Test--------------------------------------------------------------------------")
        model = build_model(self.data_handler).to(configs['device'])
        model.load_state_dict(best_state_dict)
        test_result = self.test(model)

        self.save_model(model)  # best-epoch parameters, if train.save_model

        # save result
        self.logger.log("Best Epoch {}. Final test result: {}.".format(best_epoch, test_result))
        os.makedirs(os.path.join(SAVE_DIR, 'result'), exist_ok=True)
        test_file = os.path.join(SAVE_DIR, 'result', configs['model']['name'] + '_' + configs['data']['name'] + '.txt')
        with open(test_file, 'a') as f:
            tag = configs['train'].get('tag', '')
            f.write("{}Best Epoch {}. Final test result: {}. \n".format('[{}] '.format(tag) if tag else '', best_epoch, test_result))

    @log_exceptions
    def evaluate(self, model, epoch_idx=None):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.valid_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Validation set', epoch_idx=epoch_idx)
        return eval_result

    @log_exceptions
    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log_eval(eval_result, configs['test']['k'], data_type='Test set')
        return eval_result

    def save_model(self, model):
        if configs['train']['save_model']:
            model_name = configs['model']['name']
            save_dir_path = os.path.join(SAVE_DIR, 'checkpoint', model_name)
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            tag = configs['train'].get('tag', '').replace(' ', '_').replace('=', '-')
            path = '{}/{}-{}-{}{}.pth'.format(save_dir_path, model_name, configs['data']['name'], configs['train']['seed'],
                                             '-' + tag if tag else '')
            torch.save(model.state_dict(), path)
            self.logger.log("Save model parameters to {}".format(path))
