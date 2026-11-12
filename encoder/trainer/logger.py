import os
import logging
import datetime
from config.configurator import configs

def get_local_time():
    return datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')

class Logger(object):
    def __init__(self, log_configs=True):
        model_name = configs['model']['name']
        log_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'save', 'log', model_name)

        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)
        dataset_name = configs['data']['name']
        log_file = logging.FileHandler('{}/{}_{}.log'.format(log_dir_path, dataset_name, get_local_time()), 'a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_file.setFormatter(formatter)
        self.logger.addHandler(log_file)
        if log_configs:
            self.log('Optimizer| ' + ', '.join(f"{k}:{v}" for k, v in configs['optimizer'].items()))
            self.log('Train| ' + ', '.join(f"{k}:{v}" for k, v in configs['train'].items()))
            self.log('Test| ' + ', '.join(f"{k}:{v}" for k, v in configs['test'].items()))
            self.log('Data| ' + ', '.join(f"{k}:{v}" for k, v in configs['data'].items()))
            self.log('Model| ' + ', '.join(f"{k}:{v}" for k, v in configs['model'].items()))

        self.log('-----------------------------------------------------------------------------------------------------')

    def log(self, message, save_to_log=True, print_to_console=True):
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)

    def log_eval(self, eval_result, k, data_type, save_to_log=True, print_to_console=True, epoch_idx=None):
        if epoch_idx is not None:
            message = 'Epoch {:3d} {} '.format(epoch_idx, data_type)
        else:
            message = ''

        for metric in eval_result:
            message += '['
            for i in range(len(k)):
                message += '{}@{}: {:.4f} '.format(metric, k[i], eval_result[metric][i])
            message += '] '
        if save_to_log:
            self.logger.info(message)
        if print_to_console:
            print(message)