import collections.abc as container_abcs
import errno
import hashlib
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=1, pad_value=0, value_range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, value_range=value_range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['data_size'] = {'test': len(dataset['test'])}
    return


def process_control():
    data_name_list = cfg['control']['data_name'].split('-')
    cfg['data_name'], cfg['num_dims'] = data_name_list[0], int(data_name_list[1])
    cfg['num_pre'] = int(cfg['control']['num_pre'])
    cfg['num_post'] = int(cfg['control']['num_post'])
    cfg['change_point'] = int(cfg['control']['change_point'])
    cfg['change'] = cfg['control']['change']
    cfg['noise'] = float(cfg['control']['noise'])
    cfg['test_mode'] = cfg['control']['test_mode']
    cfg['arl'] = int(cfg['control']['arl'])
    if 'pre_length' in cfg['control']:
        cfg['pre_length'] = int(cfg['control']['pre_length'])
    else:
        cfg['pre_length'] = None
    cfg['model_name'] = cfg['data_name'].lower()
    cfg['num_trials'] = 100
    cfg['cpd'] = {}
    cfg['cpd']['batch_size'] = {'test': 1}
    cfg['cpd']['shuffle'] = {'test': False}
    cfg['MVN'] = {'mean': torch.zeros(cfg['num_dims']),
                  'logvar': np.log(0.5) * torch.ones((cfg['num_dims'], cfg['num_dims']))}
    cfg['MVN']['logvar'].fill_diagonal_(0)
    cfg['EXP'] = {'power': torch.tensor([4.]), 'tau': torch.tensor([1.]), 'num_dims': torch.tensor([cfg['num_dims']])}
    dim_v = cfg['num_dims']
    dim_h = int(np.ceil(0.8 * cfg['num_dims']))
    generator = torch.Generator()
    generator.manual_seed(cfg['seed'])
    W = torch.randn(dim_v, dim_h, generator=generator)
    v = torch.randn(dim_v, generator=generator)
    h = torch.randn(dim_h, generator=generator)
    cfg['RBM'] = {'W': W, 'v': v, 'h': h, 'num_iters': int(1000)}
    cfg['params'] = make_params(cfg['data_name'])
    cfg['footprint'] = make_footprint(cfg['params'])
    # model_name = cfg['model_name']
    # cfg[model_name] = {}
    # cfg[model_name]['optimizer_name'] = 'Adam'
    # cfg[model_name]['lr'] = 1e-3
    # cfg[model_name]['betas'] = (0.9, 0.999)
    # cfg[model_name]['momentum'] = 0.9
    # cfg[model_name]['nesterov'] = True
    # cfg[model_name]['weight_decay'] = 0
    # cfg[model_name]['num_iters'] = 20
    return


def make_stats():
    stats = {}
    stats_path = os.path.join('res', 'stats')
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(parameters, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(path, verbose=True, resume_mode=1):
    if os.path.exists(path) and resume_mode == 1:
        result = load(path)
        if verbose:
            print('Resume from {}'.format(result['epoch']))
    else:
        if resume_mode == 1:
            print('Not exists: {}'.format(path))
        result = None
    return result


def collate(input):
    for k in input:
        if k in ['pre_param', 'post_param']:
            input[k] = input[k][0]
        elif k in ['pre_data', 'post_data']:
            input[k] = torch.cat(input[k], 0)
        else:
            input[k] = torch.cat(input[k], 0)
    return input


def make_params(data_name):
    if data_name == 'MVN':
        mean = cfg['MVN']['mean']
        logvar = cfg['MVN']['logvar']
        change_mean, change_logvar = cfg['change'].split('-')
        change_mean, change_logvar = float(change_mean), float(change_logvar)
        params = {'data_name': data_name, 'num_pre': cfg['num_pre'], 'num_post': cfg['num_post'],
                  'num_trials': cfg['num_trials'], 'mean': mean, 'logvar': logvar,
                  'change_mean': change_mean, 'change_logvar': change_logvar}
    elif data_name == 'EXP':
        power = cfg['EXP']['power']
        tau = cfg['EXP']['tau']
        num_dims = cfg['EXP']['num_dims']
        change_tau = float(cfg['change'])
        params = {'data_name': data_name, 'num_pre': cfg['num_pre'], 'num_post': cfg['num_post'],
                  'num_trials': cfg['num_trials'], 'power': power, 'tau': tau, 'num_dims': num_dims,
                  'change_tau': change_tau}
    elif data_name == 'RBM':
        W = cfg['RBM']['W']
        v = cfg['RBM']['v']
        h = cfg['RBM']['h']
        num_iters = cfg['RBM']['num_iters']
        change_W = float(cfg['change'])
        params = {'data_name': data_name, 'num_pre': cfg['num_pre'], 'num_post': cfg['num_post'],
                  'num_trials': cfg['num_trials'], 'W': W, 'v': v, 'h': h, 'num_iters': num_iters, 'change_W': change_W}
    else:
        raise ValueError('Not valid data name')
    return params


def make_footprint(params):
    hash_name = '_'.join([str(params[x]) for x in params]).encode('utf-8')
    m = hashlib.sha256(hash_name)
    footprint = m.hexdigest()
    return footprint
