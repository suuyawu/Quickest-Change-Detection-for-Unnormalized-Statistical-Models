import os
import torch
import models
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load, make_footprint
from config import cfg


class RBM(Dataset):
    data_name = 'RBM'

    def __init__(self, root, **params):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.num_trials = params['num_trials']
        self.num_samples = params['num_samples']
        self.W = params['W']
        self.v = params['v']
        self.h = params['h']
        self.num_iters = params['num_iters']
        self.ptb_W = params['ptb_W']
        self.footprint = make_footprint(params)
        split_name = '{}_{}'.format(self.data_name, self.footprint)
        if not check_exists(os.path.join(self.processed_folder, split_name)):
            print('Not exists {}, create from scratch with {}.'.format(split_name, params))
            self.process()
        self.null, self.alter, self.meta = load(os.path.join(os.path.join(self.processed_folder, split_name)),
                                                mode='pickle')

    def __getitem__(self, index):
        null, alter = torch.tensor(self.null[index]), torch.tensor(self.alter[index])
        null_param = {'W': self.W, 'v': self.v, 'h': self.h}
        alter_param = {'W': torch.tensor(self.meta['W'][index]), 'v': self.v, 'h': self.h}
        input = {'null': null, 'alter': alter, 'null_param': null_param, 'alter_param': alter_param}
        return input

    def __len__(self):
        return self.num_trials

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        dataset = self.make_data()
        save(dataset, os.path.join(self.processed_folder, '{}_{}'.format(self.data_name, self.footprint)),
             mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nFootprint: {}'.format(self.data_name, self.__len__(), self.root,
                                                                         self.footprint)
        return fmt_str

    def make_data(self):
        with torch.no_grad():
            d = self.v.size(0)
            params = {'W': self.W, 'v': self.v, 'h': self.h}
            null_rbm = models.rbm(params).to(cfg['device'])
            null, alter = [], []
            alter_W = []
            for i in range(self.num_trials):
                ptb_W = self.ptb_W * torch.randn(self.W.size())
                alter_W_i = self.W + ptb_W
                params_i = {'W': alter_W_i, 'v': self.v, 'h': self.h}
                alter_rbm = models.rbm(params_i).to(cfg['device'])
                v = torch.randn(self.num_samples, d, device=cfg['device'])
                null_i = null_rbm(v, self.num_iters)
                alter_i = alter_rbm(v, self.num_iters)
                null.append(null_i.cpu())
                alter.append(alter_i.cpu())
                alter_W.append(alter_W_i.cpu())
            null = torch.stack(null, dim=0)
            alter = torch.stack(alter, dim=0)
            alter_W = torch.stack(alter_W, dim=0)
            null, alter = null.numpy(), alter.numpy()
            meta = {'W': alter_W.numpy()}
        return null, alter, meta
