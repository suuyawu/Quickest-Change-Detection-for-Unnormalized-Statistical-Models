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
        self.num_pre = params['num_pre']
        self.num_total = params['num_total']
        self.num_trials = params['num_trials']
        self.W = params['W']
        self.v = params['v']
        self.h = params['h']
        self.num_iters = params['num_iters']
        self.change_W = params['change_W']
        self.footprint = make_footprint(params)
        split_name = '{}_{}'.format(self.data_name, self.footprint)
        if not check_exists(os.path.join(self.processed_folder, split_name)):
            print('Not exists {}, create from scratch with {}.'.format(split_name, params))
            self.process()
        self.pre, self.post, self.meta = load(os.path.join(os.path.join(self.processed_folder, split_name)),
                                              mode='pickle')

    def __getitem__(self, index):
        pre, post = torch.tensor(self.pre[index]), torch.tensor(self.post[index])
        pre_param = {'W': torch.tensor(self.meta['pre']['W']),
                     'v': torch.tensor(self.meta['pre']['v']),
                     'h': torch.tensor(self.meta['pre']['h'])}
        post_param = {'W': torch.tensor(self.meta['post']['W']),
                      'v': torch.tensor(self.meta['post']['v']),
                      'h': torch.tensor(self.meta['post']['h'])}
        input = {'pre_data': pre, 'pre_param': pre_param, 'post_data': post, 'post_param': post_param}
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
            num_dims = self.v.size(0)
            pre_W = self.W
            pre_params = {'W': pre_W, 'v': self.v, 'h': self.h}
            post_W = self.W + self.change_W * torch.randn(self.W.size())
            post_params = {'W': post_W, 'v': self.v, 'h': self.h}
            pre_rbm = models.rbm(pre_params).to(cfg['device'])
            post_rbm = models.rbm(post_params).to(cfg['device'])
            pre = torch.randn(self.num_trials * self.num_pre, num_dims, device=cfg['device'])
            pre = pre_rbm(pre, self.num_iters)
            pre = pre.view(self.num_trials, self.num_pre, -1)
            post = torch.randn(self.num_trials * (self.num_total - self.num_pre), num_dims, device=cfg['device'])
            post = post_rbm(post, self.num_iters)
            post = post.view(self.num_trials, self.num_total - self.num_pre, -1)
            pre, post = pre.cpu().numpy(), post.cpu().numpy()
            meta = {'pre': {'W': pre_W.cpu().numpy(), 'v': self.v.cpu().numpy(), 'h': self.h.cpu().numpy()},
                    'post': {'W': post_W.cpu().numpy(), 'v': self.v.cpu().numpy(), 'h': self.h.cpu().numpy()}}
        return pre, post, meta
