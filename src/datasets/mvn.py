import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load, make_footprint


class MVN(Dataset):
    data_name = 'MVN'

    def __init__(self, root, **params):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.num_pre = params['num_pre']
        self.num_total = params['num_total']
        self.num_trials = params['num_trials']
        self.mean = params['mean']
        self.logvar = params['logvar']
        self.change_mean = params['change_mean']
        self.change_logvar = params['change_logvar']
        self.footprint = make_footprint(params)
        split_name = '{}_{}'.format(self.data_name, self.footprint)
        if not check_exists(os.path.join(self.processed_folder, split_name)):
            print('Not exists {}, create from scratch with {}.'.format(split_name, params))
            self.process()
        self.pre, self.post, self.meta = load(os.path.join(os.path.join(self.processed_folder, split_name)),
                                              mode='pickle')

    def __getitem__(self, index):
        pre, post = torch.tensor(self.pre[index]), torch.tensor(self.post[index])
        pre_param = {'mean': torch.tensor(self.meta['pre']['mean']),
                     'logvar': torch.tensor(self.meta['pre']['logvar'])}
        post_param = {'mean': torch.tensor(self.meta['post']['mean']),
                      'logvar': torch.tensor(self.meta['post']['logvar'])}
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
        num_dims = self.mean.size(-1)
        pre_mean = self.mean
        pre_logvar = self.logvar
        pre_mvn = torch.distributions.multivariate_normal.MultivariateNormal(pre_mean, pre_logvar.exp())
        pre = pre_mvn.sample((self.num_trials, self.num_pre))
        post_mean = self.mean + self.change_mean
        post_logvar = self.logvar + self.change_logvar * torch.eye(num_dims)
        post_mvn = torch.distributions.multivariate_normal.MultivariateNormal(post_mean, post_logvar.exp())
        post = post_mvn.sample((self.num_trials, self.num_total - self.num_pre))
        pre, post = pre.numpy(), post.numpy()
        meta = {'pre': {'mean': pre_mean.numpy(), 'logvar': pre_logvar.numpy()},
                'post': {'mean': post_mean.numpy(), 'logvar': post_logvar.numpy()}}
        return pre, post, meta
