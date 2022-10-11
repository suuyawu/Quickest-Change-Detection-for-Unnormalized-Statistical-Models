import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load, make_footprint
from pyro.infer import MCMC, NUTS


class EXP(Dataset):
    data_name = 'EXP'

    def __init__(self, root, **params):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.num_pre = params['num_pre']
        self.num_post = params['num_post']
        self.num_trials = params['num_trials']
        self.power = params['power']
        self.tau = params['tau']
        self.num_dims = params['num_dims']
        self.change_tau = params['change_tau']
        self.footprint = make_footprint(params)
        split_name = '{}_{}'.format(self.data_name, self.footprint)
        if not check_exists(os.path.join(self.processed_folder, split_name)):
            print('Not exists {}, create from scratch with {}.'.format(split_name, params))
            self.process()
        self.pre, self.post, self.meta = load(os.path.join(os.path.join(self.processed_folder, split_name)),
                                              mode='pickle')

    def __getitem__(self, index):
        pre, post = torch.tensor(self.pre[index]), torch.tensor(self.post[index])
        pre_param = {'power': torch.tensor(self.meta['pre']['power']),
                     'tau': torch.tensor(self.meta['pre']['tau']),
                     'num_dims': torch.tensor(self.meta['pre']['num_dims'])}
        post_param = {'power': torch.tensor(self.meta['post']['power']),
                      'tau': torch.tensor(self.meta['post']['tau']),
                      'num_dims': torch.tensor(self.meta['post']['num_dims'])}
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
        def unnormalized_pdf_exp(x, power, tau):
            d_ = len(x['u'])
            if d_ == 1:
                u_pdf = torch.exp(-tau * (x['u'] ** power))
            elif d_ == 2:
                u_pdf = torch.exp(-tau * (x['u'][0] ** power +
                                          x['u'][1] ** power +
                                          (x['u'][0] * x['u'][1]) ** (power / 2)))
            elif d_ == 3:
                u_pdf = torch.exp(-tau * (x['u'][0] ** power +
                                          x['u'][1] ** power +
                                          x['u'][2] ** power +
                                          (x['u'][0] * x['u'][1]) ** (power / 2) +
                                          (x['u'][0] * x['u'][2]) ** (power / 2) +
                                          (x['u'][1] * x['u'][2]) ** (power / 2)))
            elif d_ == 4:
                u_pdf = torch.exp(-tau * (x['u'][0] ** power +
                                          x['u'][1] ** power +
                                          x['u'][2] ** power +
                                          x['u'][3] ** power +
                                          (x['u'][0] * x['u'][1]) ** (power / 2) +
                                          (x['u'][0] * x['u'][2]) ** (power / 2) +
                                          (x['u'][0] * x['u'][3]) ** (power / 2) +
                                          (x['u'][1] * x['u'][2]) ** (power / 2) +
                                          (x['u'][1] * x['u'][3]) ** (power / 2) +
                                          (x['u'][2] * x['u'][3]) ** (power / 2)))
            else:
                raise ValueError('Not valid d')
            return u_pdf

        num_sampling_set = 10
        num_dims = self.num_dims
        pre_power = self.power
        pre_tau = self.tau
        pre_nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_exp(x, pre_power, pre_tau)))
        mcmc = MCMC(pre_nuts, num_samples=num_sampling_set * self.num_pre,
                    initial_params={'u': torch.zeros((num_dims,))})
        mcmc.run()
        pre = mcmc.get_samples()['u']
        randidx = torch.randint(0, pre.shape[0], (self.num_trials, self.num_pre))
        pre = pre[randidx]
        pre = pre.view(self.num_trials, self.num_pre, -1)
        post_power = self.power
        post_tau = self.tau + self.change_tau
        post_nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_exp(x, post_power, post_tau)))
        mcmc = MCMC(post_nuts, num_samples=num_sampling_set * self.num_post,
                    initial_params={'u': torch.zeros((num_dims,))})
        mcmc.run()
        post = mcmc.get_samples()['u']
        randidx = torch.randint(0, post.shape[0], (self.num_trials, self.num_post))
        post = post[randidx]
        pre, post = pre.numpy(), post.numpy()
        meta = {'pre': {'power': pre_power.numpy(), 'tau': pre_tau.numpy(), 'num_dims': num_dims.numpy()},
                'post': {'power': post_power.numpy(), 'tau': post_tau.numpy(), 'num_dims': num_dims.numpy()}}
        return pre, post, meta
