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
        self.num_trials = params['num_trials']
        self.num_samples = params['num_samples']
        self.power = params['power']
        self.tau = params['tau']
        self.num_dims = params['num_dims']
        self.ptb_tau = params['ptb_tau']
        self.footprint = make_footprint(params)
        split_name = '{}_{}'.format(self.data_name, self.footprint)
        if not check_exists(os.path.join(self.processed_folder, split_name)):
            print('Not exists {}, create from scratch with {}.'.format(split_name, params))
            self.process()
        self.null, self.alter, self.meta = load(os.path.join(os.path.join(self.processed_folder, split_name)),
                                                mode='pickle')

    def __getitem__(self, index):
        null, alter = torch.tensor(self.null[index]), torch.tensor(self.alter[index])
        null_param = {'power': self.power, 'tau': self.tau, 'num_dims': self.num_dims}
        alter_param = {'power': self.power, 'tau': torch.tensor(self.meta['tau'][index]), 'num_dims': self.num_dims}
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
        def unnormalized_pdf_normal(x, power, tau):
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

        total_samples = self.num_trials * self.num_samples
        d = self.num_dims
        null_nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, self.power, self.tau)))
        mcmc = MCMC(null_nuts, num_samples=total_samples, initial_params={'u': torch.zeros((d,))})
        mcmc.run()
        null = mcmc.get_samples()['u']
        null = null.view(self.num_trials, self.num_samples, -1)
        alter_tau = self.tau + self.ptb_tau * torch.ones((self.num_trials, *self.tau.size()))
        alter_nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, self.power, alter_tau[0])))
        mcmc = MCMC(alter_nuts, num_samples=total_samples, initial_params={'u': torch.zeros((d,))})
        mcmc.run()
        alter = mcmc.get_samples()['u']
        alter = alter.view(self.num_trials, self.num_samples, -1)
        null, alter = null.numpy(), alter.numpy()
        meta = {'tau': alter_tau.numpy()}
        return null, alter, meta
