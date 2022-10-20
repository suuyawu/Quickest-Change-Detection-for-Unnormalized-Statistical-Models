import torch
import models
from config import cfg
from .cusum import CUSUM
from .scusum import SCUSUM
from .scanb import SCANB
from .calm import CALM
from .utils import load_kernel


class ChangePointDetecion:
    def __init__(self, test_mode, arl, noise, dataset, pre_length=None):
        self.test_mode = test_mode
        self.arl = arl
        self.noise = noise
        self.pre_length = pre_length
        self.reset()
        self.cpd = self.make_cpd(dataset)

    def reset(self):
        self.stats = {'score': [], 'detect': [], 'threshold': []}
        if self.test_mode == 'scusum':
            self.stats['lambda'] = []
        return

    def clean(self):
        self.cpd = None
        return

    def make_cpd(self, dataset):
        pre_data = torch.tensor(dataset.pre)
        pre_data = pre_data.view(-1, pre_data.size(-1))
        cpds = []
        for i in range(cfg['num_trials']):
            perturb_index = torch.randperm(pre_data.size(0))
            if self.pre_length is None:
                initial_data = pre_data[perturb_index][:dataset.pre.shape[1]].to(cfg['device'])
            else:
                initial_data = pre_data[perturb_index][:self.pre_length].to(cfg['device'])
            if self.test_mode == 'cusum':
                cpd = CUSUM(self.arl)
            elif self.test_mode == 'scusum':
                pre_param = dataset[0]['pre_param']
                pre_model = eval('models.{}(pre_param).to(cfg["device"])'.format(cfg['model_name']))
                cpd = SCUSUM(self.arl, initial_data, pre_model)
                self.stats['lambda'].append(cpd.hyper_lambda)
            elif self.test_mode == 'scanb':
                cpd = SCANB(kernel=load_kernel('gaussian_rbf'), initial_data=initial_data, ert=self.arl, window_size=25,
                            test_every_k=1, n_samples=25000)
            elif self.test_mode == 'calm':
                cpd = CALM(kernel=load_kernel('gaussian_rbf'), initial_data=initial_data, ert=self.arl, window_size=25,
                            test_every_k=1, n_bootstraps=25000)
            else:
                raise ValueError('Not valid test mode')
            cpds.append(cpd)
        return cpds

    def test(self, i, input):
        noise = cfg['noise']
        data = torch.cat([input['pre_data'], input['post_data']], dim=0)
        data = data + noise * torch.randn(data.size(), device=data.device)
        target_cp = len(input['pre_data'])
        cp = len(data)
        self.cpd[i]._reset()
        if cfg['test_mode'] in ['cusum', 'scusum']:
            pre_param, post_param = input['pre_param'], input['post_param']
            pre_model = eval('models.{}(pre_param).to(cfg["device"])'.format(cfg['model_name']))
            post_model = eval('models.{}(post_param).to(cfg["device"])'.format(cfg['model_name']))
        else:
            pre_model = None
            post_model = None
        if self.test_mode in ['scusum']:
            for t, data_i in enumerate(data):
                score, detect, threshold = self.cpd[i]._update(data_i.reshape(1, -1), pre_model, post_model)
                self.stats['score'].append(score)
                self.stats['detect'].append(detect)
                self.stats['threshold'].append(threshold)
                if cp == len(data) and detect and t > target_cp - 1:
                    cp = t + 1
        elif self.test_mode in ['cusum']:
            for t, data_i in enumerate(data):
                score, detect, threshold = self.cpd[i]._update(data_i.reshape(1, -1), pre_model, post_model)
                self.stats['score'].append(score)
                self.stats['detect'].append(detect)
                self.stats['threshold'].append(threshold)
                if cp == len(data) and detect and t > target_cp - 1:
                    cp = t + 1
        elif self.test_mode in ['scanb', 'calm']:
            for t, data_i in enumerate(data):
                score, detect, threshold = self.cpd[i]._update(data_i.reshape(1, -1))
                self.stats['score'].append(score)
                self.stats['detect'].append(detect)
                self.stats['threshold'].append(threshold)
                if cp == len(data) and detect and t > target_cp - 1:
                    cp = t + 1
        else:
            raise ValueError('Not valid test mode')
        output = {'cp': cp}
        return output
