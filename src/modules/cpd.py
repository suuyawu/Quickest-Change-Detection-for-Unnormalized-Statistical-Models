import torch
import models
from config import cfg
from .cusum import CUSUM
from .scusum import SCUSUM
from .scanb import SCANB
from .calm import CALM
from .utils import load_kernel
from utils import to_device

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
        pre_data = torch.tensor(dataset.pre).to(cfg['device'])
        input = to_device(dataset[0], cfg['device'])
        pre_param, post_param = input['pre_param'], input['post_param']
        initial_data = pre_data.view(-1, pre_data.size(-1))
        perturb_index = torch.randperm(initial_data.size(0))
        if self.pre_length is None:
            initial_data = initial_data[perturb_index][:dataset.pre.shape[1]].to(cfg['device'])
        else:
            initial_data = initial_data[perturb_index][:self.pre_length].to(cfg['device'])
        pre_model = eval('models.{}(pre_param).to(cfg["device"])'.format(cfg['model_name']))
        post_model = eval('models.{}(post_param).to(cfg["device"])'.format(cfg['model_name']))
        if self.test_mode == 'cusum':
            cpd = CUSUM(self.arl, pre_data, pre_model, post_model)
        elif self.test_mode == 'scusum':
            cpd = SCUSUM(self.arl, pre_data, initial_data, pre_model, post_model)
            self.stats['lambda'].append(cpd.hyper_lambda)
        elif self.test_mode == 'scanb':
            cpd = SCANB(kernel=load_kernel('gaussian_rbf'), initial_data=initial_data, ert=self.arl, window_size=25,
                        test_every_k=1, n_samples=25000)
        elif self.test_mode == 'calm':
            cpd = CALM(kernel=load_kernel('gaussian_rbf'), initial_data=initial_data, ert=self.arl, window_size=25,
                        test_every_k=1, n_bootstraps=25000)
        else:
            raise ValueError('Not valid test mode')
        return cpd

    def test(self, i, input):
        change_data = torch.cat([input['pre_data'][:cfg['change_point']], input['post_data']], dim=0)
        pre_data = torch.cat([input['pre_data'][:cfg['change_point']], input['pre_data']], dim=0)
        change_data = change_data + cfg['noise'] * torch.randn(change_data.size(), device=change_data.device)
        pre_data = pre_data + cfg['noise'] * torch.randn(pre_data.size(), device=pre_data.device)
    
        if self.test_mode in ['scusum', 'cusum']:
            pre_param, post_param = input['pre_param'], input['post_param']
            pre_model = eval('models.{}(pre_param).to(cfg["device"])'.format(cfg['model_name']))
            post_model = eval('models.{}(post_param).to(cfg["device"])'.format(cfg['model_name']))
        elif self.test_mode in ['scanb', 'calm']:
            pre_model = None
            post_model = None
        else:
            raise ValueError('Not valid test mode')
        #edd
        self.cpd._reset()
        cp = 0
        detect = False
        while (not detect) and cp < len(change_data):
            score, detect, threshold = self.cpd._update(change_data[cp].reshape(1, -1), pre_model, post_model)
            self.stats['score'].append(score)
            self.stats['detect'].append(detect)
            self.stats['threshold'].append(threshold)
            cp +=1
        #arl
        self.cpd._reset()
        arl = 0
        detect = False
        while (not detect) and arl < len(pre_data):
            score, detect, threshold = self.cpd._update(pre_data[arl].reshape(1, -1), pre_model, post_model)
            self.stats['score'].append(score)
            self.stats['detect'].append(detect)
            self.stats['threshold'].append(threshold)
            arl +=1
        output = {'cp': cp, 'arl': arl}
        return output
