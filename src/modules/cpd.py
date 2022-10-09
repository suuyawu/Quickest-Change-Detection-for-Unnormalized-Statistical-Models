import torch
import models
from config import cfg
from .cusum import CUSUM
from .scusum import SCUSUM
from .scanb import SCANB
from .utils import load_kernel


class ChangePointDetecion:
    def __init__(self, test_mode, arl, noise):
        self.test_mode = test_mode
        self.arl = arl
        self.noise = noise
        self.cpd = self.make_cpd()
        self.reset()

    def reset(self):
        self.stats = {'score': [], 'detect': []}
        return

    def make_cpd(self):
        if self.test_mode == 'cusum':
            cpd = CUSUM(self.arl)
        elif self.test_mode == 'scusum':
            cpd = SCUSUM(self.arl)
        elif self.test_mode == 'scanb':
            cpd = SCANB(kernel=load_kernel('gaussian_rbf'), initial_data=initial_data, ert=self.arl, window_size=25,
                        test_every_k=1, n_samples=25000)
        else:
            print(self.test_mode)
            raise ValueError('Not valid test mode')
        return cpd

    def test(self, input):
        noise = cfg['noise']
        data = torch.cat([input['pre_data'], input['post_data']], dim=0)
        data = data + noise * torch.randn(data.size(), device=data.device)
        cp = len(data)
        self.cpd._reset()
        if cfg['test_mode'] in ['cusum', 'scusum']:
            pre_param, post_param = input['pre_param'], input['post_param']
            pre_model = eval('models.{}(pre_param).to(cfg["device"])'.format(cfg['model_name']))
            post_model = eval('models.{}(post_param).to(cfg["device"])'.format(cfg['model_name']))
        else:
            pre_model = None
            post_model = None
        if self.test_mode in ['cusum', 'scusum']:
            for t, data_i in enumerate(data):
                score, detect = self.cpd._update(data_i.reshape(1, -1), pre_model, post_model)
                self.stats['score'].append(score)
                self.stats['detect'].append(detect)
                if cp == len(data) and detect:
                    cp = t + 1
        elif self.test_mode in ['scanb']:
            for t, data_i in enumerate(data):
                score, detect = self.cpd._update(data_i.reshape(1, -1))
                self.stats['score'].append(score)
                self.stats['detect'].append(detect)
                if cp == len(data) and detect:
                    cp = t + 1
        output = {'cp': cp}
        return output
