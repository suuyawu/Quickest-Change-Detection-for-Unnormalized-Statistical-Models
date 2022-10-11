import torch
import torch.nn.functional as F
from config import cfg
from utils import recur


def EDD(cp, pre_data, post_data):
    with torch.no_grad():
        target_cp = len(pre_data)
        N = len(pre_data) + len(post_data)
        edd = cp - target_cp
    return edd


def RMSE(output, target):
    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    return rmse


class Metric(object):
    def __init__(self, metric_name):
        self.metric_name = self.make_metric_name(metric_name)
        self.metric = {'EDD': (lambda input, output: EDD(output['cp'], input['pre_data'], input['post_data']))}
        self.reset()

    def make_metric_name(self, metric_name):
        return metric_name

    def reset(self):
        pivot = None
        pivot_name = None
        pivot_direction = None
        self.pivot, self.pivot_name, self.pivot_direction = pivot, pivot_name, pivot_direction
        return

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

    def load_state_dict(self, state_dict):
        self.pivot = state_dict['pivot']
        self.pivot_name = state_dict['pivot_name']
        self.pivot_direction = state_dict['pivot_direction']
        return

    def state_dict(self):
        return {'pivot': self.pivot, 'pivot_name': self.pivot_name, 'pivot_direction': self.pivot_direction}
