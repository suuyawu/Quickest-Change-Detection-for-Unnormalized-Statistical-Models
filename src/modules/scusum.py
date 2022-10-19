import math
import numpy as np
from scipy.optimize import fsolve


class SCUSUM:
    def __init__(self, arl, pre_data, pre_model):
        super().__init__()
        self.arl = arl
        self.threshold = math.log(arl)
        self.hyper_lambda = self.make_hyper_lambda(pre_data, pre_model)
        self._initialize()

    def _reset(self):
        self._initialize()

    def _initialize(self, ):
        self.stop = False
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_hst = []
        self.cum_scores = []

    def make_hyper_lambda(self, pre_data, pre_model):
        def func(x):
            hs = pre_model.hscore(pre_data).detach().cpu().numpy()
            out = np.exp(x * hs).mean() - 1
            return [out]
        hyper_lambda = fsolve(func, [10])[0]
        return hyper_lambda

    def _update(self, sample, pre_model, post_model):
        self.stopping_time += 1
        inst_hst = self.hst(sample, pre_model.hscore, post_model.hscore)
        self.detector_score = max((self.detector_score + inst_hst).item(), 0)
        self.cum_hst.append(inst_hst.item())
        self.cum_scores.append(self.detector_score)
        if self.detector_score > self.threshold:
            self.detect = True
        else:
            self.detect = False
        return self.detector_score, self.detect, self.threshold

    def hst(self, sample, pre_hscore, post_hscore):
        """Calculate instant Hyvarinen Score Difference"""
        Hscore_item = -post_hscore(sample) + pre_hscore(sample)
        return self.hyper_lambda * Hscore_item
