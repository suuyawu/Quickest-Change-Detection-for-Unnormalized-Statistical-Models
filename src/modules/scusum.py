import math
import numpy as np
from scipy.optimize import fsolve, minimize_scalar


class SCUSUM:
    def __init__(self, arl, pre_data, initial_data, pre_model, post_model):
        super().__init__()
        self.arl = arl
        self._initialize()
        self.hyper_lambda = self.make_hyper_lambda(initial_data, pre_model)
        self.threshold = self.make_threshold(arl, pre_data, pre_model, post_model, 0, 0, math.log(arl))

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_hst = []
        self.cum_scores = []

    def make_threshold(self, exp_arl, pre_data, pre_model, post_model, threshold, low_threshold, high_threshold,
                       repeat=0):
        self.threshold = threshold
        arl = []
        for i in range(36):
            self._reset()
            t = 0
            detect = False
            while (not detect) and t < len(pre_data[i]):
                _, detect, _ = self._update(pre_data[i][t].reshape(1, -1), pre_model, post_model)
                t += 1
            arl.append(t)
        emp_arl = np.mean(arl)
        repeat += 1
        if abs(emp_arl - exp_arl) <= 5 or repeat > 20:
            return threshold
        elif emp_arl > exp_arl:
            high_threshold = threshold
        else:
            low_threshold = threshold
        return self.make_threshold(exp_arl, pre_data, pre_model, post_model,
                                   (high_threshold + low_threshold) / 2, low_threshold, high_threshold, repeat=repeat)

    def make_hyper_lambda(self, initial_data, pre_model):
        def func(x):
            hs = pre_model.hscore(initial_data).detach().cpu().numpy()
            out = np.exp(x * hs).mean() - 1
            return [out]
        hyper_lambda = fsolve(func, [1])[0]
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
