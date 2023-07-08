import math
import numpy as np

class CUSUM:
    def __init__(self, arl, pre_data, pre_model, post_model):
        super().__init__()
        self.arl = arl
        self._initialize()
        self.threshold = self.make_threshold(arl, pre_data, pre_model, post_model, np.log(arl), 0, np.log(arl))

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_nll = []
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

    def _update(self, sample, pre_model, post_model):
        self.stopping_time += 1
        inst_nll = self.nll(sample, pre_model.pdf, post_model.pdf)
        self.detector_score = max((self.detector_score + inst_nll).item(), 0)
        self.cum_nll.append(inst_nll.item())
        self.cum_scores.append(self.detector_score)
        if self.detector_score > self.threshold:
            self.detect = True
        else:
            self.detect = False
        return self.detector_score, self.detect, self.threshold

    def nll(self, sample, pre_pdf, post_pdf):
        """Calculate instant Negative Log Likelihood"""
        NLL_item = post_pdf(sample).log() - pre_pdf(sample).log()
        return NLL_item
