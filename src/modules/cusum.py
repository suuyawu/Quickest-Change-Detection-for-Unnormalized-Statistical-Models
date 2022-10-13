import math


class CUSUM:
    def __init__(self, arl):
        super().__init__()
        self.arl = arl
        self.threshold = math.log(arl)
        self._initialize()

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.stop = False
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_nll = []
        self.cum_scores = []

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
        return self.detector_score, self.detect

    def nll(self, sample, pre_pdf, post_pdf):
        """Calculate instant Negative Log Likelihood"""
        NLL_item = post_pdf(sample).log() - pre_pdf(sample).log()
        return NLL_item
