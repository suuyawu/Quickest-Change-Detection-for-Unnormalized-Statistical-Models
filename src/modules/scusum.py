import math


class SCUSUM:
    def __init__(self, arl):
        super().__init__()
        self.arl = arl
        self.hyper_threshold = math.log(arl)
        self._initialize()

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.stop = False
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_hst = []
        self.cum_scores = []

    def _update(self, sample, pre_model, post_model):
        self.stopping_time += 1
        inst_hst = self.hst(sample, pre_model.hscore, post_model.hscore)
        self.detector_score = max((self.detector_score + inst_hst).item(), 0)
        self.cum_hst.append(inst_hst.item())
        self.cum_scores.append(self.detector_score)
        if self.detector_score > self.hyper_threshold:
            self.detect = True
        else:
            self.detect = False
        return self.detector_score, self.detect

    def hst(self, sample, pre_hscore, post_hscore):
        """Calculate instant Hyvarinen Score Difference"""
        Hscore_item = -post_hscore(sample) + pre_hscore(sample)
        return Hscore_item
