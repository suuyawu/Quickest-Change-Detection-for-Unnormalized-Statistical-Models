#CUSUM from Fisher Divergence

class CUSUMF:
    def __init__(self, hyper_lambda, hyper_threshold):
        super().__init__()
        self.hyper_lambda = hyper_lambda
        self.hyper_threshold = hyper_threshold
        self._initialize()

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.stop = False
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_hst = []
        self.cum_scores = []

    def _update(self, sample, null_model, alter_model):
        self.stopping_time += 1
        inst_hst = self.hst(sample, null_model.hscore, alter_model.hscore)
        self.detector_score = max((self.detector_score+inst_hst).item(), 0)
        self.cum_hst.append(inst_hst.item())
        self.cum_scores.append(self.detector_score)
        if self.detector_score > self.hyper_threshold:
                self.stop = True
        return self.stop

    def hst(self, sample, null_hscore,  alter_hscore):
        """Calculate instant Hyvarinen Score Difference"""
        Hscore_item = -alter_hscore(sample) + null_hscore(sample)
        return self.hyper_lambda*Hscore_item