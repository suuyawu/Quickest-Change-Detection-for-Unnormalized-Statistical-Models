#Page's CUSUM rule

class CUSUM:
    def __init__(self, hyper_threshold):
        super().__init__()
        self.hyper_threshold = hyper_threshold
        self._initialize()

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.stop = False
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_nll = []
        self.cum_scores = []

    def _update(self, sample, null_model, alter_model):
        self.stopping_time += 1
        inst_nll = self.nll(sample, null_model.pdf, alter_model.pdf)
        self.detector_score = max((self.detector_score+inst_nll).item(), 0)
        self.cum_nll.append(inst_nll.item())
        self.cum_scores.append(self.detector_score)
        if self.detector_score > self.hyper_threshold:
                self.stop = True
        return self.stop
    
    def nll(self, sample, null_pdf, alter_pdf):
        """Calculate instant Negative Log Likelihood"""
        NLL_item = alter_pdf(sample).log() - null_pdf(sample).log()
        return NLL_item