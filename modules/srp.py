#Shiryaev–Roberts-Pollak (SRP) procedure family
#vanilla SR: with initial self.detector_score = 0
#SRP: with initial self.detector_score that is random
#SRP-r: with initial self.detector_score = r

class SRP:
    def __init__(self, hyper_threshold):
        super().__init__()
        self.hyper_threshold = hyper_threshold

    def _reset(self):
        self._initialize()

    def _initialize(self):
        self.stop = False
        self.stopping_time = 0
        self.detector_score = 0
        self.cum_lr = []
        self.cum_scores = []

    def _update(self, sample, null_model, alter_model):
        self.stopping_time += 1
        inst_sr = self.lr(sample, null_model.pdf, alter_model.pdf)
        self.detector_score = max(1,self.detector_score)*inst_sr.item()
        self.cum_lr.append(inst_sr.item())
        self.cum_scores.append(self.detector_score)
        if self.detector_score > self.hyper_threshold:
                self.stop = True
        return self.stop

    def lr(self, sample, null_pdf,  alter_pdf):
        """Calculate instant Hyvarinen Score Difference"""
        SR_item = alter_pdf(sample)/null_pdf(sample)
        return SR_item