from .Evaluator import Evaluator
from .Learner import Learner
from .PMLogger import PMLogger


class Runner():
    def __init__(self,
                 evaluator: Evaluator,
                 learner: Learner,
                 logger: PMLogger,
                 max_epoch=12,
                 ):
        self.evaluator = evaluator
        self.learner = learner
        self.classifier = learner.model
        self.max_epoch = max_epoch
        self.logger = logger

    def run(self):
        for i in range(self.max_epoch):
            self.logger.show_progress(i)
            self.learner.training()
            self.evaluator.evaluating()
            if (i + 1) % 5 == 0:
                self.logger.save_checkpoint(self.classifier, i)
        return
