import torch
from ..model.classifiers.Classifier import BaseClassifier
from .PMLogger import PMLogger
from tqdm import tqdm
class Evaluator:
    def __init__(self,
                 dataloader,
                 model: BaseClassifier,
                 logger: PMLogger,
                 device='cpu'):
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.logger = logger

    def evaluating(self):
        size = len(self.dataloader.dataset)
        num_batches = len(self.dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.model.loss(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        self.logger.print_evaluate_information(test_loss, correct)
