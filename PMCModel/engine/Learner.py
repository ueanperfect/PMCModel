import logging
from .PMLogger import PMLogger
import torch
from ..model.classifiers.Classifier import BaseClassifier


class Learner():
    def __init__(self,
                 dataloader,
                 model: BaseClassifier,
                 optimizer,
                 logger: PMLogger,
                 device = 'cpu'
                 ):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
    def training(self):
        size = len(self.dataloader.dataset)
        current = 0
        self.model.train()
        for batch, (X, y) in enumerate(self.dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.model.loss(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, current = loss.item(), current+len(X)
            self.logger.print_training_information(loss,current,size)



