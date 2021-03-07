import matplotlib.pyplot as plt
import numpy as np
import itertools
import operator

class Logger:
    def __init__(self) -> None:
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def log(self, train_loss, val_loss, train_acc, val_acc):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

    def check_early_stop(self, patience):
        if patience > 1:
            if len(self.val_loss) < patience: return False
            return all(itertools.starmap(operator.ge, zip(self.val_loss[-patience:], self.val_loss[-patience+1:])))
        return False

