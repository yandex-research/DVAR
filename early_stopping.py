import numpy as np


class ClipEarlyStopper:
    """
    Args:
        eps: minimum change in the monitored metric to qualify as an improvement
        patience: number of checks with no improvement
    """

    def __init__(self, eps: float = 5e-2, patience: int = 10):
        self.eps = eps
        self.patience = patience
        self.stopped = False
        self.best_value = None
        self.no_improvement = 0

    def __call__(self, metric: float):
        if not self.stopped:
            if self.best_value is not None:
                if metric > self.best_value + self.eps:
                    self.best_value = metric
                    self.no_improvement = 0
                else:
                    self.no_improvement += 1
                    if self.no_improvement > self.patience:
                        self.stopped = True
            else:
                self.best_value = metric
        return self.stopped


class VarEarlyStopper:
    """
    Args:
        eps: variance threshold
        window: window size
    """

    def __init__(self, eps: float = 0.15, window: int = 200):
        self.eps = eps
        self.window = window
        self.stopped = False
        self.history = np.array([])
        self.normalized_var = 1

    def __call__(self, loss: float):
        self.history = np.append(self.history, loss)
        if len(self.history) >= self.window:
            self.normalized_var = np.var(self.history[-self.window:]) / np.var(self.history)
        if self.normalized_var < self.eps:
            self.stopped = True

        return self.normalized_var
