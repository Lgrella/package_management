### IMPORTS
import pandas as pd
###

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self, grad):
        self.params -= self.lr*grad
