### IMPORTS
import pandas as pd
import pyspark as ps
###

class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self, grad):
        for item in self.params:
            if item not in grad:
                raise Exception("Bad thing happened")
            self.params[item] -= self.lr*grad[item]
