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
                raise Exception("Developer did something wrong, you need a timeout")
            self.params[item] -= self.lr*grad[item]
