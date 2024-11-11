### IMPORTS
import pandas as pd
import numpy as np
from core.main import SGDModel
###

class LinearRegression(SGDModel):
    def __init__(self, params):
        super().__init__(params)

    def loss(self, data, targets):
        preds = self.predict(data)
        return np.mean(np.square(preds-targets), axis=1)
    
    def predict(self, data):
        W = self.params["W"]
        b = self.params["b"]
        return np.matmul(W, data) + b
