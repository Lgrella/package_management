### IMPORTS
import pyspark.pandas as pd
import numpy as np
from pyspark import RDD
from core.main import SGDModel
from core.sgd import SGD
from core.point import Point
from typing import Union
###

class LinearRegression(SGDModel):
    def __init__(self, w_shape, b_shape):
        self.params = {"W": np.zeros(w_shape), "b": np.zeros(b_shape)}

    def loss(self, data: Union[Point, RDD[Point]]):
        if isinstance(data, RDD):
            count = data.count()
            return data.map(self.grad).sum() / count
        preds = self.predict(data)
        return 0.5*(preds-data.labels)**2

    def grad(self, data: Union[Point, RDD[Point]]):
        if isinstance(data, RDD):
            count = data.count()
            grads = data.map(self.grad).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
            return grads[0]/count, grads[1]/count
        preds = self.predict(data)
        diff = preds-data.labels
        return np.multiply(diff, data.data), diff
    
    def predict(self, data: Union[Point, RDD[Point]]):
        W = self.params["W"]
        b = self.params["b"]
        if isinstance(data, RDD):
            return data.map(self.predict)
        else:
            return np.matmul(W, data.data) + b
        
    def train(self, data: RDD[RDD[Point]], num_epochs, lr):
        sgd = SGD(self.params, lr)
        for _ in range(num_epochs):
            grad = data.map(self.grad).reduce(lambda x, y: (x[0]+y[0], x[1]+y[1]))
            sgd.step({"W": grad[0], "b": grad[1]})
