### IMPORTS
import pyspark.pandas as pd
import numpy as np
from pyspark import RDD
from core.main import SGDModel
from core.point import Point
from typing import Union
###

class LinearRegression(SGDModel):
    def __init__(self, w_shape, b_shape):
        self.params = {"W": np.zeros(w_shape), "b": np.zeros(b_shape)}

    def loss(self, data: Union[Point, RDD[Point]]):
        if isinstance(data, RDD):
            return data.map(self.grad)
        preds = self.predict(data)
        return 0.5*(preds-data.labels)**2

    def grad(self, data: Union[Point, RDD[Point]]):
        if isinstance(data, RDD):
            return data.map(self.grad)
        preds = self.predict(data)
        return np.multiply(preds-data.labels, data.data)
    
    def predict(self, data: Union[Point, RDD[Point]]):
        W = self.params["W"]
        b = self.params["b"]
        if isinstance(data, RDD):
            return data.map(self.predict)
        else:
            return pd.dot(W, data.data) + b
