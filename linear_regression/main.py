### IMPORTS
#import pandas as pd
import numpy as np
from pyspark import RDD
from core.main import SGDModel
from core.sgd import SGD
from core.point import Point
from typing import Union
###
class LinearRegression(SGDModel):
    def __init__(self, w_shape, b_shape, batch_size=32):
        self.params = {"W": [0.0] * w_shape[0], "b": 0.0}  # Initialize b as a scalar
        self.batch_size = batch_size

    def loss(self, data: Union[Point, RDD[Point]]):
        if isinstance(data, RDD):
            count = data.count()
            batches = data.randomSplit([self.batch_size] * (count // self.batch_size))
            losses = [batch.map(self.grad).sum() / batch.count() for batch in batches]
            return sum(losses) / len(losses)
        preds = self.predict(data)
        return 0.5 * (preds - data.labels) ** 2

    def predict(self, data):
        W = self.params["W"]
        b = self.params["b"]
        if isinstance(data, RDD):
            return data.map(lambda point: sum(w * x for w, x in zip(W, point.data)) + b)
        else:
            return sum(w * x for w, x in zip(W, data.data)) + b


    def grad(self, point):
        W = self.params["W"]
        b = self.params["b"]
        prediction = self.predict(point)
        diff = prediction - point.label
        grad_W = [diff * x for x in point.data]
        grad_b = diff
        return grad_W, grad_b

    def train(self, data, num_epochs, lr):
        for epoch in range(num_epochs):
            data = data.sample(False, 1.0).cache()  # Shuffle the data
            num_batches = data.count() // self.batch_size

            # Split the data into batches
            batches = data.randomSplit([1.0 / num_batches] * num_batches)

            for i in range(num_batches):
                batch = batches[i]  # Access the i-th batch

                # Aggregate gradients across the batch
                grads = batch.map(self.grad).reduce(lambda g1, g2: (
                    [g1_w + g2_w for g1_w, g2_w in zip(g1[0], g2[0])],
                    g1[1] + g2[1]
                ))

                grad_W, grad_b = grads
                self.params["W"] = [w - lr * g / self.batch_size for w, g in zip(self.params["W"], grad_W)]
                self.params["b"] -= lr * grad_b / self.batch_size

        print("Training complete.")
