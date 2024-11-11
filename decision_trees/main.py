### IMPORTS
import pandas as pd
from core.main import Model
###

def greater_than(x, y):
    return x > y

def less_than(x, y):
    return x < y

class TreeNode():
    def __init__(self, feature, threshold, operator=greater_than) -> None:
        self.feature = feature
        self.threshold = threshold
        self.operator = operator

    def predict(self, X):
        predictions = self.operator(X[self.feature], self.threshold)
        return predictions


class DecisionTree(Model):
    def __init__(self, params):
        super().__init__(params)

    def fit():
        pass

    def predict():
        pass
