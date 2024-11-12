### IMPORTS
import pyspark.pandas as pd
from abc import ABC, abstractmethod
###

class Model(ABC):
    def __init__(self, context):
        self.sc = context
        # Initialize parameters
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass
    
    @abstractmethod
    def loss(self, data, targets):
        pass

class SGDModel(Model):
    @abstractmethod
    def grad(self, data, targets):
        pass
