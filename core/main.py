### IMPORTS
import pandas as pd
from abc import ABC, abstractmethod
###

class Model(ABC):
    def __init__(self, params):
        self.params = params

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass
