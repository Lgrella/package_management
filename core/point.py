### IMPORTS
from abc import ABC, abstractmethod
###

class Point(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def data(self):
        pass
    
    @property
    @abstractmethod
    def labels(self):
        pass

class LabeledPoint(Point):
    def __init__(self, data, label) -> None:
        self.data = data
        self.label = label

class UnlabeledPoint(Point):
    def __init__(self, data) -> None:
        self.data = data
        self.label = None
