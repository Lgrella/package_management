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

class LabeledPoint:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

class UnlabeledPoint(Point):
    def __init__(self, data) -> None:
        self.data = data
    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return None
