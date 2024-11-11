### IMPORTS
import pandas as pd
from core.main import Model
###

class LogisticRegression(Model):
    def __init__(self, params):
        super().__init__(params)
