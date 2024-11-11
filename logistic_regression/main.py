### IMPORTS
import pandas as pd
from core.main import SGDModel
###

class LogisticRegression(SGDModel):
    def __init__(self, params):
        super().__init__(params)
