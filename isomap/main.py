import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark import RDD

class SparkIsomap:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, rdd: RDD):
        """
        Compute the principal components and explained variance from the RDD.
        """
        stats = rdd.map(lambda x: (np.array(x), np.array(x) ** 2)) \
                   .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                   
        #NOTE: Stadardizing the input data
        n_samples = rdd.count()
        self.mean = stats[0] / n_samples
        self.variance = (stats[1] / n_samples - self.mean) ** 2
        self.scale = np.sqrt(self.variance)

        standardized_rdd = rdd.map(lambda x: (np.array(x) - self.mean) / self.scale)
        standardized_data = np.array(standardized_rdd.collect())

        #NOTE: Doing SVD
        u, s, vt = np.linalg.svd(standardized_data, full_matrices=False)

        sorted_indices = np.argsort(s)[::-1]
        s = s[sorted_indices]
        vt = vt[sorted_indices]

        self.components = vt[:self.n_components]
        return self

    def transform(self, rdd: RDD):
        """
        Project the data onto the principal components.
        """
        standardized_rdd = rdd.map(lambda x: (np.array(x) - self.mean) / self.scale)
        projected_rdd = standardized_rdd.map(lambda x: np.dot(x, self.components.T))
        return projected_rdd

