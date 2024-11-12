from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
import numpy as np

class SimpleLinearRegressionModel:
    def __init__(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    def predict(self, x):
        return np.dot(x, self.weights) + self.intercept


def train(data, iterations=100, step=0.01, regParam=0.0, regType=None):
    """
    Train a linear regression model using distributed SGD in PySpark.
    
    Parameters:
    - data (RDD[LabeledPoint]): Training data as an RDD of LabeledPoint objects.
    - iterations (int): Number of iterations for gradient descent.
    - step (float): Step size (learning rate) for gradient descent.
    - regParam (float): Regularization parameter.
    - regType (str): Type of regularization ('l1' or 'l2'). None means no regularization.
    
    Returns:
    - SimpleLinearRegressionModel: Trained model with weights and intercept.
    """
    num_features = len(data.first().features)
    weights = np.zeros(num_features)
    intercept = 0.0

    for i in range(iterations):
        # Calculate the gradient across each partition
        gradients = data.mapPartitions(lambda partition: compute_gradients(partition, weights, intercept)).reduce(add_gradients)

        # Update weights and intercept using the computed gradients
        weight_gradients, intercept_gradient = gradients
        if regType == 'l1':
            weight_gradients += regParam * np.sign(weights)
        elif regType == 'l2':
            weight_gradients += regParam * weights

        weights -= step * weight_gradients
        intercept -= step * intercept_gradient

    return SimpleLinearRegressionModel(weights, intercept)


def compute_gradients(partition, weights, intercept):
    weight_gradients = np.zeros_like(weights)
    intercept_gradient = 0.0
    count = 0

    for point in partition:
        prediction = np.dot(point.features, weights) + intercept
        error = prediction - point.label
        weight_gradients += error * point.features
        intercept_gradient += error
        count += 1

    # Average the gradients
    if count > 0:
        weight_gradients /= count
        intercept_gradient /= count

    yield weight_gradients, intercept_gradient


def add_gradients(g1, g2):
    weight_gradients1, intercept_gradient1 = g1
    weight_gradients2, intercept_gradient2 = g2
    return weight_gradients1 + weight_gradients2, intercept_gradient1 + intercept_gradient2


# Example usage
sc = SparkContext("local", "LinearRegressionWithSGD")

# Create some example training data
data = [
    LabeledPoint(2.0, [1.0, 2.0]),
    LabeledPoint(3.0, [2.0, 1.0]),
    LabeledPoint(4.0, [3.0, 2.0]),
    LabeledPoint(5.0, [4.0, 3.0])
]
rdd = sc.parallelize(data)

# Train the model
model = train(rdd, iterations=100, step=0.01, regParam=0.1, regType='l2')

# Print the model weights and intercept
print("Weights:", model.weights)
print("Intercept:", model.intercept)

# Predict on a new data point
print("Prediction for [3.0, 2.0]:", model.predict([3.0, 2.0]))

sc.stop()
