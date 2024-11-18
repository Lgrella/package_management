from linear_regression.main import LinearRegression
from utils.main import SparkSessionBuilder
from core.point import LabeledPoint
import numpy as np
from core.preprocess import preprocess_data
#from core.point import UnlabeledPoint

# Create a Spark session
spark = SparkSessionBuilder().get_session()

# Load your data
data = spark.read.csv('/workspaces/sparkit/data/diabetes.csv', header=True, inferSchema=True)

data_rdd = preprocess_data(data)

training_rdd, test_rdd = data_rdd.randomSplit([0.8, 0.2])

# Initialize and train LinearRegression model
model = LinearRegression(w_shape=(len(training_rdd.first().data),), b_shape=(1,), batch_size=10)
model.train(training_rdd, num_epochs=5, lr=0.01)


print("Trained weights:", model.params["W"])
print("Trained bias:", model.params["b"])

# Compare with scikit-learn
training_data = np.array(training_rdd.map(lambda p: p.data).collect())
training_labels = np.array(training_rdd.map(lambda p: p.label).collect())

from sklearn.linear_model import LinearRegression as SklearnLinearRegression

sklearn_model = SklearnLinearRegression()
sklearn_model.fit(training_data, training_labels)

print("Scikit-learn weights:", sklearn_model.coef_)
print("Scikit-learn bias:", sklearn_model.intercept_)

#check if my model matches (closely) to sklearn (print the difference)
print("Difference in weights:", model.params["W"] - sklearn_model.coef_)
print("Difference in bias:", model.params["b"] - sklearn_model.intercept_)

# Stop Spark session
spark.stop()