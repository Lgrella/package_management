from linear_regression.main import LinearRegression
from utils.main import SparkSessionBuilder
from core.point import LabeledPoint
import numpy as np
#from core.point import UnlabeledPoint

# Create a Spark session
spark = SparkSessionBuilder().get_session()

# Load your data
data = spark.read.csv('/workspaces/sparkit/data/diabetes.csv', header=True, inferSchema=True)

# Normalize the data
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="assembledFeatures")
scaler = StandardScaler(inputCol="assembledFeatures", outputCol="scaledFeatures", withMean=True, withStd=True)
pipeline = Pipeline(stages=[assembler, scaler])
data = pipeline.fit(data).transform(data)

# Convert to RDD of LabeledPoint objects, assuming the last column is the label
data = data.rdd.map(lambda row: LabeledPoint(row.scaledFeatures.values, row[-1]))

# Split the data into training and test sets
training, test = data.randomSplit([0.8, 0.2])

print("Training set size:", training.count())
# Initialize and train LinearRegression model
model = LinearRegression(w_shape=(len(training.first().data),), b_shape=(1,), batch_size=10)
model.train(training, num_epochs=25, lr=0.01)

# Print trained parameters
print("Trained weights:", model.params["W"])
print("Trained bias:", model.params["b"])

# Compare with scikit-learn
training_data = np.array(training.map(lambda p: p.data).collect())
training_labels = np.array(training.map(lambda p: p.label).collect())

from sklearn.linear_model import LinearRegression as SklearnLinearRegression

sklearn_model = SklearnLinearRegression()
sklearn_model.fit(training_data, training_labels)

print("Scikit-learn weights:", sklearn_model.coef_)
print("Scikit-learn bias:", sklearn_model.intercept_)

#check if my model matches (closely) to sklearn
assert np.allclose(model.params["W"], sklearn_model.coef_, atol=0.1)

# Stop Spark session
spark.stop()