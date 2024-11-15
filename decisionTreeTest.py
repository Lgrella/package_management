import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pyspark import SparkContext
from pyspark.sql import SparkSession
from decision_trees.main import DecisionTree

# spark = SparkSession.builder.appName("DecisionTreeTest").getOrCreate()
# sc = SparkContext.getOrCreate()

max_depth = 3

# TODO: convert diabetes data to X and y
df = pd.read_csv('/Users/pranavireddi/sparkit/data/diabetes.csv') 
print(df.head)
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

#Sklearn model + predictions
skModel = DecisionTreeClassifier(max_depth=max_depth)
skModel.fit(X, y)
skPredictions = skModel.predict(X)


#Spark Tree Model
params = {'max_depth': max_depth}  # Define additional parameters if needed
sparkItModel = DecisionTree(params)
sparkItModel.train(X)
sparkItPredictions = sparkItModel.predict(X).collect()
sparkItPredictionsArray = np.array(sparkItPredictions)

#Accuracy Comparisons
sparkItAccuracy = accuracy_score(y, sparkItPredictions)
skAccuracy = accuracy_score(y, skPredictions)

print(f"SparkIt Model Accuracy: {sparkItAccuracy}")
print(f"Sklearn Model Accuracy: {skAccuracy}")
