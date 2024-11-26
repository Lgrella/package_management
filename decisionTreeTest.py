import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pyspark import SparkContext
from pyspark.sql import SparkSession
from decision_trees.main import DecisionTree
from sklearn.model_selection import train_test_split

# spark = SparkSession.builder.appName("DecisionTreeTest").getOrCreate()
# sc = SparkContext.getOrCreate()

max_depth = 1
n_thresholds = 5

# TODO: convert diabetes data to X and y
df = pd.read_csv('data/diabetes.csv') 
#print(df.head)
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

df_train = X_train.copy()
df_train["Outcome"] = y_train

#Sklearn model + predictions
skModel = DecisionTreeClassifier(max_depth=max_depth)
skModel.fit(X_train, y_train)
skPredictions = skModel.predict(X_test)

#Spark Tree Model
params = {'max_depth': max_depth, 'n_thresholds': n_thresholds}  # Define additional parameters if needed
sparkItModel = DecisionTree(params)
sparkItModel.train(df_train)
print("Training worked!")
sparkItPredictions = sparkItModel.predict(X_test).collect()
sparkItPredictionsArray = np.array(sparkItPredictions)

#Accuracy Comparisons
sparkItAccuracy = accuracy_score(y_test, sparkItPredictions)
skAccuracy = accuracy_score(y_test, skPredictions)

#print(sparkItPredictions)
print(f"SparkIt Model Accuracy: {sparkItAccuracy}")
print(f"Sklearn Model Accuracy: {skAccuracy}")

#breakpoint()
