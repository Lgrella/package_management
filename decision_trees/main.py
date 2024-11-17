### IMPORTS
from core.main import Model
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import RDD

import pyspark
def row_to_tuple(data):
    return data.map(lambda row: tuple(row))

def to_rdd(data):
    spark = SparkSession.builder.appName("DecisionTreeTest").getOrCreate()
    spark_df = spark.createDataFrame(data)
    rdd = spark_df.rdd
    #rdd.persist()

    return row_to_tuple(rdd)

class TreeNode():
    def __init__(self, feature, threshold, gini=None, left=None, right=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.gini = gini  # gini from training data
        self.left = left  # Left child (another TreeNode)
        self.right = right  # Right child (another TreeNode)

    def predict(self, row):
        if self.left is None and self.right is None:
            return 1 if row[self.feature] > self.threshold else 0
        
        if row[self.feature] <= self.threshold:
            return self.left.predict(row)
        
        else:
            return self.right.predict(row)

class DecisionTree(Model):
    def __init__(self, params):
        super().__init__(params)
        self.head = None
        self.max_depth = params["max_depth"] if "max_depth" in params else 1
        self.n_thresholds = params["n_thresholds"] if "n_thresholds" in params else 2

    @staticmethod
    def gini(data: RDD):
        total = data.count()
        gini = 0

        for category in [0, 1]:  # Assuming binary classification
            split = data.filter(lambda x: x[-1] == category)
            size = split.count()

            if size == 0:  # avoid division by zero
                continue
            p_squareds = sum((count / size) ** 2 for _, count in split.countByValue().items())
            gini += (1 - p_squareds) * (size / total)

        return gini

    def recursive_fit(self, data: RDD, features: list, depth: int):
        if depth >= self.max_depth or len(features) == 0:
            return None  # Return None for leaf nodes
        
        best_gini = float('inf')
        best_node = None

        for feature in features:
            feature_values = sorted(data.map(lambda row: row[feature]).distinct().collect())
            #print("###", feature, len(feature_values), "###")

            skip = max(len(feature_values) // self.n_thresholds - 1, 1)

            for i in range(skip, len(feature_values), skip):
                threshold = feature_values[i]
                #print(i, threshold)
                left = data.filter(lambda row: row[feature] <= threshold)
                right = data.filter(lambda row: row[feature] > threshold)

                if left.count() == 0 or right.count() == 0:  # Skip if no split
                    continue

                gini_left = DecisionTree.gini(left)
                gini_right = DecisionTree.gini(right)
                gini_split = (left.count() / data.count()) * gini_left + (right.count() / data.count()) * gini_right

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_node = TreeNode(feature=feature, threshold=threshold, gini=gini_split, left=None, right=None)
                    
        leftover_features =  [f for f in features if f != best_node.feature]
        best_node.left = self.recursive_fit(left, leftover_features, depth + 1)
        best_node.right = self.recursive_fit(right, leftover_features, depth + 1)

        return best_node

    def train(self, data):
        if not isinstance(data, RDD):
            data = to_rdd(data)

        features = range(len(data.first()) - 1)  # Exclude the target variable column
        self.head = self.recursive_fit(data, features, depth=0)
        return self

    def predict(self, data):
        if not isinstance(data, RDD):
            data = to_rdd(data)
        
        return data.map(lambda row: self.head.predict(row))
    
    def loss(self):
        return 0
