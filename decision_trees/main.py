### IMPORTS
from core.main import Model
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark import RDD
from tqdm import tqdm

import pyspark
def row_to_tuple(data):
    return data.map(lambda row: tuple(row))

def to_rdd(data):
    spark = SparkSession.builder.appName("DecisionTreeTest").getOrCreate()
    spark_df = spark.createDataFrame(data)
    rdd = spark_df.rdd
    rdd.persist()

    return row_to_tuple(rdd)

class TreeNode():
    def __init__(self, feature, threshold, gini=None, left=None, right=None, flip=False) -> None:
        self.feature = feature
        self.threshold = threshold
        self.gini = gini  # gini from training data
        self.left = left  # left child
        self.right = right  # light child
        self.flip = False # whether to flip predictions

    def predict(self, row):
        if self.left is None and self.right is None:
            prediction = 1 if row[self.feature] > self.threshold else 0
            return int(not prediction) if self.flip else prediction
        
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
        if total == 0:
            return 0  # Avoid division by zero if the dataset is empty

        # Count the instances of each class (assumes labels are in the last column)
        label_counts = data.map(lambda x: x[-1]).countByValue()

        # Compute the Gini index
        gini = 1 - sum((count / total) ** 2 for count in label_counts.values())
        return gini

    def recursive_fit(self, data: RDD, features: list, depth: int):
        if depth > self.max_depth or len(features) == 0:
            return None  # Return None for leaf nodes
        
        best_gini = float('inf')
        best_node = None
        feature_progress = tqdm(total=len(features), desc=f"Depth {depth}", position=0, leave=True)

        for feature_idx, feature in enumerate(features):
            feature_values = sorted(data.map(lambda row: row[feature]).distinct().collect())

            skip = max(len(feature_values) // self.n_thresholds - 1, 1)

            threshold_progress = tqdm(
                total=len(range(0, len(feature_values), skip)),
                desc=f"Feature {feature_idx}",
                position=1,
                leave=False
            )

            for i in range(0, len(feature_values), skip):
                threshold = feature_values[i]
                left = data.filter(lambda row: row[feature] <= threshold)
                right = data.filter(lambda row: row[feature] > threshold)

                left_count = left.count()
                right_count = right.count()
                total = data.count()

                if left_count == 0 or right_count == 0:  # Skip if no split
                    threshold_progress.update(1)
                    continue

                gini_left = DecisionTree.gini(left)
                gini_right = DecisionTree.gini(right)
                gini_split = (left_count / total) * gini_left + (right_count / total) * gini_right

                #print(f"feature: {feature}, threshold: {threshold}, gini: {gini_split}")
                #breakpoint()
                if gini_split < best_gini:
                    #print(f"updating best node with gini of {gini_split} from {best_gini}")
                    best_gini = gini_split

                    best_node = TreeNode(feature=feature, threshold=threshold, gini=gini_split, left=None, right=None, flip=False)

                    correct = data.filter(lambda row: best_node.predict(row) == row[-1]).count()
                    split_accuracy = (correct) / total

                    #breakpoint()

                    if split_accuracy < 0.5:
                        best_node.flip = True
                    
                    threshold_progress.update(1)
            threshold_progress.close()
            feature_progress.update(1)

        leftover_features =  [f for f in features if f != best_node.feature]
        best_node.left = self.recursive_fit(left, leftover_features, depth + 1)
        best_node.right = self.recursive_fit(right, leftover_features, depth + 1)

        return best_node

    def train(self, data):
        if not isinstance(data, RDD):
            data = to_rdd(data)

        features = range(len(data.first()) - 1)  # Exclude the target variable column
        self.head = self.recursive_fit(data, features, depth=1)
        return self

    def predict(self, data):
        if not isinstance(data, RDD):
            data = to_rdd(data)
        
        return data.map(lambda row: self.head.predict(row))
    
    def loss(self):
        return 0
