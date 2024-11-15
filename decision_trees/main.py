### IMPORTS
from core.main import Model
from pyspark.sql import SparkSession
from pyspark import SparkContext
###

#CONFUSED: 

#REVIEW: Check over entire thing and make sure there's no overlapping spark contexts

def gini(data):
    
    #TODO: Need to rewrite this since data is now an rdd
    # print(data.take(5))
    
    # target = data.map(lambda row: row[-1])
    total = data.count()
    counts = data.countByValue()
    
    # sum of p squared
    p_squareds = sum((count / total) ** 2 for count in counts.values())
    return 1 - p_squareds

# add y as the last column of X
def add_column(X, y):
    return X.zip(y).map(lambda x: x[0] + (x[1],))

def remove_last_column(X):
    return X.map(lambda x: x[:-1])

class TreeNode():
    def __init__(self, feature, threshold, gini=None, left_datapoints=None, right_datapoints=None, left=None, right=None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.gini = gini #gini from training data

        # IMPORTANT DISTINCTION: left and right datapoints are from the data it was trained on. not used during prediction
        self.left_datapoints = left_datapoints
        self.right_datapoints = right_datapoints

        # left and right children
        self.left = left
        self.right = right

    # predict on if the feature is greater than the threshold
    def predict(self, X):
        predictions = X.map(lambda x: x[self.feature] > self.threshold)
        return predictions
    
    # split the data based on threshold, then pass datapoints to children
    def recursive_pred(self, X):
        my_preds = self.predict(X)
        if self.left is None and self.right is None:
            return my_preds
        
        data_with_preds = add_column(X, my_preds)

        left_datapoints = data_with_preds.filter(lambda x: not x[-1]) # if predicted 0
        right_datapoints = data_with_preds.filter(lambda x: x[-1]) # predicted 1

        left_preds = self.left.recursive_pred(left_datapoints)
        right_preds = self.right.recursive_pred(right_datapoints)

        # total dataset is predicted from left and right childrens' predictions
        total_preds = left_preds.union(right_preds)
        return total_preds

class DecisionTree(Model):
    def __init__(self, params):
        super().__init__(params)
        self.head = None
        self.max_depth = params["max_depth"]

    @staticmethod
    def optimal_node_for_feature(data, feature):
        feature_values = data.map(lambda row: row[feature]).distinct().collect()
        best_node = None
        best_gini = 1

        # loop over every possible threshold to get the threshold with the minimum gini index
        for threshold in feature_values:
            node = TreeNode(feature, threshold)
            preds = node.predict(data)
            
            print("Preds:", preds.take(5))
            
            giniVal = gini(preds)
            node.gini = giniVal
            print(node.gini)

            if giniVal < best_gini:
                best_gini = giniVal
                best_node = node

            return best_node
    
    @staticmethod
    def optimal_node(data, features):
        
        #TODO: Persist the data here
        
        best_gini = 1
        best_node = None
        
        if not features:
            return None
        
        print("features:", features)
        
        # results = features.map(lambda x: DecisionTree.optimal_node_for_feature(data, x)).collect()
        # best_node = min(results, key=lambda node: node.gini)
        # best_gini = best_node.gini
        
        #compute the best node for every feature, select the best overall
        for feature in features:
            node = DecisionTree.optimal_node_for_feature(data, feature)
            if node.gini < best_gini:
                best_gini = node.gini
                best_node = node

        preds = best_node.predict(data)
        data_and_preds = add_column(data, preds)

        # keep track of the datapoints that get split left and right based on the best split    
        best_node.left_datapoints = remove_last_column(data_and_preds.filter(lambda x: not x[-1])) #get rid of preds column before saving
        best_node.right_datapoints = remove_last_column(data_and_preds.filter(lambda x: x[-1]))

        return best_node
    
    # recursively build out the tree from a set of features
    def recursive_fit(self, data, features, depth):
        if depth > self.max_depth:
            return None
        
        if not features:
            return None
        
        # calculate the optimal split based on remaining features
        curr = DecisionTree.optimal_node(data, features)
        
        if not curr:
          return None 

        #filter(lambda x: x != curr.feature, features) 
        leftover_features =  [feature for feature in features if feature != curr.feature]
        #remove feature we just used from pool

        # recursively build left and right children
        curr.left = self.recursive_fit(curr.left_datapoints, leftover_features, depth + 1)
        curr.right = self.recursive_fit(curr.right_datapoints, leftover_features, depth + 1)

        return curr

    def train(self, data):
        
        
        #TODO: Make sure data is an RDD + persist!
        spark = SparkSession.builder.appName("DecisionTreeTest").getOrCreate()
        spark_df = spark.createDataFrame(data)
        rdd = spark_df.rdd
        rdd.persist()
        
        print(rdd.take(5))
        
        
        #TODO: Convert features to RDD.
        #CONFUSED: Not rlly sure how this works in the rest of the code
        # features = [field.name for field in spark_df.schema[:-1]]
        # features = spark.sparkContext.parallelize(features)
        
        features = range(len(rdd.first()) - 1)
        

        self.head = self.recursive_fit(rdd, features, depth=0)
        
        return self.head

    def predict(self, data):
        
        # #TODO: Make sure data is an RDD
        # sc4 = SparkContext("Converting prediction data to an RDD")
        # parallelizedData = sc4.parallelize(data)
        # sc4.stop()
        
        return self.head.recursive_pred(data)
    
    def loss(self):
        return 0
