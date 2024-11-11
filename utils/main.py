from pyspark.sql import SparkSession

class SparkSessionBuilder:
    def __init__(self, app_name="Spark Application", master="local[*]"):
        self.app_name = app_name
        self.master = master

    def get_session(self):
        return SparkSession.builder \
            .appName(self.app_name) \
            .master(self.master) \
            .getOrCreate()