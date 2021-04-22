# Databricks notebook source
# Extract MLFlow
# Extract App Insights
# Extract Azure Metrics

## Student Version -- pull prepared datasets from GitHub and then transform / load to Delta for Azure Metrics / App Insights
## Instructor Version -- extract directly from sources

## Both would still pull from MLFlow

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from delta.tables import *
import mlflow
import json

# COMMAND ----------

# Check to see if the table exists
tableExists = spark._jsparkSession.catalog().tableExists(DB_NAME, f'experiment_data_bronze')
expId = mlflow.get_experiment_by_name(EXP_PATH).experiment_id
if tableExists:
  # if the table exists, get the latest experiment data based on the last experiment written to the target table. 
  latestExpermentEndTime = spark.read.table(f"{DB_NAME}.experiment_data_bronze") \
                                      .select("end_time").agg(max("end_time")).collect()[0][0]
  # use the latest experiment time to filter the experiment data so that only the newest, unwritten experiment data is used
  df = spark.read.format("mlflow-experiment").load(expId).filter(f"end_time > '{latestExpermentEndTime}'")
  no_data = df.count() == 0
else:
  # load the experiment data
  df = spark.read.format("mlflow-experiment").load(expId)
  no_data = df.count() == 0
  
  
if not no_data:
  # take the experiment data and denormalize the metrics and parameters recorded for the experiment
  refined_df = df.select(col('run_id'), col("experiment_id"), explode(map_concat(col("metrics"), col("params"))), col('start_time'), col("end_time")) \
                  .filter("key != 'model'") \
                  .select("run_id", "experiment_id", "key", col("value").cast("float"), col('start_time'), col("end_time")) \
                  .groupBy("run_id", "experiment_id", "start_time", "end_time") \
                  .pivot("key") \
                  .sum("value") \
                  .withColumn("trainingDuration", col("end_time").cast("integer")-col("start_time").cast("integer")) # example of added column
else:
  dbutils.notebook.exit(json.dumps({
    "status":205,
    "message":"No Data Found In Given Period."
  }))


