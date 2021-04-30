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
from pyspark.sql.types import *
from pyspark.sql.functions import *
import requests
import json
import logging
import datetime
import os
import sys
import pandas as pd
from delta.tables import *

# COMMAND ----------

PATH_TO_MLFLOW_EXPERIMENT = "ENTER PATH TO MLFLOW EXPERIMENT"

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFlow

# COMMAND ----------

expId = mlflow.get_experiment_by_name(PATH_TO_MLFLOW_EXPERIMENT).experiment_id

df = spark.read.format("mlflow-experiment").load(expId)

refined_df = df.select(col('run_id'), col("experiment_id"), explode(map_concat(col("metrics"), col("params"))), col('start_time'), col("end_time")) \
                .filter("key != 'model'") \
                .select("run_id", "experiment_id", "key", col("value").cast("float"), col('start_time'), col("end_time")) \
                .groupBy("run_id", "experiment_id", "start_time", "end_time") \
                .pivot("key") \
                .sum("value") \
                .withColumn("trainingDuration", col("end_time").cast("integer")-col("start_time").cast("integer")) # example of added column

# COMMAND ----------

refined_df.write.saveAsTable(f"{DB_NAME}.experiment_data_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure App Insights

# COMMAND ----------

# MAGIC %sh
# MAGIC git clone https://github.com/mpfis/unified-ml-monitoring-on-databricks.git

# COMMAND ----------

import pickle
app_insights = pickle.load(open("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/appInsightsRawData.pkl"))

# COMMAND ----------

## Load from PKL file

# COMMAND ----------

def extractRequiredAppInsightsData (row):
  return [row[0], json.loads(row[4])["Workspace Name"], json.loads(row[4])["Service Name"], json.loads(row[4])["Container Id"], 
          json.loads(row[4])["Prediction"], json.loads(row[4])["Request Id"], json.loads(row[4])["Models"], json.loads(row[4])["Input"], row[-5]]

columns = response_sample['Columns']
rows = [extractRequiredAppInsightsData(row) for row in response_sample['Rows']]

responseSchema = StructType([
  StructField("ResponseValue", StringType(), True), 
  StructField("ContainerId", StringType(), True)
])

# COMMAND ----------

appInsightsDF = spark.createDataFrame(rows, ["timestamp", "workspaceName", "endpointName", "containerId", "response", "requestId", "model", "inputData", "mlWorkspace"])
appInsightsDF_Filtered = appInsightsDF.filter(appInsightsDF.endpointName.contains(endpointName))
appInsightsDF_Filtered = appInsightsDF_Filtered.withColumn("pathToExperiment", lit(pathToExperiment)) \
                                                .withColumn("run_id", lit(run_id)) \
                                                .withColumn("endpointName", lit(endpointName)) \
                                                .withColumn("deploymentTarget", lit(deployment_target))

# COMMAND ----------

appInsightsDF_Filtered.write.saveAsTable(f"{DB_NAME}.response_data_bronze", format="delta", mode="overwrite")

# COMMAND ----------

## merge
existingTable = DeltaTable.forName(spark, f"{DB_NAME}.response_data_bronze")
existingTable.alias("s").merge(
  appInsightsDF_Filtered.alias("t"),
  "s.requestID = t.requestID") \
.whenNotMatchedInsertAll() \
.execute()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure Metrics

# COMMAND ----------

## Load data from PKL file

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame
unionedDFs = reduce(DataFrame.unionAll, metricDfs)

# COMMAND ----------

unioned_metrics_for_model = unionedDFs.withColumn("date", col("timeStamp").cast("date")) \
                                          .withColumn("hour", hour("timeStamp")) \
                                          .groupBy("endpoint_name", "date", "hour", "metric").agg(avg("average").alias("value")) \
                                          .groupBy("endpoint_name", "date", "hour").pivot("metric").sum("value") \
                                          .withColumn("timeStamp", (unix_timestamp(col('date').cast("timestamp"))+(col("hour")*3600)).cast("timestamp")) \
                                          .withColumn("MemoryUsageMB", col("MemoryUsage")/1000000) \
                                          .withColumn("pathToExperiment", lit(pathToExperiment)) \
                                          .withColumn("run_id", lit(run_id)) \
                                          .withColumn("endpointName", lit(endpointName)) \
                                          .withColumn("deploymentTarget", lit(deployment_target))

# COMMAND ----------

unioned_metrics_for_model.write.saveAsTable(f"{DB_NAME}.endpoint_metrics_bronze", format="delta", mode="overwrite")

# COMMAND ----------

## merge
existingTable = DeltaTable.forName(spark, f"{DB_NAME}.endpoint_metrics_bronze")
existingTable.alias("s").merge(
  unioned_metrics_for_model.alias("t"),
  "s.timeStamp = t.timeStamp") \
.whenNotMatchedInsertAll() \
.execute()
