# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *
from delta.tables import *
import json, datetime, os, sys, pickle
import mlflow
import requests
import pandas as pd

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

app_insights_data = pickle.load(open("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/appInsightsRawData.pkl", "rb"))

# COMMAND ----------

def extractRequiredAppInsightsData (row):
  return [row[0], json.loads(row[4])["Workspace Name"], json.loads(row[4])["Service Name"], json.loads(row[4])["Container Id"], 
          json.loads(row[4])["Prediction"], json.loads(row[4])["Request Id"], json.loads(row[4])["Models"], json.loads(row[4])["Input"], row[-5]]

rows = [extractRequiredAppInsightsData(row) for row in app_insights_data]

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

# MAGIC %md
# MAGIC ## Azure Metrics

# COMMAND ----------

metrics_data_files = [spark.createDataFrame(json.loads('{ "data":['+pickle.load(open("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/"+filename,"rb"))[1:-1]+']}')['data']) for filename in os.listdir("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/") if "_raw_data" in filename]

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame
unionedDFs = reduce(DataFrame.unionAll, metrics_data_files)

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
