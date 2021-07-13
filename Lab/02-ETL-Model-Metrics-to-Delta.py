# Databricks notebook source
# MAGIC %md
# MAGIC # Extract Model Lifecycle Information into Delta Lake

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from delta.tables import *
import json, datetime, os, sys, pickle
import mlflow
import requests
import pandas as pd

# COMMAND ----------

# MAGIC %run ../Lab/00-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFlow  
# MAGIC   
# MAGIC First, we will extract the data in the MLFlow Tracking Server to obtain the information surrounding the action experiments. In the cell below, we just provide the path to the MLFLow Experiment, derive the experiment ID, and then leverage the `mlflow-experiment` Reader in Databricks to extract the information for our sensor prediction experiment.  
# MAGIC   
# MAGIC The actual transformations applied to the DataFrame of extracted data can be applied to **any MLFLow experiment** and it will extract the metrics and parameters logged to the experiment. This is because the actual structure of the expected DataFrame is constant and we simply need to `map_concat()` and then `explode()` the metrics and paramters to get them pivoted into `key` and `value` columns. This also gets it into an ideal structure for querying and visualization in SQL Analytics.

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

refined_df.write.mode("overwrite").saveAsTable(f"{DB_NAME}.experiment_data_bronze")

# COMMAND ----------

display(refined_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure App Insights  
# MAGIC   
# MAGIC After pulling information about the experiment from MLFlow, we can extract information from Azure Application Insights. In Azure Application Insights, we can extract trace and response information. **Trace** information holds the data sent to the model to be scored, and the **Response** inforamtion holds the response / score that the model returned.  
# MAGIC   
# MAGIC For the lab, we will parse an example raw dump of this Application Insights data to get the **trace** and **response** data. To extract Application Insights data directly, check out the **AzureDataModelExtraction** notebook in the `/Examples/Azure/` folder of this [repository](https://github.com/mpfis/unified-ml-monitoring-on-databricks).

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

endpointName = "sensorpredictionbeta-service"
appInsightsDF = spark.createDataFrame(rows, ["timestamp", "workspaceName", "endpointName", "containerId", "response", "requestId", "model", "inputData", "mlWorkspace"])
appInsightsDF_Filtered = appInsightsDF.filter(appInsightsDF.endpointName.contains(endpointName))
appInsightsDF_Filtered = appInsightsDF_Filtered.withColumn("pathToExperiment", lit(PATH_TO_MLFLOW_EXPERIMENT)) \
                                                .withColumn("endpointName", lit(endpointName))

# COMMAND ----------

appInsightsDF_Filtered.write.saveAsTable(f"{DB_NAME}.response_data_bronze", format="delta", mode="overwrite")

# COMMAND ----------

display(appInsightsDF_Filtered.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC This data is still *pretty* messy. Let's **refine** this bronze-level data a bit more so that we have a silver level table.

# COMMAND ----------

bronze_data = spark.read.table(f"{DB_NAME}.response_data_bronze")

columns = bronze_data.columns

bronze_data = bronze_data.select(col("timestamp").cast("timestamp"), *columns[1:])

def str_to_json(t):
  return json.loads(t)

jsonify = udf(str_to_json, StringType())

spark.udf.register('jsonify', jsonify)

columns_copy = columns.copy()
columns_copy.remove("response")

schema = StructType([
  StructField("columns", ArrayType(StringType()), True),
  StructField("index", ArrayType(IntegerType()), True),
  StructField("data", ArrayType(ArrayType(FloatType())), True)
])

bronze_to_silver = bronze_data.select(regexp_replace("response", '\[|\]', "").cast("float").alias("response"), *columns_copy) \
                                    .withColumn("processedInput", from_json(jsonify(col("inputData")), schema)) \
                                    .withColumn("input", col("processedInput.data")) \
                                    .withColumn("extractedColumns", col("processedInput.columns")) \
                                    .select(explode("input").alias("inputPart"), "*").withColumn("mappedInput", map_from_arrays(col("extractedColumns"), col("inputPart"))) \
                                    .select("timestamp", "pathToExperiment","model", "requestId", "response", "mappedInput") \
                                    .groupBy(["timestamp", "pathToExperiment","model", "requestId", "response"]).agg(collect_list("mappedInput").alias('input')) \
                                    .withColumn("mappedInputandPrediction", struct(col("response"), col("input")))

# COMMAND ----------

bronze_to_silver.write.saveAsTable(f"{DB_NAME}.response_data_silver", format="delta", mode="overwrite")

# COMMAND ----------

display(bronze_to_silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure Metrics  
# MAGIC   
# MAGIC Another important set of data points to measure and monitor is the information coming from the underlying infrastructure supporting the model application itself. Azure Metrics can be queried to get information on the CPU Usage, Memory Usage, and Network Traffic of an endpoint that is supporting a model in Azure.  
# MAGIC   
# MAGIC For the lab, we will parse an example raw dump of this Azure Metrics data to get data. To extract Azure Metrics data directly, check out the **AzureDataModelExtraction** notebook in the `/Examples/Azure/` folder of this [repository](https://github.com/mpfis/unified-ml-monitoring-on-databricks).

# COMMAND ----------

metric_file_names = [name for name in os.listdir("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/") if "_raw_data" in name]
metrics_data_files = [json.loads('{ "data":['+pickle.load(open("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/"+filename,"rb"))[1:-1]+']}')['data'] for filename in os.listdir("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/") if "_raw_data" in filename]
for idx, metric_datafile in enumerate(metrics_data_files):
  for reading in metric_datafile:
    reading["metric"] = metric_file_names[idx].split("_raw_data.pkl")[0]
metrics_data_dfs = [spark.createDataFrame(metric_data_file) for metric_data_file in metrics_data_files]

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame
unionedDFs = reduce(DataFrame.unionAll, metrics_data_dfs)

# COMMAND ----------

unioned_metrics_for_model = unionedDFs.withColumn("date", col("timeStamp").cast("date")) \
                                          .withColumn("hour", hour("timeStamp")) \
                                          .groupBy("date", "hour", "metric").agg(avg("average").alias("value")) \
                                          .groupBy("date", "hour").pivot("metric").sum("value") \
                                          .withColumn("timeStamp", (unix_timestamp(col('date').cast("timestamp"))+(col("hour")*3600)).cast("timestamp")) \
                                          .withColumn("MemoryUsageMB", col("MemoryUsage")/1000000) \
                                          .withColumn("pathToExperiment", lit(PATH_TO_MLFLOW_EXPERIMENT))

# COMMAND ----------

unioned_metrics_for_model.write.saveAsTable(f"{DB_NAME}.endpoint_metrics_bronze", format="delta", mode="overwrite")

# COMMAND ----------

display(unioned_metrics_for_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step  
# MAGIC With the model trained, deployed and its operational / training metrics extracted we can leverage these data points to first establish a process for calculating data drift and build a unified monitoring solution.  
