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

dbutils.widgets.text("PATH_TO_EXPERIMENT", "")

# COMMAND ----------

DB_NAME = "UMLWorkshop"
PATH_TO_EXPERIMENT = dbutils.widgets.get("PATH_TO_EXPERIMENT")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFlow  
# MAGIC   
# MAGIC First, we will extract the data in the MLFlow Tracking Server to obtain the information surrounding the action experiments. In the cell below, we just provide the path to the MLFLow Experiment, derive the experiment ID, and then leverage the `mlflow-experiment` Reader in Databricks to extract the information for our sensor prediction experiment.  
# MAGIC   
# MAGIC The actual transformations applied to the DataFrame of extracted data can be applied to **any MLFLow experiment** and it will extract the metrics and parameters logged to the experiment. This is because the actual structure of the expected DataFrame is constant and we simply need to `map_concat()` and then `explode()` the metrics and paramters to get them pivoted into `key` and `value` columns. This also gets it into an ideal structure for querying and visualization in SQL Analytics.

# COMMAND ----------

expId = mlflow.get_experiment_by_name(PATH_TO_EXPERIMENT).experiment_id

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
appInsightsDF_Filtered = appInsightsDF_Filtered.withColumn("pathToExperiment", lit(PATH_TO_EXPERIMENT)) \
                                                .withColumn("endpointName", lit(endpointName))

# COMMAND ----------

appInsightsDF_Filtered.write.saveAsTable(f"{DB_NAME}.response_data_bronze", format="delta", mode="overwrite")

# COMMAND ----------

display(appInsightsDF_Filtered.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Azure Metrics  
# MAGIC   
# MAGIC Another important set of data points to measure and monitor is the information coming from the underlying infrastructure supporting the model application itself. Azure Metrics can be queried to get information on the CPU Usage, Memory Usage, and Network Traffic of an endpoint that is supporting a model in Azure.  
# MAGIC   
# MAGIC For the lab, we will parse an example raw dump of this Azure Metrics data to get data. To extract Azure Metrics data directly, check out the **AzureDataModelExtraction** notebook in the `/Examples/Azure/` folder of this [repository](https://github.com/mpfis/unified-ml-monitoring-on-databricks).

# COMMAND ----------

metrics_data_files = [spark.createDataFrame(json.loads('{ "data":['+pickle.load(open("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/"+filename,"rb"))[1:-1]+']}')['data']) for filename in os.listdir("/databricks/driver/unified-ml-monitoring-on-databricks/Datasets/") if "_raw_data" in filename]

# COMMAND ----------

from functools import reduce
from pyspark.sql import DataFrame
unionedDFs = reduce(DataFrame.unionAll, metrics_data_files)

# COMMAND ----------

display(unionedDFs)

# COMMAND ----------

unioned_metrics_for_model = unionedDFs.withColumn("date", col("timeStamp").cast("date")) \
                                          .withColumn("hour", hour("timeStamp")) \
                                          .groupBy("date", "hour", "metric").agg(avg("average").alias("value")) \
                                          .groupBy("date", "hour").pivot("metric").sum("value") \
                                          .withColumn("timeStamp", (unix_timestamp(col('date').cast("timestamp"))+(col("hour")*3600)).cast("timestamp")) \
                                          .withColumn("MemoryUsageMB", col("MemoryUsage")/1000000) \
                                          .withColumn("pathToExperiment", lit(pathToExperiment)) \
                                          .withColumn("run_id", lit(run_id)) \
                                          .withColumn("endpointName", lit(endpointName)) \
                                          .withColumn("deploymentTarget", lit(deployment_target))

# COMMAND ----------

unioned_metrics_for_model.write.saveAsTable(f"{DB_NAME}.endpoint_metrics_bronze", format="delta", mode="overwrite")

# COMMAND ----------

display(unioned_metrics_for_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Step  
# MAGIC With the model trained, deployed and its operational / training metrics extracted we can leverage these data points to first establish a process for calculating data drift and build a unified monitoring solution.  
