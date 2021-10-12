# Databricks notebook source
# MAGIC %md
# MAGIC # Calculating Data Drift

# COMMAND ----------

# MAGIC %run ../Lab/00-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Cleaning up the Trace Data from Azure Application Insights  
# MAGIC   
# MAGIC The data that is extracted from Azure Application Insights is very rough and needs to be cleaned before we can actually work with the data. In this next cell, we extract the data we need from the Application Insights datasets and parse the information from the trace portion of the message.  
# MAGIC 
# MAGIC This will give us a cleaned, tabularized form of the trace data which we can then use to calculate data drift.

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.types._
# MAGIC import org.apache.spark.sql.functions._
# MAGIC import org.apache.spark.sql.DataFrame
# MAGIC 
# MAGIC def jsonToDataFrame(json: String, schema: StructType = null): DataFrame = {
# MAGIC   // SparkSessions are available with Spark 2.0+
# MAGIC   val reader = spark.read
# MAGIC   Option(schema).foreach(reader.schema)
# MAGIC   reader.json(sc.parallelize(Array(json)))
# MAGIC }
# MAGIC 
# MAGIC val responseDataDF = spark.read.table(s"$DB_NAME.response_data_bronze")
# MAGIC val json = responseDataDF.select("inputData").collect()(0)(0).toString.replace("\\", "")
# MAGIC val json_adj = json.substring(1, json.length()-1)
# MAGIC val schema_of_json = jsonToDataFrame(json_adj).schema
# MAGIC val schematized_trace_data = responseDataDF.select($"timestamp", $"endpointName", $"inputData")
# MAGIC                                            .withColumn("inputData_clean", from_json(rtrim(ltrim(regexp_replace($"inputData", "\\\\", ""), "\""), "\""), schema_of_json))
# MAGIC                                            .select($"timestamp", $"endpointName", $"inputData_clean".alias("inputData"))
# MAGIC                                            .select($"timestamp", $"endpointName", map_from_arrays($"inputData.columns", $"inputData.data".getItem(0).alias("data")).alias("mapped_data"))
# MAGIC 
# MAGIC val keysDF = schematized_trace_data.select(explode(map_keys($"mapped_data")).alias("mapped_data")).distinct()
# MAGIC val keys = keysDF.collect().map(f=>f.get(0))
# MAGIC val keyCols = keys.map(f=> col(s"mapped_data.$f"))
# MAGIC val expanded_df = schematized_trace_data.select($"*" +: keyCols:_*).drop($"mapped_data").withColumn("date", $"timestamp".cast("date")).withColumn("hour", hour($"timestamp"))
# MAGIC expanded_df.createOrReplaceTempView("parsedTraceData")

# COMMAND ----------

# MAGIC %md
# MAGIC Here is what the data looked like **before**

# COMMAND ----------

# MAGIC %scala
# MAGIC display(responseDataDF.select($"inputData").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC Here is what the data looks like now:

# COMMAND ----------

# MAGIC %scala
# MAGIC display(expanded_df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Calculate Data Drift  
# MAGIC   
# MAGIC Understanding data drift is key to understanding when it is time to retrain your model. When you train a model, you are training it on a sample of data. While these training datasets are usually quite large, they don't represent changes that may happend to the data in the future. For instance, if you are training a predictive maintenance model that leverages sensors (like the model we are using today), new environmental factors could appear in the data coming into the model to be scored that the model does not know how to properly score.  
# MAGIC   
# MAGIC Monitoring for this drift is important so that you can retrain and refresh the model to allow for the model to adapt.  
# MAGIC   
# MAGIC The short example of this that we are showing today uses the [Hellingers Distance](https://en.wikipedia.org/wiki/Hellinger_distance) to compare the distribution of the training dataset with the incoming data that is being scored by the model.

# COMMAND ----------

import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import *

## Training Dataset
sensorTrainingDataDF = spark.read.table(f"{DB_NAME}.sensor")
## Trace Data / Data Sent to the Model to be Scored
sensorTraceDataDF = spark.read.table('parsedTraceData')

# Function for calulcating hellinger distance
def hellingerDist(p, q):
  _SQRT2 = np.sqrt(2)
  return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def calculateDriftBySensor (sensor, training_data, trace_data):
  train_arr = np.array([row[0] for row in training_data.select(f"{sensor}").collect()])
  trace_arr = np.array([row[0] for row in trace_data.select(f"{sensor}").collect()])
  train_arr /= np.sum(train_arr)
  trace_arr /= np.sum(trace_arr)
  dist = hellingerDist(trace_arr, train_arr[:2776])
  return dist

# COMMAND ----------

# MAGIC %md
# MAGIC Here we apply the calculation to all four sensors.

# COMMAND ----------

sensors_drift_calculations = list(map(lambda x: float(calculateDriftBySensor(x, sensorTrainingDataDF, sensorTraceDataDF)), 
                                      ["Sensor1", "Sensor2", "Sensor3", "Sensor4"]))

# COMMAND ----------

# MAGIC %md
# MAGIC With the data drift metrics calculated, we can write them to a Delta Lake Table to be then used by our monitoring appliance or dashboard to check and see if this new drift calculation warrants a retrainign of the model.

# COMMAND ----------

schema = StructType([
  StructField("Sensor1", FloatType(), True),
  StructField("Sensor2", FloatType(), True),
  StructField("Sensor3", FloatType(), True),
  StructField("Sensor4", FloatType(), True)
])

data_drift = spark.createDataFrame([sensors_drift_calculations], schema).withColumn("_ts", current_timestamp())

data_drift.write.saveAsTable(f"{DB_NAME}.sensor_data_drift", mode="append", format="delta")

# COMMAND ----------

display(spark.table(f"{DB_NAME}.sensor_data_drift"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps  
# MAGIC   
# MAGIC **For those with Databricks SQL**: You can follow along and create some of the assets that we will walkthrough in Databricks SQL.  
# MAGIC   
# MAGIC **For those without Databricks SQL**: You can import these tables into the BI tool of your choice (i.e. PowerBI, Tableau), but if you are interested in doing this in Databricks SQL contact your Databricks account team and get Databricks SQL enabled for your workspace. We have provided a guide for implementing everything we are doing in Databricks SQL so you can come back to this and test it out on your own.
