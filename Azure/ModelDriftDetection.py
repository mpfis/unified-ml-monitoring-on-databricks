# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
import json

# COMMAND ----------

response_df = spark.read.table("sensor_prediction_model.response_data_bronze")

columns_copy = response_df.columns.copy()
columns_copy.remove("response")

# COMMAND ----------

def str_to_json(t):
  return json.loads(t)

jsonify = udf(str_to_json, StringType())

spark.udf.register('jsonify', jsonify)

# COMMAND ----------

schema = StructType([
  StructField("columns", ArrayType(StringType()), True),
  StructField("index", ArrayType(IntegerType()), True),
  StructField("data", ArrayType(ArrayType(FloatType())), True)
])

# COMMAND ----------

processed_response_df = response_df.select(regexp_replace("response", '(\[|\])', "").cast("float").alias("response"), *columns_copy) \
                                    .withColumn("processedInput", from_json(jsonify(col("inputData")), schema)) \
                                    .withColumn("input", col("processedInput.data")) \
                                    .withColumn("extractedColumns", col("processedInput.columns")) \
                                    .select(explode("input").alias("inputPart"), "*").withColumn("mappedInput", map_from_arrays(col("extractedColumns"), col("inputPart"))) \
                                    .select("timestamp", "pathToExperiment","model", "run_id", "requestId", "response", "mappedInput") \
                                    .groupBy(["timestamp", "pathToExperiment","model", "run_id", "requestId", "response"]).agg(collect_list("mappedInput").alias('input')) \
                                    .withColumn("mappedInputandPrediction", struct(col("response"), col("input")))

# COMMAND ----------

processed_response_df.write.mode("overwrite").format("delta").saveAsTable("sensor_prediction_model.response_data_silver")
