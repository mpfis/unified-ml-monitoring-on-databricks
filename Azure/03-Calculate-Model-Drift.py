# Databricks notebook source
# MAGIC %md
# MAGIC ## Input Data Deviation

# COMMAND ----------

dbutils.widgets.text("DB_NAME", "")

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
# MAGIC def deviationFromAverage(avg_col:org.apache.spark.sql.Column, avg_val:Double) : org.apache.spark.sql.Column = {
# MAGIC   abs(avg_col-avg_val)
# MAGIC }
# MAGIC 
# MAGIC val DB_NAME = dbutils.widgets.get("DB_NAME")
# MAGIC val responseDataDF = spark.read.table(s"$DB_NAME.response_data_bronze")
# MAGIC val json = responseDataDF.select("inputData").collect()(0)(0).toString.replace("\\", "")
# MAGIC val json_adj = json.substring(1, json.length()-1)
# MAGIC val schema_of_json = jsonToDataFrame(json_adj).schema

# COMMAND ----------

# MAGIC %scala
# MAGIC val schematized_trace_data = responseDataDF.select($"timestamp", $"endpointName", $"inputData")
# MAGIC                                            .withColumn("inputData_clean", from_json(rtrim(ltrim(regexp_replace($"inputData", "\\\\", ""), "\""), "\""), schema_of_json))
# MAGIC                                            .select($"timestamp", $"endpointName", $"inputData_clean".alias("inputData"))
# MAGIC                                            .select($"timestamp", $"endpointName", map_from_arrays($"inputData.columns", $"inputData.data".getItem(0).alias("data")).alias("mapped_data"))

# COMMAND ----------

# MAGIC %scala
# MAGIC val keysDF = schematized_trace_data.select(explode(map_keys($"mapped_data")).alias("mapped_data")).distinct()
# MAGIC val keys = keysDF.collect().map(f=>f.get(0))
# MAGIC val keyCols = keys.map(f=> col(s"mapped_data.$f"))
# MAGIC val expanded_df = schematized_trace_data.select($"*" +: keyCols:_*).drop($"mapped_data").withColumn("date", $"timestamp".cast("date")).withColumn("hour", hour($"timestamp"))

# COMMAND ----------

# MAGIC %scala
# MAGIC val stddev_expanded_df = expanded_df.groupBy($"date", $"hour", $"endpointName")
# MAGIC                                             .agg(stddev_samp($"Sensor1").alias("Sensor1_STDDEV"), 
# MAGIC                                                  stddev_samp($"Sensor2").alias("Sensor2_STDDEV"), 
# MAGIC                                                  stddev_samp($"Sensor3").alias("Sensor3_STDDEV"),
# MAGIC                                                  stddev_samp($"Sensor4").alias("Sensor4_STDDEV"),
# MAGIC                                                  avg($"Sensor1").alias("Sensor1_avg"), 
# MAGIC                                                  avg($"Sensor2").alias("Sensor2_avg"), 
# MAGIC                                                  avg($"Sensor3").alias("Sensor3_avg"),
# MAGIC                                                  avg($"Sensor4").alias("Sensor4_avg"))
# MAGIC                                             .withColumn("sensor1_upper_bound", col("Sensor1_STDDEV")*2+col("Sensor1_avg"))
# MAGIC                                             .withColumn("sensor1_lower_bound", col("Sensor1_avg")-col("Sensor1_STDDEV")*2)
# MAGIC                                             .withColumn("sensor2_upper_bound", col("Sensor2_STDDEV")*2+col("Sensor2_avg"))
# MAGIC                                             .withColumn("sensor2_lower_bound", col("Sensor2_avg")-col("Sensor2_STDDEV")*2)
# MAGIC                                             .withColumn("sensor3_upper_bound", col("Sensor3_STDDEV")*2+col("Sensor3_avg"))
# MAGIC                                             .withColumn("sensor3_lower_bound", col("Sensor3_avg")-col("Sensor3_STDDEV")*2)
# MAGIC                                             .withColumn("sensor4_upper_bound", col("Sensor4_STDDEV")*2+col("Sensor4_avg"))
# MAGIC                                             .withColumn("sensor4_lower_bound", col("Sensor4_avg")-col("Sensor4_STDDEV")*2)
# MAGIC 
# MAGIC 
# MAGIC val joined_df = expanded_df.drop("endpointName").join(stddev_expanded_df, (stddev_expanded_df("date")===expanded_df("date")) && (stddev_expanded_df("hour") === expanded_df("hour")), "left")
# MAGIC                               .withColumn("sensor1_dev_check", when((col("sensor1_lower_bound") < col("Sensor1")) && (col("Sensor1") <= col("sensor1_upper_bound")), lit(0)).otherwise(lit(1)))
# MAGIC                               .withColumn("sensor2_dev_check", when((col("sensor2_lower_bound") < col("Sensor2")) && (col("Sensor2") <= col("sensor2_upper_bound")), lit(0)).otherwise(lit(1)))
# MAGIC                               .withColumn("sensor3_dev_check", when((col("sensor3_lower_bound") < col("Sensor3")) && (col("Sensor3") <= col("sensor3_upper_bound")), lit(0)).otherwise(lit(1)))
# MAGIC                               .withColumn("sensor4_dev_check", when((col("sensor4_lower_bound") < col("Sensor4")) && (col("Sensor4") <= col("sensor4_upper_bound")), lit(0)).otherwise(lit(1)))
# MAGIC                               .drop("hour").drop("date")
# MAGIC 
# MAGIC joined_df.write.mode("overwrite").format("delta").saveAsTable(s"$DB_NAME.input_data_analysis")
