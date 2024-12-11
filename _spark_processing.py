from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("fMRI Data Processing") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.2") \
    .getOrCreate()

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "fmri-data") \
    .load()

# Parse the value column
parsed_df = df.selectExpr("CAST(value AS STRING)").alias("raw_data")

# Define processing logic
processed_df = parsed_df.withColumn("processed_field", col("value"))  # Example transformation

# Write to console or another sink
query = processed_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .start()

query.awaitTermination()
