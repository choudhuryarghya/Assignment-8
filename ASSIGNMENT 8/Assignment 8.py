# Step 1: Set Up the Environment
# First, let's set up our Databricks environment and load the data from the Kaggle dataset.

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Create Spark session
spark = SparkSession.builder \
    .appName("NYCTaxiAnalysis_Kaggle") \
    .getOrCreate()

# Set configuration for optimized performance
spark.conf.set("spark.sql.shuffle.partitions", "8")
# Step 2: Load Data from Kaggle Dataset
# Assuming you've uploaded the dataset to DBFS or mounted it to your Databricks workspace:


# File location and type
file_location = "/FileStore/tables/revenue_for_cab_drivers.csv"  # Update with your actual path
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# Load the data
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

# Display dataframe schema
df.printSchema()

# Show sample data
display(df.limit(5))
# Step 3: Data Preprocessing
# Based on the Kaggle dataset, let's clean and prepare the data:


# Convert timestamp columns to proper format (if available)
if "pickup_datetime" in df.columns:
    df = df.withColumn("pickup_datetime", to_timestamp(col("pickup_datetime")))
if "dropoff_datetime" in df.columns:
    df = df.withColumn("dropoff_datetime", to_timestamp(col("dropoff_datetime")))

# Handle missing values
columns_to_fill = ["passenger_count", "trip_distance", "fare_amount", "extra", 
                  "mta_tax", "tip_amount", "tolls_amount", "total_amount"]
for col_name in columns_to_fill:
    if col_name in df.columns:
        df = df.na.fill(0, [col_name])

# Filter out invalid records
df = df.filter((col("passenger_count") > 0) & 
               (col("trip_distance") > 0) & 
               (col("fare_amount") > 0))

# Standardize column names if needed
df = df.withColumnRenamed("VendorID", "vendor_id") \
       .withColumnRenamed("PULocationID", "pu_location_id") \
       .withColumnRenamed("DOLocationID", "do_location_id")
# Step 4: Query Solutions (Adapted for Kaggle Dataset)
# Query 1: Add Revenue Column

# Add Revenue column as sum of specified columns
revenue_columns = ["fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount", "total_amount"]
existing_columns = [col for col in revenue_columns if col in df.columns]

df = df.withColumn("Revenue", sum(col(c) for c in existing_columns))

# Show the result
display(df.select(*existing_columns, "Revenue").limit(5))
# Query 2: Increasing Count of Total Passengers by Area
# python
# Group by pickup location
if "pu_location_id" in df.columns:
    passenger_count_by_area = df.groupBy("pu_location_id") \
        .agg(sum("passenger_count").alias("total_passengers")) \
        .orderBy("total_passengers", ascending=False)
    display(passenger_count_by_area)
else:
    print("Pickup location ID column not found in dataset")
# Query 3: Realtime Average Fare/Total Earning by Vendors

if "vendor_id" in df.columns:
    vendor_earnings = df.groupBy("vendor_id") \
        .agg(avg("fare_amount").alias("avg_fare"),
             sum("Revenue").alias("total_earning")) \
        .orderBy("total_earning", ascending=False)
    display(vendor_earnings)
else:
    print("Vendor ID column not found in dataset")
# Query 4: Moving Count of Payments by Payment Mode

if "payment_type" in df.columns and "pickup_datetime" in df.columns:
    windowSpec = Window.partitionBy("payment_type").orderBy("pickup_datetime")
    payment_counts = df.withColumn("moving_count", 
                                count("*").over(windowSpec.rowsBetween(-1000, 0)))
    display(payment_counts.select("payment_type", "pickup_datetime", "moving_count") \
           .orderBy("payment_type", "pickup_datetime"))
else:
    print("Required columns (payment_type or pickup_datetime) not found in dataset")
# Query 5: Highest Two Gaining Vendors on a Particular Date

from datetime import datetime

if "vendor_id" in df.columns and "pickup_datetime" in df.columns:
    specific_date = datetime(2020, 1, 15)  # Adjust based on your data
    vendor_performance = df.filter(date_trunc("day", col("pickup_datetime")) == specific_date) \
        .groupBy("vendor_id") \
        .agg(sum("Revenue").alias("total_revenue"),
             sum("passenger_count").alias("total_passengers"),
             sum("trip_distance").alias("total_distance")) \
        .orderBy("total_revenue", ascending=False) \
        .limit(2)
    display(vendor_performance)
else:
    print("Required columns (vendor_id or pickup_datetime) not found in dataset")
# Query 6: Most Passengers Between Route Locations

if "pu_location_id" in df.columns and "do_location_id" in df.columns:
    route_passengers = df.groupBy("pu_location_id", "do_location_id") \
        .agg(sum("passenger_count").alias("total_passengers")) \
        .orderBy("total_passengers", ascending=False) \
        .limit(10)
    display(route_passengers)
else:
    print("Pickup or dropoff location columns not found in dataset")
# Query 7: Top Pickup Locations with Most Passengers in Last 5/10 Seconds

if "pu_location_id" in df.columns and "pickup_datetime" in df.columns:
    windowDuration = "10 seconds"
    pickup_trends = df.groupBy(
        window(col("pickup_datetime"), windowDuration),
        col("pu_location_id")
    ).agg(
        sum("passenger_count").alias("passengers_last_10_sec")
    ).orderBy(
        col("window.start").desc(),
        col("passengers_last_10_sec").desc()
    )
    display(pickup_trends.limit(20))
else:
    print("Required columns (pu_location_id or pickup_datetime) not found in dataset")
# Step 5: Save Flattened Data as Parquet Table

# Save the processed DataFrame as a Parquet table
parquet_path = "/FileStore/tables/revenue_cab_drivers_processed.parquet"
df.write.mode("overwrite").parquet(parquet_path)

# Create external table
spark.sql("""
CREATE TABLE IF NOT EXISTS revenue_cab_drivers_external
USING PARQUET
LOCATION '{}'
""".format(parquet_path))

# Verify the table
display(spark.sql("SELECT * FROM revenue_cab_drivers_external LIMIT 5"))
Additional Analysis (Specific to Kaggle Dataset)
Since this is a revenue-focused dataset, let's add some additional revenue-specific analysis:

# 1. Revenue distribution by hour of day
if "pickup_datetime" in df.columns:
    revenue_by_hour = df.withColumn("pickup_hour", hour(col("pickup_datetime"))) \
        .groupBy("pickup_hour") \
        .agg(sum("Revenue").alias("total_revenue")) \
        .orderBy("pickup_hour")
    display(revenue_by_hour)

# 2. Average revenue per passenger by vendor
if "vendor_id" in df.columns:
    revenue_per_passenger = df.groupBy("vendor_id") \
        .agg((sum("Revenue")/sum("passenger_count")).alias("avg_revenue_per_passenger")) \
        .orderBy("avg_revenue_per_passenger", ascending=False)
    display(revenue_per_passenger)

# 3. Top 10 most profitable trips (revenue per mile)
if "trip_distance" in df.columns:
    profitable_trips = df.withColumn("revenue_per_mile", col("Revenue")/col("trip_distance")) \
        .orderBy("revenue_per_mile", ascending=False) \
        .limit(10)
    display(profitable_trips.select("vendor_id", "pu_location_id", "do_location_id", 
                                  "trip_distance", "Revenue", "revenue_per_mile"))
