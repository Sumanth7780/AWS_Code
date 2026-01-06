import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F

args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "VALIDATED_DELTA_PATH",
    "RAW_ZONES_CSV_PATH",
    "CURATED_DELTA_PATH"
])

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

VALIDATED = args["VALIDATED_DELTA_PATH"]
ZONES = args["RAW_ZONES_CSV_PATH"]
CURATED = args["CURATED_DELTA_PATH"]

trips = spark.read.format("delta").load(VALIDATED)

zones = (
    spark.read.option("header", True).csv(ZONES)
    .withColumnRenamed("LocationID", "location_id")
    .withColumnRenamed("Zone", "zone_name")
    .withColumnRenamed("Borough", "borough")
    .withColumnRenamed("service_zone", "service_zone")
)

# Derived feature
trips = trips.withColumn(
    "trip_duration_min",
    (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long")) / 60.0
)

# Join pickup zone enrichment if column exists
if "PULocationID" in trips.columns:
    zones_pu = zones.select(
        F.col("location_id").alias("pu_location_id"),
        F.col("zone_name").alias("pu_zone"),
        F.col("borough").alias("pu_borough"),
        F.col("service_zone").alias("pu_service_zone"),
    )
    trips = trips.join(zones_pu, trips["PULocationID"] == zones_pu["pu_location_id"], "left")

# Join dropoff zone enrichment if column exists
if "DOLocationID" in trips.columns:
    zones_do = zones.select(
        F.col("location_id").alias("do_location_id"),
        F.col("zone_name").alias("do_zone"),
        F.col("borough").alias("do_borough"),
        F.col("service_zone").alias("do_service_zone"),
    )
    trips = trips.join(zones_do, trips["DOLocationID"] == zones_do["do_location_id"], "left")

curated = (
    trips
    .withColumn("governance_zone", F.lit("curated"))
    .withColumn("curated_at", F.current_timestamp())
)

(curated.write.format("delta")
 .mode("overwrite")
 .partitionBy("pickup_date")
 .save(CURATED))

job.commit()
print(f"✅ Day 7 done: {CURATED}")
