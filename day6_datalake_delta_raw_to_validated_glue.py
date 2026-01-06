import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DoubleType

# ------------------------------------------------------------
# Required job arguments (match what you're passing in CLI)
# ------------------------------------------------------------
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "RAW_TRIPS_PATH",
    "RAW_ZONES_PATH",
    "DELTA_OUT_PATH"
])

RAW_TRIPS_PATH = args["RAW_TRIPS_PATH"]
RAW_ZONES_PATH = args["RAW_ZONES_PATH"]
DELTA_OUT_PATH = args["DELTA_OUT_PATH"]

# ------------------------------------------------------------
# Glue / Spark session
# ------------------------------------------------------------
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ------------------------------------------------------------
# Read raw trips (parquet folder)
# ------------------------------------------------------------
trips = spark.read.parquet(RAW_TRIPS_PATH)

# Detect pickup/dropoff columns safely
pickup_candidates = ["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"]
dropoff_candidates = ["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"]

def first_existing(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

pickup_col = first_existing(pickup_candidates, trips.columns)
dropoff_col = first_existing(dropoff_candidates, trips.columns)

if not pickup_col or not dropoff_col:
    raise ValueError(f"Pickup/Dropoff columns not found. Columns={trips.columns}")

df = (
    trips
    .withColumn("pickup_ts", F.col(pickup_col).cast(TimestampType()))
    .withColumn("dropoff_ts", F.col(dropoff_col).cast(TimestampType()))
    .withColumn("pickup_date", F.to_date("pickup_ts"))
)

# ------------------------------------------------------------
# Minimal validation gates
# ------------------------------------------------------------
df = df.filter(F.col("pickup_ts").isNotNull() & F.col("dropoff_ts").isNotNull())
df = df.filter(F.col("dropoff_ts") >= F.col("pickup_ts"))

if "fare_amount" in df.columns:
    df = df.filter(F.col("fare_amount").cast(DoubleType()) > F.lit(0.0))

# ------------------------------------------------------------
# Optional: read zones (keeps your RAW_ZONES_PATH argument meaningful)
# ------------------------------------------------------------
# If this path is a folder with CSVs, this works. If not needed, you can delete this block.
try:
    zones = (
        spark.read.option("header", "true").csv(RAW_ZONES_PATH)
    )
    # Example: just count to validate it’s reachable
    zones_count = zones.count()
    print(f"Zones lookup loaded successfully. Row count = {zones_count}")
except Exception as e:
    print(f"WARNING: Could not read RAW_ZONES_PATH={RAW_ZONES_PATH}. Continuing. Error={e}")

# ------------------------------------------------------------
# Governance metadata
# ------------------------------------------------------------
df = (
    df
    .withColumn("governance_zone", F.lit("validated"))
    .withColumn("ingested_at", F.current_timestamp())
    .withColumn("source_system", F.lit("NYC_TLC"))
)

# ------------------------------------------------------------
# Write Delta partitioned
# ------------------------------------------------------------
(
    df.write.format("delta")
      .mode("overwrite")
      .partitionBy("pickup_date")
      .save(DELTA_OUT_PATH)
)

job.commit()
print(f"SUCCESS: Wrote validated Delta to: {DELTA_OUT_PATH}")
