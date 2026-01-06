import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import functions as F

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "INPUT_DELTA_PATH",
    "CURATED_DELTA_PATH",
    "QUARANTINE_PATH",
    "DQ_REPORT_PATH",
])

INPUT_DELTA_PATH = args["INPUT_DELTA_PATH"].rstrip("/")
CURATED_DELTA_PATH = args["CURATED_DELTA_PATH"].rstrip("/")
QUARANTINE_PATH = args["QUARANTINE_PATH"].rstrip("/")
DQ_REPORT_PATH = args["DQ_REPORT_PATH"].rstrip("/")

# ------------------------------------------------------------
# Glue / Spark
# ------------------------------------------------------------
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

print("=== Day 8 Quality Gates ===")
print("INPUT_DELTA_PATH:", INPUT_DELTA_PATH)
print("CURATED_DELTA_PATH:", CURATED_DELTA_PATH)
print("QUARANTINE_PATH:", QUARANTINE_PATH)
print("DQ_REPORT_PATH:", DQ_REPORT_PATH)
print("spark.version:", spark.version)

# ------------------------------------------------------------
# Read validated Delta
# ------------------------------------------------------------
df = spark.read.format("delta").load(INPUT_DELTA_PATH)

total_rows = df.count()
print("Total input rows:", total_rows)
print("Columns:", df.columns)

# ------------------------------------------------------------
# Helper: detect pickup/dropoff datetime column names
# (NYC Taxi usually: tpep_pickup_datetime / tpep_dropoff_datetime)
# ------------------------------------------------------------
pickup_col = None
dropoff_col = None
for c in ["tpep_pickup_datetime", "pickup_datetime", "lpep_pickup_datetime"]:
    if c in df.columns:
        pickup_col = c
        break
for c in ["tpep_dropoff_datetime", "dropoff_datetime", "lpep_dropoff_datetime"]:
    if c in df.columns:
        dropoff_col = c
        break

# ------------------------------------------------------------
# Define Quality Rules (practical, not too strict)
# - Required IDs present
# - Non-negative numeric measures
# - Pickup <= Dropoff (if both exist)
# - Passenger count reasonable
# - Location IDs present (if columns exist)
# ------------------------------------------------------------
rules = []

# Always enforce: VendorID if present
if "VendorID" in df.columns:
    rules.append(("vendor_not_null", F.col("VendorID").isNotNull()))

# Datetime checks (only if columns exist)
if pickup_col:
    rules.append(("pickup_not_null", F.col(pickup_col).isNotNull()))
if dropoff_col:
    rules.append(("dropoff_not_null", F.col(dropoff_col).isNotNull()))
if pickup_col and dropoff_col:
    rules.append(("pickup_before_dropoff",
                  F.to_timestamp(F.col(pickup_col)) <= F.to_timestamp(F.col(dropoff_col))))

# Location checks
if "PULocationID" in df.columns:
    rules.append(("pu_location_not_null", F.col("PULocationID").isNotNull()))
if "DOLocationID" in df.columns:
    rules.append(("do_location_not_null", F.col("DOLocationID").isNotNull()))

# Non-negative numeric measures (apply only if columns exist)
nonneg_cols = ["trip_distance", "fare_amount", "total_amount", "tip_amount", "tolls_amount", "extra"]
for c in nonneg_cols:
    if c in df.columns:
        rules.append((f"{c}_non_negative", F.col(c).isNull() | (F.col(c) >= F.lit(0))))

# Passenger count sanity (0..8 is common)
if "passenger_count" in df.columns:
    rules.append(("passenger_count_range",
                  F.col("passenger_count").isNull() | ((F.col("passenger_count") >= 0) & (F.col("passenger_count") <= 8))))

# Optional: ratecode sanity (1..6 common)
if "RatecodeID" in df.columns:
    rules.append(("ratecode_range",
                  F.col("RatecodeID").isNull() | ((F.col("RatecodeID") >= 1) & (F.col("RatecodeID") <= 6))))

# ------------------------------------------------------------
# Build rule flags + overall pass/fail
# ------------------------------------------------------------
df_q = df
for name, cond in rules:
    df_q = df_q.withColumn(name, F.when(cond, F.lit(True)).otherwise(F.lit(False)))

# Pass = all rule flags true
rule_cols = [name for name, _ in rules]
if rule_cols:
    pass_expr = F.lit(True)
    for c in rule_cols:
        pass_expr = pass_expr & F.col(c)
    df_q = df_q.withColumn("dq_pass", pass_expr)
else:
    df_q = df_q.withColumn("dq_pass", F.lit(True))

passed = df_q.filter(F.col("dq_pass") == True).drop("dq_pass")
failed = df_q.filter(F.col("dq_pass") == False)

passed_count = passed.count()
failed_count = failed.count()

print("Passed rows:", passed_count)
print("Failed rows:", failed_count)

# ------------------------------------------------------------
# Write outputs
# 1) Curated Delta (passed only)
# 2) Quarantine Parquet (failed + rule flags)
# ------------------------------------------------------------
passed.write.format("delta").mode("overwrite").save(CURATED_DELTA_PATH)

# Keep rule columns in quarantine for steward review
failed.write.mode("overwrite").format("parquet").save(QUARANTINE_PATH)

print("✅ Curated Delta written:", CURATED_DELTA_PATH)
print("✅ Quarantine written:", QUARANTINE_PATH)

# ------------------------------------------------------------
# DQ Summary Report (JSON) - single file
# ------------------------------------------------------------
# rule failure counts
rule_fail_exprs = []
for c in rule_cols:
    rule_fail_exprs.append(F.sum(F.when(F.col(c) == False, 1).otherwise(0)).alias(f"{c}_fail_count"))

summary = (
    df_q.agg(
        F.count("*").alias("total_rows"),
        F.sum(F.when(F.col("dq_pass") == True, 1).otherwise(0)).alias("passed_rows"),
        F.sum(F.when(F.col("dq_pass") == False, 1).otherwise(0)).alias("failed_rows"),
        *rule_fail_exprs
    )
    .withColumn("input_path", F.lit(INPUT_DELTA_PATH))
    .withColumn("curated_delta_path", F.lit(CURATED_DELTA_PATH))
    .withColumn("quarantine_path", F.lit(QUARANTINE_PATH))
    .withColumn("job_name", F.lit(args["JOB_NAME"]))
    .withColumn("generated_at_utc", F.current_timestamp())
)

summary.coalesce(1).write.mode("overwrite").json(DQ_REPORT_PATH)
print("✅ DQ report written:", DQ_REPORT_PATH)
