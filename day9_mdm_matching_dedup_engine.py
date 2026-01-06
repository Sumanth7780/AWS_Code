import sys
import re
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql import Window

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "RAW_ZONES_PATH",
    "MASTER_OUT_PATH",
    "STEWARD_QUEUE_PATH",
    "REJECTS_PATH",
    "AUDIT_PATH",
    "HIGH_CONF",
    "MED_CONF",
])

RAW_ZONES_PATH = args["RAW_ZONES_PATH"].rstrip("/")
MASTER_OUT_PATH = args["MASTER_OUT_PATH"].rstrip("/")
STEWARD_QUEUE_PATH = args["STEWARD_QUEUE_PATH"].rstrip("/")
REJECTS_PATH = args["REJECTS_PATH"].rstrip("/")
AUDIT_PATH = args["AUDIT_PATH"].rstrip("/")
HIGH_CONF = float(args["HIGH_CONF"])
MED_CONF = float(args["MED_CONF"])

# ------------------------------------------------------------
# Spark
# ------------------------------------------------------------
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

print("=== Day 9 MDM Matching & Dedup ===")
print("RAW_ZONES_PATH:", RAW_ZONES_PATH)
print("MASTER_OUT_PATH:", MASTER_OUT_PATH)
print("STEWARD_QUEUE_PATH:", STEWARD_QUEUE_PATH)
print("REJECTS_PATH:", REJECTS_PATH)
print("AUDIT_PATH:", AUDIT_PATH)
print("HIGH_CONF:", HIGH_CONF, "MED_CONF:", MED_CONF)

# ------------------------------------------------------------
# Read zones reference (CSV)
# ------------------------------------------------------------
zones = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RAW_ZONES_PATH)
)

print("Zones rows:", zones.count())
print("Zones columns:", zones.columns)

# Expected NYC Taxi zone columns (common):
# LocationID, Borough, Zone, service_zone
if "LocationID" not in zones.columns:
    raise Exception(f"Expected LocationID in zones. Found: {zones.columns}")
if "Zone" not in zones.columns and "zone_name" not in zones.columns:
    raise Exception(f"Expected Zone or zone_name in zones. Found: {zones.columns}")

zone_col = "Zone" if "Zone" in zones.columns else "zone_name"

# ------------------------------------------------------------
# Standardize fields
# ------------------------------------------------------------
def normalize_text_col(col):
    # Uppercase, trim, remove non-alphanum (keep spaces), collapse spaces
    return F.trim(
        F.regexp_replace(
            F.regexp_replace(F.upper(F.col(col).cast("string")), r"[^A-Z0-9 ]", " "),
            r"\s+", " "
        )
    )

z = zones.withColumn("zone_norm", normalize_text_col(zone_col))

# Create match key: Borough + normalized zone
if "Borough" in z.columns:
    z = z.withColumn("borough_norm", normalize_text_col("Borough"))
    z = z.withColumn("match_key", F.concat_ws("::", F.col("borough_norm"), F.col("zone_norm")))
else:
    z = z.withColumn("match_key", F.col("zone_norm"))

# ------------------------------------------------------------
# Dedup logic + confidence scoring
# ------------------------------------------------------------
# We’ll treat duplicates as rows sharing the same match_key.
# Confidence:
# - If match_key duplicates exist but zone_norm identical -> very high
# - If match_key duplicates exist but slight differences -> medium
# - Else -> high (unique)
w = Window.partitionBy("match_key")

z = (
    z
    .withColumn("dupe_count", F.count("*").over(w))
)

# Create a representative "canonical_zone" = first zone_norm in key group
z = z.withColumn("canonical_zone_norm", F.first("zone_norm").over(w))

# similarity proxy:
# exact match = 1.0, else 0.85 (simple deterministic proxy without extra libs)
z = z.withColumn(
    "name_similarity",
    F.when(F.col("zone_norm") == F.col("canonical_zone_norm"), F.lit(1.0)).otherwise(F.lit(0.85))
)

# confidence scoring heuristic
z = z.withColumn(
    "match_confidence",
    F.when(F.col("dupe_count") == 1, F.lit(0.99))  # unique keys are very reliable
     .when((F.col("dupe_count") > 1) & (F.col("name_similarity") == 1.0), F.lit(0.97))
     .otherwise(F.lit(0.88))
)

# Decide workflow bucket
z = z.withColumn(
    "decision",
    F.when(F.col("match_confidence") >= F.lit(HIGH_CONF), F.lit("AUTO_MERGE"))
     .when(F.col("match_confidence") >= F.lit(MED_CONF), F.lit("STEWARD_REVIEW"))
     .otherwise(F.lit("REJECT"))
)

# ------------------------------------------------------------
# Build Golden Master (AUTO_MERGE only)
# Survivorship: keep smallest LocationID per match_key as the "golden" record
# ------------------------------------------------------------
if "LocationID" in z.columns:
    w_gold = Window.partitionBy("match_key").orderBy(F.col("LocationID").asc())
else:
    w_gold = Window.partitionBy("match_key").orderBy(F.col(zone_col).asc())

golden = (
    z.filter(F.col("decision") == "AUTO_MERGE")
     .withColumn("rn", F.row_number().over(w_gold))
     .filter(F.col("rn") == 1)
     .drop("rn")
     .withColumn("mdm_state", F.lit("ACTIVE"))
     .withColumn("created_at_utc", F.current_timestamp())
     .withColumn("updated_at_utc", F.current_timestamp())
     .withColumn("source_system", F.lit("raw_reference"))
     .withColumn("survivorship_rule", F.lit("min(LocationID) per match_key"))
)

# Steward queue (STEWARD_REVIEW only)
steward = (
    z.filter(F.col("decision") == "STEWARD_REVIEW")
     .withColumn("mdm_state", F.lit("PROPOSED"))
     .withColumn("submitted_at_utc", F.current_timestamp())
     .withColumn("review_reason", F.lit("Medium confidence match; requires steward review"))
)

# Rejects (REJECT only)
rejects = (
    z.filter(F.col("decision") == "REJECT")
     .withColumn("rejected_at_utc", F.current_timestamp())
     .withColumn("reject_reason", F.lit("Low confidence match; insufficient data quality"))
)

# ------------------------------------------------------------
# Write outputs
# ------------------------------------------------------------
# Golden master as Delta (needs --datalake-formats=delta only if using delta;
# for simplicity, you can also write parquet if you prefer)
golden.write.mode("overwrite").format("delta").save(MASTER_OUT_PATH)

steward.write.mode("overwrite").format("parquet").save(STEWARD_QUEUE_PATH)
rejects.write.mode("overwrite").format("parquet").save(REJECTS_PATH)

print("✅ Master (golden) written:", MASTER_OUT_PATH)
print("✅ Steward queue written:", STEWARD_QUEUE_PATH)
print("✅ Rejects written:", REJECTS_PATH)

# ------------------------------------------------------------
# Audit summary (JSON)
# ------------------------------------------------------------
audit = (
    z.agg(
        F.count("*").alias("total_rows"),
        F.sum(F.when(F.col("decision") == "AUTO_MERGE", 1).otherwise(0)).alias("auto_merge_rows"),
        F.sum(F.when(F.col("decision") == "STEWARD_REVIEW", 1).otherwise(0)).alias("steward_review_rows"),
        F.sum(F.when(F.col("decision") == "REJECT", 1).otherwise(0)).alias("reject_rows"),
        F.avg("match_confidence").alias("avg_match_confidence"),
    )
    .withColumn("high_conf_threshold", F.lit(HIGH_CONF))
    .withColumn("med_conf_threshold", F.lit(MED_CONF))
    .withColumn("raw_zones_path", F.lit(RAW_ZONES_PATH))
    .withColumn("master_out_path", F.lit(MASTER_OUT_PATH))
    .withColumn("job_name", F.lit(args["JOB_NAME"]))
    .withColumn("generated_at_utc", F.current_timestamp())
)

audit.coalesce(1).write.mode("overwrite").json(AUDIT_PATH)
print("✅ Audit report written:", AUDIT_PATH)
