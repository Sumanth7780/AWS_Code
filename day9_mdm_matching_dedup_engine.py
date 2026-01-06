import sys
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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
HIGH_CONF = float(args["HIGH_CONF"])   # e.g., 0.95
MED_CONF = float(args["MED_CONF"])     # e.g., 0.80

# ------------------------------------------------------------
# Spark
# ------------------------------------------------------------
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

print("=== Day 9 MDM Matching & Dedup (Improved) ===")
print("RAW_ZONES_PATH:", RAW_ZONES_PATH)
print("MASTER_OUT_PATH:", MASTER_OUT_PATH)
print("STEWARD_QUEUE_PATH:", STEWARD_QUEUE_PATH)
print("REJECTS_PATH:", REJECTS_PATH)
print("AUDIT_PATH:", AUDIT_PATH)
print("HIGH_CONF:", HIGH_CONF, "MED_CONF:", MED_CONF)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def normalize_text_col(colname: str):
    # Uppercase, trim, remove non-alphanum (keep spaces), collapse spaces
    return F.trim(
        F.regexp_replace(
            F.regexp_replace(F.upper(F.col(colname).cast("string")), r"[^A-Z0-9 ]", " "),
            r"\s+", " "
        )
    )

def normalized_lev_similarity(col_a, col_b):
    # similarity = 1 - levenshtein(a,b)/max(len(a), len(b))
    max_len = F.greatest(F.length(col_a), F.length(col_b))
    lev = F.levenshtein(col_a, col_b)
    return F.when(max_len == 0, F.lit(0.0)).otherwise(F.round(1 - (lev / max_len), 4))

def write_delta_if_available(df, path):
    """
    Try to write Delta. If Delta isn't available, fall back to Parquet.
    This prevents ClassNotFoundException: delta.
    """
    try:
        df.write.mode("overwrite").format("delta").save(path)
        print(f"✅ Wrote DELTA: {path}")
        return "delta"
    except Exception as e:
        print("⚠ Delta write failed, falling back to Parquet.")
        print("Reason:", str(e)[:400])
        df.write.mode("overwrite").format("parquet").save(path)
        print(f"✅ Wrote PARQUET: {path}")
        return "parquet"

# ------------------------------------------------------------
# Read zones reference (CSV)
# ------------------------------------------------------------
zones = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RAW_ZONES_PATH)
)

if "LocationID" not in zones.columns:
    raise Exception(f"Expected LocationID in zones. Found: {zones.columns}")

zone_col = "Zone" if "Zone" in zones.columns else ("zone_name" if "zone_name" in zones.columns else None)
if zone_col is None:
    raise Exception(f"Expected Zone or zone_name in zones. Found: {zones.columns}")

# Standardize columns
z = (
    zones
    .withColumnRenamed("LocationID", "location_id")
    .withColumn("location_id", F.col("location_id").cast("int"))
    .withColumn("zone_norm", normalize_text_col(zone_col))
)

if "Borough" in z.columns:
    z = z.withColumn("borough_norm", normalize_text_col("Borough"))
else:
    z = z.withColumn("borough_norm", F.lit("UNKNOWN"))

# Deterministic match_key
z = z.withColumn("match_key", F.concat_ws("::", F.col("borough_norm"), F.col("zone_norm")))

print("Zones rows:", z.count())

# ------------------------------------------------------------
# 1) Exact duplicate evidence (for screenshots)
# ------------------------------------------------------------
dup_by_match_key = z.groupBy("match_key").count().filter(F.col("count") > 1)

dup_by_id = z.groupBy("location_id").count().filter(F.col("count") > 1)

# ------------------------------------------------------------
# 2) Pairwise matching (A vs B) within same borough (blocking)
#    This is what makes steward review queue “real”.
# ------------------------------------------------------------
a = z.alias("a")
b = z.alias("b")

pairs = (
    a.join(
        b,
        (F.col("a.borough_norm") == F.col("b.borough_norm")) &
        (F.col("a.location_id") < F.col("b.location_id")),
        "inner"
    )
    .select(
        F.col("a.location_id").alias("id_a"),
        F.col("b.location_id").alias("id_b"),
        F.col("a.zone_norm").alias("zone_a"),
        F.col("b.zone_norm").alias("zone_b"),
        F.col("a.borough_norm").alias("borough"),
    )
)

pairs = pairs.withColumn(
    "name_similarity",
    normalized_lev_similarity(F.col("zone_a"), F.col("zone_b"))
)

# Confidence score (simple + explainable):
# Mostly name similarity; small boost if first token matches (helps abbreviations)
pairs = pairs.withColumn("token_a", F.split(F.col("zone_a"), " ").getItem(0)) \
             .withColumn("token_b", F.split(F.col("zone_b"), " ").getItem(0)) \
             .withColumn("token_match", F.when(F.col("token_a") == F.col("token_b"), F.lit(1.0)).otherwise(F.lit(0.0))) \
             .withColumn(
                 "match_confidence",
                 F.round(F.col("name_similarity") * F.lit(0.90) + F.col("token_match") * F.lit(0.10), 4)
             )

pairs = pairs.withColumn(
    "decision",
    F.when(F.col("match_confidence") >= F.lit(HIGH_CONF), F.lit("AUTO_MERGE"))
     .when(F.col("match_confidence") >= F.lit(MED_CONF), F.lit("STEWARD_REVIEW"))
     .otherwise(F.lit("REJECT"))
)

# Outputs for screenshots
match_confidence_output = pairs.orderBy(F.desc("match_confidence"))

steward_queue_pairs = (
    pairs.filter(F.col("decision") == "STEWARD_REVIEW")
         .withColumn("review_status", F.lit("PENDING"))
         .withColumn("submitted_at_utc", F.current_timestamp())
         .select("id_a", "id_b", "zone_a", "zone_b", "borough",
                 "name_similarity", "match_confidence", "decision",
                 "review_status", "submitted_at_utc")
         .orderBy(F.desc("match_confidence"))
)

reject_pairs = pairs.filter(F.col("decision") == "REJECT")

auto_merge_pairs = pairs.filter(F.col("decision") == "AUTO_MERGE")

# ------------------------------------------------------------
# 3) Golden master creation
#    For zones, your “golden” can be one row per match_key (deduped view)
# ------------------------------------------------------------
# Choose canonical per (borough_norm, zone_norm) deterministically: min(location_id)
gold_w = Window.partitionBy("match_key").orderBy(F.col("location_id").asc())

golden_master = (
    z.withColumn("rn", F.row_number().over(gold_w))
     .filter(F.col("rn") == 1)
     .drop("rn")
     .withColumn("mdm_state", F.lit("ACTIVE"))
     .withColumn("created_at_utc", F.current_timestamp())
     .withColumn("updated_at_utc", F.current_timestamp())
     .withColumn("survivorship_rule", F.lit("min(location_id) per match_key"))
)

# ------------------------------------------------------------
# 4) Write outputs
# ------------------------------------------------------------
# Master: try Delta, else parquet
master_format = write_delta_if_available(golden_master, MASTER_OUT_PATH)

# Steward queue + rejects as parquet (easy to query + screenshot)
steward_queue_pairs.write.mode("overwrite").format("parquet").save(STEWARD_QUEUE_PATH)
reject_pairs.write.mode("overwrite").format("parquet").save(REJECTS_PATH)

# Optional extra evidence folders (good for screenshots)
dup_by_id.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{AUDIT_PATH}/dup_by_location_id/")
dup_by_match_key.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{AUDIT_PATH}/dup_by_match_key/")
match_confidence_output.write.mode("overwrite").format("parquet").save(f"{AUDIT_PATH}/match_confidence_output/")

print("✅ Steward queue written:", STEWARD_QUEUE_PATH)
print("✅ Reject pairs written:", REJECTS_PATH)

# ------------------------------------------------------------
# 5) Audit summary (JSON)
# ------------------------------------------------------------
audit = (
    z.agg(F.count("*").alias("total_zone_rows"))
     .crossJoin(
        pairs.agg(
            F.count("*").alias("total_pairs_compared"),
            F.sum(F.when(F.col("decision") == "AUTO_MERGE", 1).otherwise(0)).alias("auto_merge_pairs"),
            F.sum(F.when(F.col("decision") == "STEWARD_REVIEW", 1).otherwise(0)).alias("steward_review_pairs"),
            F.sum(F.when(F.col("decision") == "REJECT", 1).otherwise(0)).alias("reject_pairs"),
            F.avg("match_confidence").alias("avg_match_confidence"),
        )
     )
     .withColumn("high_conf_threshold", F.lit(HIGH_CONF))
     .withColumn("med_conf_threshold", F.lit(MED_CONF))
     .withColumn("raw_zones_path", F.lit(RAW_ZONES_PATH))
     .withColumn("master_out_path", F.lit(MASTER_OUT_PATH))
     .withColumn("master_format", F.lit(master_format))
     .withColumn("steward_queue_path", F.lit(STEWARD_QUEUE_PATH))
     .withColumn("rejects_path", F.lit(REJECTS_PATH))
     .withColumn("job_name", F.lit(args["JOB_NAME"]))
     .withColumn("generated_at_utc", F.current_timestamp())
)

audit.coalesce(1).write.mode("overwrite").json(AUDIT_PATH)
print("✅ Audit report written:", AUDIT_PATH)
print("✅ Day 9 done.")
