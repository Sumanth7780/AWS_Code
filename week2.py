import sys
import json
from typing import Any, Dict, List, Optional

from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import TimestampType, DoubleType
from pyspark.sql import Window

# DeltaTable history audit (optional)
try:
    from delta.tables import DeltaTable
    DELTA_AVAILABLE = True
except Exception:
    DELTA_AVAILABLE = False

# YAML parsing (Glue may or may not have PyYAML by default)
try:
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:
    YAML_AVAILABLE = False


# ============================================================
# Utility helpers
# ============================================================

def _strip(v: str) -> str:
    return (v or "").strip()

def require_non_empty(name: str, value: str) -> str:
    v = _strip(value)
    if not v:
        raise ValueError(f"Argument {name} is empty. Pass a non-empty value/path.")
    return v

def rstrip_slash(p: str) -> str:
    return (p or "").rstrip("/")

def first_existing(candidates, cols):
    for c in candidates:
        if c in cols:
            return c
    return None

def spark_read_small_text(spark, path: str) -> str:
    """
    Reads a small text file from S3 (or local) into a single string.
    Works well for small JSON/YAML governance artifacts.
    """
    lines = [r["value"] for r in spark.read.text(path).collect()]
    return "\n".join(lines)

def load_json_from_path(spark, path: str) -> Dict[str, Any]:
    raw = spark_read_small_text(spark, path)
    return json.loads(raw)

def load_yaml_from_path(spark, path: str) -> Dict[str, Any]:
    raw = spark_read_small_text(spark, path)

    # If yaml is available, parse YAML.
    if YAML_AVAILABLE:
        return yaml.safe_load(raw)

    # Fallback: attempt JSON parsing if user provided JSON in a .yaml file
    try:
        return json.loads(raw)
    except Exception:
        raise RuntimeError(
            "PyYAML is not available in this Glue runtime, and the rules file is not valid JSON. "
            "Fix options:\n"
            "1) Add Glue job param: --additional-python-modules pyyaml==6.0.1\n"
            "2) Convert the YAML files to JSON and pass JSON paths instead."
        )

def get_optional_arg(argv: List[str], key: str) -> Optional[str]:
    """
    Optional arg parsing for --KEY value.
    Example: --MIN_PASS_RATE 0.97
    """
    flag = f"--{key}"
    for i in range(len(argv) - 1):
        if argv[i] == flag:
            return argv[i + 1]
    return None


# ============================================================
# Load governance artifacts (metadata + DQ + MDM rules)
# ============================================================

def load_governance_artifacts(spark, metadata_json_path: str,
                              quality_rules_path: str,
                              match_merge_rules_path: str,
                              lifecycle_states_path: str) -> Dict[str, Any]:
    metadata = load_json_from_path(spark, metadata_json_path)
    quality = load_yaml_from_path(spark, quality_rules_path)
    match_merge = load_yaml_from_path(spark, match_merge_rules_path)
    lifecycle = load_yaml_from_path(spark, lifecycle_states_path)

    # Basic sanity prints
    print("=== Governance Artifacts Loaded ===")
    print("metadata.dataset_name:", metadata.get("dataset_name"))
    print("metadata.classification:", metadata.get("classification"))
    print("quality.version:", quality.get("version"))
    print("match_merge.version:", match_merge.get("version"))
    print("lifecycle.version:", lifecycle.get("version"))

    return {
        "metadata": metadata,
        "quality": quality,
        "match_merge": match_merge,
        "lifecycle": lifecycle,
    }


# ============================================================
# Day 6: Raw -> Validated (Delta) with schema alignment from metadata JSON
# ============================================================

def day6_raw_to_validated(
    spark,
    RAW_TRIPS_PATH: str,
    VALIDATED_DELTA_PATH: str,
    metadata: Dict[str, Any]
):
    print("\n=== Day 6: Raw -> Validated (Delta) ===")
    print("RAW_TRIPS_PATH:", RAW_TRIPS_PATH)
    print("VALIDATED_DELTA_PATH:", VALIDATED_DELTA_PATH)

    trips = spark.read.parquet(RAW_TRIPS_PATH)

    # Use metadata schema (if present) to cast known columns safely
    # We won't drop extra columns—just standardize types for known schema fields.
    schema_list = metadata.get("schema", [])
    type_map = {s["name"]: s.get("type") for s in schema_list if isinstance(s, dict) and "name" in s}

    df = trips

    # Cast key columns based on metadata contract (best-effort)
    for col_name, col_type in type_map.items():
        if col_name in df.columns:
            if col_type == "timestamp":
                df = df.withColumn(col_name, F.col(col_name).cast(TimestampType()))
            elif col_type == "double":
                df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))
            elif col_type == "int":
                df = df.withColumn(col_name, F.col(col_name).cast("int"))
            elif col_type == "string":
                df = df.withColumn(col_name, F.col(col_name).cast("string"))

    # Standardize pickup/dropoff and partition column
    pickup_col = first_existing(["tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"], df.columns)
    dropoff_col = first_existing(["tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime"], df.columns)
    if not pickup_col or not dropoff_col:
        raise ValueError(f"Pickup/Dropoff columns not found. Columns={df.columns}")

    df = (
        df
        .withColumn("pickup_ts", F.col(pickup_col).cast(TimestampType()))
        .withColumn("dropoff_ts", F.col(dropoff_col).cast(TimestampType()))
        .withColumn("pickup_date", F.to_date("pickup_ts"))
    )

    # Basic validation gates (safe defaults)
    df = df.filter(F.col("pickup_ts").isNotNull() & F.col("dropoff_ts").isNotNull())
    df = df.filter(F.col("dropoff_ts") >= F.col("pickup_ts"))

    # Keep legitimate zero values; drop negatives only
    if "fare_amount" in df.columns:
        df = df.filter(F.col("fare_amount").cast(DoubleType()) >= F.lit(0.0))
    if "total_amount" in df.columns:
        df = df.filter(F.col("total_amount").cast(DoubleType()) >= F.lit(0.0))
    if "trip_distance" in df.columns:
        df = df.filter(F.col("trip_distance").cast(DoubleType()) >= F.lit(0.0))
    if "passenger_count" in df.columns:
        df = df.filter(F.col("passenger_count") >= F.lit(0))

    # Fix common TLC naming mismatch
    if "Airport_fee" in df.columns and "airport_fee" not in df.columns:
        df = df.withColumnRenamed("Airport_fee", "airport_fee")

    # Governance metadata columns
    df = (
        df
        .withColumn("governance_zone", F.lit("validated"))
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("source_system", F.lit(metadata.get("source", "NYC_TLC")))
        .withColumn("dataset_name", F.lit(metadata.get("dataset_name", "unknown_dataset")))
        .withColumn("classification", F.lit(metadata.get("classification", "unknown")))
    )

    (df.write.format("delta")
        .mode("overwrite")
        .partitionBy("pickup_date")
        .save(VALIDATED_DELTA_PATH))

    print(" Day 6 complete:", VALIDATED_DELTA_PATH)


# ============================================================
# Day 7: Validated -> Curated (Enrich with zones; business-ready shape)
# ============================================================

def day7_validated_to_curated(
    spark,
    VALIDATED_DELTA_PATH: str,
    RAW_ZONES_CSV_PATH: str,
    CURATED_DELTA_PATH: str
):
    print("\n=== Day 7: Validated -> Curated (Delta) ===")
    print("VALIDATED_DELTA_PATH:", VALIDATED_DELTA_PATH)
    print("RAW_ZONES_CSV_PATH:", RAW_ZONES_CSV_PATH)
    print("CURATED_DELTA_PATH:", CURATED_DELTA_PATH)

    trips = spark.read.format("delta").load(VALIDATED_DELTA_PATH)

    zones = (
        spark.read.option("header", True).csv(RAW_ZONES_CSV_PATH)
        .withColumn("LocationID", F.col("LocationID").cast("int"))
        .withColumn("Borough", F.col("Borough").cast("string"))
        .withColumn("Zone", F.col("Zone").cast("string"))
        .withColumn("service_zone", F.col("service_zone").cast("string"))
    )

    # Derived feature: trip duration minutes (if timestamps exist)
    if "pickup_ts" in trips.columns and "dropoff_ts" in trips.columns:
        trips = trips.withColumn(
            "trip_duration_min",
            (F.col("dropoff_ts").cast("long") - F.col("pickup_ts").cast("long")) / 60.0
        )

    # Join pickup zone details
    if "PULocationID" in trips.columns:
        pu = zones.select(
            F.col("LocationID").alias("PULocationID"),
            F.col("Zone").alias("pu_zone"),
            F.col("Borough").alias("pu_borough"),
            F.col("service_zone").alias("pu_service_zone")
        )
        trips = trips.join(pu, on="PULocationID", how="left")

    # Join dropoff zone details
    if "DOLocationID" in trips.columns:
        do = zones.select(
            F.col("LocationID").alias("DOLocationID"),
            F.col("Zone").alias("do_zone"),
            F.col("Borough").alias("do_borough"),
            F.col("service_zone").alias("do_service_zone")
        )
        trips = trips.join(do, on="DOLocationID", how="left")

    curated = (
        trips
        .withColumn("governance_zone", F.lit("curated"))
        .withColumn("curated_at", F.current_timestamp())
    )

    (curated.write.format("delta")
        .mode("overwrite")
        .partitionBy("pickup_date")
        .save(CURATED_DELTA_PATH))

    print(" Day 7 complete:", CURATED_DELTA_PATH)


# ============================================================
# Day 8: Data Quality from quality_rules.yaml (expressions_sql)
# - Writes: curated(delta passed) + rejects(parquet) + dq_report(json)
# - Enforces min pass rate / fail job based on policy
# ============================================================

def day8_quality_gates(
    spark,
    INPUT_DELTA_PATH: str,
    CURATED_DELTA_OUT_PATH: str,
    REJECTS_PATH: str,
    DQ_REPORT_PATH: str,
    quality_rules: Dict[str, Any],
    job_name: str
):
    print("\n=== Day 8: Data Quality Gates (Config-driven) ===")
    print("INPUT_DELTA_PATH:", INPUT_DELTA_PATH)
    print("CURATED_DELTA_OUT_PATH:", CURATED_DELTA_OUT_PATH)
    print("REJECTS_PATH:", REJECTS_PATH)
    print("DQ_REPORT_PATH:", DQ_REPORT_PATH)

    df = spark.read.format("delta").load(INPUT_DELTA_PATH)

    rules = quality_rules.get("rules", [])
    if not rules:
        raise ValueError("quality_rules.yaml has no rules[]")

    # Build a boolean column per rule using expression_sql
    # Example: "VendorID IS NOT NULL"
    for r in rules:
        rid = r.get("id")
        expr_sql = r.get("expression_sql")
        if not rid or not expr_sql:
            raise ValueError(f"Bad quality rule entry: {r}")
        df = df.withColumn(rid, F.expr(expr_sql))

    rule_ids = [r["id"] for r in rules]
    dq_pass = F.lit(True)
    for rid in rule_ids:
        dq_pass = dq_pass & F.col(rid)

    df = df.withColumn("dq_pass", dq_pass)

    # Add list of failed rule ids for rejects
    failed_array = F.array(*[
        F.when(F.col(rid) == False, F.lit(rid)).otherwise(F.lit(None))
        for rid in rule_ids
    ])
    df = df.withColumn("failed_rules", F.expr("filter(failed_rules_tmp, x -> x is not null)")
                       .alias("failed_rules") if False else df["dq_pass"])  # placeholder to avoid Spark analyzer issues

    # Safe approach: create failed_rules using array_remove repeatedly
    failed_rules_col = failed_array
    for _ in range(5):  # remove nulls a few times (array_remove removes one value; loop is fine for small arrays)
        failed_rules_col = F.array_remove(failed_rules_col, F.lit(None))
    df = df.withColumn("failed_rules", failed_rules_col)

    passed = df.filter(F.col("dq_pass") == True).drop("dq_pass")
    rejected = df.filter(F.col("dq_pass") == False)

    # Write outputs
    passed.write.format("delta").mode("overwrite").save(CURATED_DELTA_OUT_PATH)
    rejected.write.format("parquet").mode("overwrite").save(REJECTS_PATH)

    total_rows = df.count()
    passed_rows = passed.count()
    failed_rows = total_rows - passed_rows
    pass_rate = (passed_rows / total_rows) if total_rows > 0 else 0.0

    # Rule-level fail counts
    agg_exprs = [
        F.sum(F.when(F.col(rid) == False, 1).otherwise(0)).alias(f"{rid}_fail_count")
        for rid in rule_ids
    ]
    summary = (
        df.agg(
            F.count("*").alias("total_rows"),
            F.lit(passed_rows).alias("passed_rows"),
            F.lit(failed_rows).alias("failed_rows"),
            F.lit(pass_rate).alias("pass_rate"),
            *agg_exprs
        )
        .withColumn("job_name", F.lit(job_name))
        .withColumn("input_path", F.lit(INPUT_DELTA_PATH))
        .withColumn("curated_out_path", F.lit(CURATED_DELTA_OUT_PATH))
        .withColumn("rejects_path", F.lit(REJECTS_PATH))
        .withColumn("generated_at_utc", F.current_timestamp())
    )

    summary.coalesce(1).write.mode("overwrite").json(DQ_REPORT_PATH)

    # Enforce policy threshold
    policy = quality_rules.get("policy", {})
    default_min_pass_rate = float(policy.get("default_min_pass_rate", 0.95))
    fail_job = bool(policy.get("fail_job_if_below_threshold", True))

    # Allow override from CLI optional arg: --MIN_PASS_RATE
    override = get_optional_arg(sys.argv, "MIN_PASS_RATE")
    min_pass_rate = float(override) if override is not None else default_min_pass_rate

    print(f"Day8 pass_rate={pass_rate:.4f}, min_pass_rate={min_pass_rate:.4f}, fail_job={fail_job}")

    if fail_job and pass_rate < min_pass_rate:
        raise RuntimeError(
            f"Data quality threshold not met. pass_rate={pass_rate:.4f} < min_pass_rate={min_pass_rate:.4f}. "
            f"Rejects saved to {REJECTS_PATH} and report saved to {DQ_REPORT_PATH}."
        )

    print(" Day 8 complete:", CURATED_DELTA_OUT_PATH)


# ============================================================
# Day 9: MDM Matching/Dedup (Config-driven from match_merge_rules.yaml)
# - Builds match_key using configured SQL expression
# - Uses thresholds from YAML (or optional CLI overrides)
# - Writes master zones Delta + steward queue + rejects + audit
# ============================================================

def day9_mdm_matching_dedup(
    spark,
    RAW_ZONES_CSV_PATH: str,
    MASTER_ZONES_DELTA_PATH: str,
    STEWARD_QUEUE_PATH: str,
    MDM_REJECTS_PATH: str,
    MDM_AUDIT_PATH: str,
    match_merge_rules: Dict[str, Any],
    lifecycle_states: Dict[str, Any]
):
    print("\n=== Day 9: MDM Match/Merge (Config-driven) ===")
    print("RAW_ZONES_CSV_PATH:", RAW_ZONES_CSV_PATH)
    print("MASTER_ZONES_DELTA_PATH:", MASTER_ZONES_DELTA_PATH)

    zones = (
        spark.read.option("header", True).csv(RAW_ZONES_CSV_PATH)
        .withColumn("LocationID", F.col("LocationID").cast("int"))
        .withColumn("Borough", F.col("Borough").cast("string"))
        .withColumn("Zone", F.col("Zone").cast("string"))
        .withColumn("service_zone", F.col("service_zone").cast("string"))
    )

    # Build match key using configured blocking expression
    blocking_keys = match_merge_rules.get("blocking_keys", [])
    if not blocking_keys:
        raise ValueError("match_merge_rules.yaml missing blocking_keys[]")

    match_key_expr = blocking_keys[0].get("expression")
    if not match_key_expr:
        raise ValueError("match_merge_rules.yaml blocking_keys[0].expression is missing")

    z = zones.withColumn("match_key", F.expr(match_key_expr))

    # Thresholds (allow override via optional CLI --HIGH_CONF / --MED_CONF)
    thr = match_merge_rules.get("thresholds", {})
    high_default = float(thr.get("high_confidence_auto_merge", 0.95))
    med_default = float(thr.get("medium_confidence_steward_review", 0.80))

    high_override = get_optional_arg(sys.argv, "HIGH_CONF")
    med_override = get_optional_arg(sys.argv, "MED_CONF")
    HIGH_CONF = float(high_override) if high_override is not None else high_default
    MED_CONF = float(med_override) if med_override is not None else med_default

    # Simple deterministic scoring (per match_scoring notes)
    # - exact match_key within group -> high
    w = Window.partitionBy("match_key")
    z = z.withColumn("dupe_count", F.count("*").over(w))

    z = z.withColumn(
        "match_confidence",
        F.when(F.col("dupe_count") == 1, F.lit(0.99))
         .when(F.col("dupe_count") > 1, F.lit(0.88))
         .otherwise(F.lit(0.88))
    )

    z = z.withColumn(
        "decision",
        F.when(F.col("match_confidence") >= F.lit(HIGH_CONF), F.lit("AUTO_MERGE"))
         .when(F.col("match_confidence") >= F.lit(MED_CONF), F.lit("STEWARD_REVIEW"))
         .otherwise(F.lit("REJECT"))
    )

    # Lifecycle states validation
    allowed_states = {s["name"] for s in lifecycle_states.get("states", []) if isinstance(s, dict) and "name" in s}
    if "ACTIVE" not in allowed_states or "PROPOSED" not in allowed_states or "REJECTED" not in allowed_states:
        raise ValueError("lifecycle_states.yaml must include ACTIVE, PROPOSED, REJECTED states")

    # Survivorship: min(LocationID) per match_key (configured in spirit of merge_policy)
    w_gold = Window.partitionBy("match_key").orderBy(F.col("LocationID").asc())

    master = (
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

    steward_queue = (
        z.filter(F.col("decision") == "STEWARD_REVIEW")
         .withColumn("mdm_state", F.lit("PROPOSED"))
         .withColumn("submitted_at_utc", F.current_timestamp())
         .withColumn("submitted_by", F.lit("glue_job"))
         .withColumn("review_reason", F.lit("Medium confidence match; requires steward review"))
    )

    mdm_rejects = (
        z.filter(F.col("decision") == "REJECT")
         .withColumn("mdm_state", F.lit("REJECTED"))
         .withColumn("rejected_at_utc", F.current_timestamp())
         .withColumn("reject_reason", F.lit("Low confidence match; insufficient data quality"))
    )

    # Write outputs
    master.write.format("delta").mode("overwrite").save(MASTER_ZONES_DELTA_PATH)
    steward_queue.write.format("parquet").mode("overwrite").save(STEWARD_QUEUE_PATH)
    mdm_rejects.write.format("parquet").mode("overwrite").save(MDM_REJECTS_PATH)

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
        .withColumn("raw_zones_path", F.lit(RAW_ZONES_CSV_PATH))
        .withColumn("master_out_path", F.lit(MASTER_ZONES_DELTA_PATH))
        .withColumn("generated_at_utc", F.current_timestamp())
    )

    audit.coalesce(1).write.mode("overwrite").json(MDM_AUDIT_PATH)

    print("Day 9 complete:", MASTER_ZONES_DELTA_PATH)


# ============================================================
# Day 10: Orphans + Lifecycle snapshot + Delta history audit + Run summary
# ============================================================

def day10_lifecycle_orphans_audit(
    spark,
    JOB_NAME: str,
    MASTER_ZONES_DELTA_PATH: str,
    CURATED_TRIPS_DELTA_PATH: str,
    ORPHANS_OUT_PATH: str,
    LIFECYCLE_SNAPSHOT_PATH: str,
    AUDIT_HISTORY_OUT_PATH: str,
    RUN_SUMMARY_PATH: str,
    lifecycle_states: Dict[str, Any]
):
    print("\n=== Day 10: Lifecycle + Orphans + Audit ===")

    zones = spark.read.format("delta").load(MASTER_ZONES_DELTA_PATH)
    trips = spark.read.format("delta").load(CURATED_TRIPS_DELTA_PATH)

    # Orphan detection requires these keys
    if "LocationID" not in zones.columns:
        raise ValueError(f"Master zones must contain LocationID. Found: {zones.columns}")
    if "PULocationID" not in trips.columns or "DOLocationID" not in trips.columns:
        raise ValueError("Trips must contain PULocationID and DOLocationID for orphan detection.")

    zones_ids = zones.select(F.col("LocationID").cast("int").alias("LocationID")).dropna().distinct()

    trips_ids = (
        trips
        .withColumn("PULocationID_int", F.col("PULocationID").cast("int"))
        .withColumn("DOLocationID_int", F.col("DOLocationID").cast("int"))
    )

    pu_orphans = (
        trips_ids.join(zones_ids, trips_ids["PULocationID_int"] == zones_ids["LocationID"], "left_anti")
        .withColumn("orphan_type", F.lit("PICKUP_NOT_IN_MASTER"))
    )
    do_orphans = (
        trips_ids.join(zones_ids, trips_ids["DOLocationID_int"] == zones_ids["LocationID"], "left_anti")
        .withColumn("orphan_type", F.lit("DROPOFF_NOT_IN_MASTER"))
    )

    orphans = (
        pu_orphans.unionByName(do_orphans, allowMissingColumns=True)
        .withColumn("job_name", F.lit(JOB_NAME))
        .withColumn("detected_at_utc", F.current_timestamp())
    )

    orphans_count = orphans.count()
    orphans.write.mode("overwrite").format("delta").save(ORPHANS_OUT_PATH)

    # Lifecycle snapshot (validate mdm_state against config)
    allowed_states = {s["name"] for s in lifecycle_states.get("states", []) if isinstance(s, dict) and "name" in s}
    if "mdm_state" in zones.columns:
        lifecycle = (
            zones.withColumn(
                "mdm_state_valid",
                F.when(F.col("mdm_state").isin(list(allowed_states)), F.lit(True)).otherwise(F.lit(False))
            )
            .groupBy("mdm_state", "mdm_state_valid")
            .agg(F.count("*").alias("record_count"))
            .withColumn("job_name", F.lit(JOB_NAME))
            .withColumn("snapshot_at_utc", F.current_timestamp())
        )
    else:
        lifecycle = (
            zones.select(F.lit("ACTIVE").alias("mdm_state"), F.lit(True).alias("mdm_state_valid"))
            .groupBy("mdm_state", "mdm_state_valid")
            .agg(F.count("*").alias("record_count"))
            .withColumn("job_name", F.lit(JOB_NAME))
            .withColumn("snapshot_at_utc", F.current_timestamp())
        )

    lifecycle.write.mode("overwrite").format("delta").save(LIFECYCLE_SNAPSHOT_PATH)

    # Delta history audit (time travel compliance)
    if DELTA_AVAILABLE:
        def hist(path: str, table_name: str):
            dt = DeltaTable.forPath(spark, path)
            h = dt.history()
            return (
                h.withColumn("table_name", F.lit(table_name))
                 .withColumn("table_path", F.lit(path))
                 .withColumn("job_name", F.lit(JOB_NAME))
                 .withColumn("captured_at_utc", F.current_timestamp())
            )

        history = hist(MASTER_ZONES_DELTA_PATH, "master_zones").unionByName(
            hist(CURATED_TRIPS_DELTA_PATH, "curated_trips"),
            allowMissingColumns=True
        )
        history.write.mode("overwrite").format("delta").save(AUDIT_HISTORY_OUT_PATH)
    else:
        print("WARNING: delta.tables.DeltaTable not available; skipping Delta history audit.")

    # Run summary
    summary = spark.createDataFrame([{
        "job_name": JOB_NAME,
        "master_zones_path": MASTER_ZONES_DELTA_PATH,
        "curated_trips_path": CURATED_TRIPS_DELTA_PATH,
        "orphans_out_path": ORPHANS_OUT_PATH,
        "lifecycle_snapshot_path": LIFECYCLE_SNAPSHOT_PATH,
        "audit_history_out_path": AUDIT_HISTORY_OUT_PATH,
        "orphans_count": int(orphans_count),
    }]).withColumn("generated_at_utc", F.current_timestamp())

    summary.write.mode("overwrite").format("delta").save(RUN_SUMMARY_PATH)

    print(" Day 10 complete. Orphans:", orphans_count)


# ============================================================
# MAIN
# ============================================================

args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "RUN_MODE",

    # Data paths
    "RAW_TRIPS_PATH",
    "RAW_ZONES_PATH",

    "VALIDATED_DELTA_PATH",
    "CURATED_DELTA_PATH",

    # DQ outputs
    "DQ_CURATED_DELTA_PATH",
    "REJECTS_PATH",
    "DQ_REPORT_PATH",

    # MDM outputs
    "MASTER_ZONES_DELTA_PATH",
    "STEWARD_QUEUE_PATH",
    "MDM_REJECTS_PATH",
    "MDM_AUDIT_PATH",

    # Day10 outputs
    "ORPHANS_OUT_PATH",
    "LIFECYCLE_SNAPSHOT_PATH",
    "AUDIT_HISTORY_OUT_PATH",
    "RUN_SUMMARY_PATH",

    # Governance artifacts
    "METADATA_JSON_PATH",
    "QUALITY_RULES_YAML_PATH",
    "MATCH_MERGE_RULES_YAML_PATH",
    "LIFECYCLE_STATES_YAML_PATH",
])

JOB_NAME = args["JOB_NAME"]
RUN_MODE = _strip(args["RUN_MODE"]).lower()

RAW_TRIPS_PATH = require_non_empty("RAW_TRIPS_PATH", args["RAW_TRIPS_PATH"])
RAW_ZONES_PATH = require_non_empty("RAW_ZONES_PATH", args["RAW_ZONES_PATH"])

VALIDATED_DELTA_PATH = require_non_empty("VALIDATED_DELTA_PATH", rstrip_slash(args["VALIDATED_DELTA_PATH"]))
CURATED_DELTA_PATH = require_non_empty("CURATED_DELTA_PATH", rstrip_slash(args["CURATED_DELTA_PATH"]))

DQ_CURATED_DELTA_PATH = require_non_empty("DQ_CURATED_DELTA_PATH", rstrip_slash(args["DQ_CURATED_DELTA_PATH"]))
REJECTS_PATH = require_non_empty("REJECTS_PATH", rstrip_slash(args["REJECTS_PATH"]))
DQ_REPORT_PATH = require_non_empty("DQ_REPORT_PATH", rstrip_slash(args["DQ_REPORT_PATH"]))

MASTER_ZONES_DELTA_PATH = require_non_empty("MASTER_ZONES_DELTA_PATH", rstrip_slash(args["MASTER_ZONES_DELTA_PATH"]))
STEWARD_QUEUE_PATH = require_non_empty("STEWARD_QUEUE_PATH", rstrip_slash(args["STEWARD_QUEUE_PATH"]))
MDM_REJECTS_PATH = require_non_empty("MDM_REJECTS_PATH", rstrip_slash(args["MDM_REJECTS_PATH"]))
MDM_AUDIT_PATH = require_non_empty("MDM_AUDIT_PATH", rstrip_slash(args["MDM_AUDIT_PATH"]))

ORPHANS_OUT_PATH = require_non_empty("ORPHANS_OUT_PATH", rstrip_slash(args["ORPHANS_OUT_PATH"]))
LIFECYCLE_SNAPSHOT_PATH = require_non_empty("LIFECYCLE_SNAPSHOT_PATH", rstrip_slash(args["LIFECYCLE_SNAPSHOT_PATH"]))
AUDIT_HISTORY_OUT_PATH = require_non_empty("AUDIT_HISTORY_OUT_PATH", rstrip_slash(args["AUDIT_HISTORY_OUT_PATH"]))
RUN_SUMMARY_PATH = require_non_empty("RUN_SUMMARY_PATH", rstrip_slash(args["RUN_SUMMARY_PATH"]))

METADATA_JSON_PATH = require_non_empty("METADATA_JSON_PATH", args["METADATA_JSON_PATH"])
QUALITY_RULES_YAML_PATH = require_non_empty("QUALITY_RULES_YAML_PATH", args["QUALITY_RULES_YAML_PATH"])
MATCH_MERGE_RULES_YAML_PATH = require_non_empty("MATCH_MERGE_RULES_YAML_PATH", args["MATCH_MERGE_RULES_YAML_PATH"])
LIFECYCLE_STATES_YAML_PATH = require_non_empty("LIFECYCLE_STATES_YAML_PATH", args["LIFECYCLE_STATES_YAML_PATH"])

print("=== Week2 Governed Pipeline (Glue 4 + Delta) ===")
print("JOB_NAME:", JOB_NAME)
print("RUN_MODE:", RUN_MODE)

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(JOB_NAME, args)

# Load governance artifacts once
gov = load_governance_artifacts(
    spark,
    METADATA_JSON_PATH,
    QUALITY_RULES_YAML_PATH,
    MATCH_MERGE_RULES_YAML_PATH,
    LIFECYCLE_STATES_YAML_PATH
)

metadata = gov["metadata"]
quality = gov["quality"]
match_merge = gov["match_merge"]
lifecycle = gov["lifecycle"]

# Execute
if RUN_MODE in ("all", "day6"):
    day6_raw_to_validated(spark, RAW_TRIPS_PATH, VALIDATED_DELTA_PATH, metadata)

if RUN_MODE in ("all", "day7"):
    day7_validated_to_curated(spark, VALIDATED_DELTA_PATH, RAW_ZONES_PATH, CURATED_DELTA_PATH)

if RUN_MODE in ("all", "day8"):
    day8_quality_gates(
        spark,
        INPUT_DELTA_PATH=CURATED_DELTA_PATH,
        CURATED_DELTA_OUT_PATH=DQ_CURATED_DELTA_PATH,
        REJECTS_PATH=REJECTS_PATH,
        DQ_REPORT_PATH=DQ_REPORT_PATH,
        quality_rules=quality,
        job_name=JOB_NAME
    )

if RUN_MODE in ("all", "day9"):
    day9_mdm_matching_dedup(
        spark,
        RAW_ZONES_CSV_PATH=RAW_ZONES_PATH,
        MASTER_ZONES_DELTA_PATH=MASTER_ZONES_DELTA_PATH,
        STEWARD_QUEUE_PATH=STEWARD_QUEUE_PATH,
        MDM_REJECTS_PATH=MDM_REJECTS_PATH,
        MDM_AUDIT_PATH=MDM_AUDIT_PATH,
        match_merge_rules=match_merge,
        lifecycle_states=lifecycle
    )

if RUN_MODE in ("all", "day10"):
    day10_lifecycle_orphans_audit(
        spark,
        JOB_NAME=JOB_NAME,
        MASTER_ZONES_DELTA_PATH=MASTER_ZONES_DELTA_PATH,
        CURATED_TRIPS_DELTA_PATH=DQ_CURATED_DELTA_PATH,
        ORPHANS_OUT_PATH=ORPHANS_OUT_PATH,
        LIFECYCLE_SNAPSHOT_PATH=LIFECYCLE_SNAPSHOT_PATH,
        AUDIT_HISTORY_OUT_PATH=AUDIT_HISTORY_OUT_PATH,
        RUN_SUMMARY_PATH=RUN_SUMMARY_PATH,
        lifecycle_states=lifecycle
    )

job.commit()
print(" Governed Week2 pipeline finished successfully.")
