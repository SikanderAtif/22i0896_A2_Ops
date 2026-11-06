"""Inspect the VisionGuard prediction trail with plain SQL. Works for SQLite and PostgreSQL."""
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

load_dotenv()
url = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")
if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
    # Flask-SQLAlchemy stores relative SQLite files in the app's instance/ folder
    url = "sqlite:///instance/" + url[len("sqlite:///"):]
engine = create_engine(url)
print("Database:", engine.url.render_as_string(hide_password=True))

insp = inspect(engine)
print("\n== Table: predictions ==")
for c in insp.get_columns("predictions"):
    print(f"  {c['name']:16s} {str(c['type']):24s} {'NULL' if c['nullable'] else 'NOT NULL'}")
print("  primary key:", insp.get_pk_constraint("predictions")["constrained_columns"])
print("  indexes:    ", [i["name"] for i in insp.get_indexes("predictions")])

pct = lambda col: f"ROUND(CAST({col} * 100 AS NUMERIC), 1)"
QUERIES = {
    "Total predictions":
        "SELECT COUNT(*) AS total FROM predictions",
    "Latest 10: what was decided, when, by which model, how confidently":
        f"""SELECT id, created_at, filename, label, {pct('confidence')} AS confidence_pct,
                   ROUND(CAST(p_ai AS NUMERIC), 3) AS p_ai, model_name, model_version
            FROM predictions ORDER BY created_at DESC LIMIT 10""",
    "Verdicts per model version":
        f"""SELECT model_name, model_version, label, COUNT(*) AS n,
                   {pct('AVG(confidence)')} AS avg_conf_pct,
                   ROUND(CAST(AVG(latency_ms) AS NUMERIC), 0) AS avg_latency_ms
            FROM predictions GROUP BY model_name, model_version, label
            ORDER BY model_version, label""",
    "Low-confidence verdicts (below 50%)":
        f"""SELECT id, created_at, filename, label, {pct('confidence')} AS confidence_pct
            FROM predictions WHERE confidence < 0.5 ORDER BY created_at DESC""",
    "Same image submitted more than once (matched by SHA-256)":
        """SELECT SUBSTR(image_sha256, 1, 12) AS sha256_prefix, COUNT(*) AS times,
                  MIN(created_at) AS first_seen, MAX(created_at) AS last_seen
           FROM predictions GROUP BY image_sha256 HAVING COUNT(*) > 1""",
}
with engine.connect() as conn:
    for title, sql in QUERIES.items():
        print(f"\n== {title} ==")
        df = pd.read_sql(text(sql), conn)
        print(df.to_string(index=False) if len(df) else "  (no rows)")