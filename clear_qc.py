import psycopg2, os
from dotenv import load_dotenv

load_dotenv("D:/Hare Krishna_ai_blog/.env")
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    sslmode=os.getenv("DB_SSLMODE", "require"),
)
with conn.cursor() as cur:
    cur.execute(
        "UPDATE public.article_requests SET gcs_qc_uri=NULL, qc_fingerprint=NULL, qc_summary=NULL, qc_meta=NULL WHERE request_id=%s::uuid",
        ("e7f64088-52bf-43ae-9d8b-476c791f8b21",),
    )
conn.commit()
conn.close()
print("QC pointers cleared")
