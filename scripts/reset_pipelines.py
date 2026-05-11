"""Force-fail all stuck running/pending pipelines so new ones can start."""
import psycopg2

conn = psycopg2.connect(
    host="aws-1-ap-southeast-2.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.dppacvjatqcbolulzxud",
    password="Aiblog@2026", sslmode="require", connect_timeout=8
)
cur = conn.cursor()

cur.execute(
    "SELECT pipeline_id, status, created_at FROM public.pipeline_runs "
    "WHERE status IN ('running','pending') ORDER BY created_at"
)
stuck = cur.fetchall()
print(f"Found {len(stuck)} stuck pipeline(s):")
for r in stuck:
    print(f"  {r[0]}  status={r[1]}  created={r[2]}")

if stuck:
    cur.execute(
        "UPDATE public.pipeline_runs SET status='failed', "
        "error_detail='Manually reset — stuck in running state', "
        "completed_at=now() "
        "WHERE status IN ('running','pending')"
    )
    conn.commit()
    print(f"\nReset {cur.rowcount} pipeline(s) to failed. You can now start a new one.")
else:
    print("No stuck pipelines found.")

conn.close()
