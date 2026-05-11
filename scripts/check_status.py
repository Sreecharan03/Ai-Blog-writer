import psycopg2
conn = psycopg2.connect(
    host="aws-1-ap-southeast-2.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.dppacvjatqcbolulzxud",
    password="Aiblog@2026", sslmode="require", connect_timeout=8
)
cur = conn.cursor()
cur.execute("SELECT DISTINCT status FROM public.article_requests ORDER BY status")
print("Valid statuses:", [r[0] for r in cur.fetchall()])
cur.execute("SELECT request_id, title, status, gcs_draft_uri, created_at FROM public.article_requests ORDER BY created_at DESC LIMIT 5")
for r in cur.fetchall():
    print(r)
conn.close()
