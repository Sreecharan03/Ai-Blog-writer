import psycopg2
conn = psycopg2.connect(host="aws-1-ap-southeast-2.pooler.supabase.com", port=6543,
    dbname="postgres", user="postgres.dppacvjatqcbolulzxud",
    password="Aiblog@2026", sslmode="require", connect_timeout=8)
cur = conn.cursor()
cur.execute("SELECT kb_id, name, description FROM public.knowledge_bases ORDER BY created_at")
for r in cur.fetchall():
    print(f"kb_id: {r[0]}\nname: {r[1]}\ndesc: {r[2]}\n")
conn.close()
