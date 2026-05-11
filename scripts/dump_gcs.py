"""Dump the first 3000 chars of the latest GCS article markdown for inspection."""
import json, os, sys
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                      r"D:\Hare Krishna_ai_blog\ai-blog-writer-sa-key.json")
from google.cloud import storage

GCS_URI = "gs://ai_blog_02/articles/e8327eb8-4719-4218-bf86-e8005f0e7b20/70b8ff2c-3292-4d52-9f1d-5ad002bc48e0/b193b383-d257-40ed-8f55-ea1732d58f50/attempt_3/draft_v1/ae9141852dbf443a0c768e9388b3e53330e3d077a534f50da561aedaa1bc5afe.json"

bucket_name, obj_path = GCS_URI[5:].split("/", 1)
client = storage.Client()
raw = client.bucket(bucket_name).blob(obj_path).download_as_text()
data = json.loads(raw)
draft = data.get("draft") or {}
md = (data.get("draft_markdown") or draft.get("draft_markdown") or "").strip()

print(f"Total chars: {len(md)}")
print(f"Keys in JSON: {list(data.keys())}")
print()
# Show first 4000 chars with explicit newline markers
preview = md[:4000]
for i, line in enumerate(preview.split('\n')):
    print(f"L{i:03d}: {repr(line) if len(line) > 100 else line}")
