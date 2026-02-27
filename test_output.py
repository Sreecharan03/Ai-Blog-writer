import json, os
from google.cloud import storage

uri="gs://ai_blog_02/articles/5ebc3c3a-5e0e-42a5-a350-76f1b792ac15/69898811-3114-4dce-bebb-a7d2bb205b3d/3babc489-aa86-4671-828b-011e7e0495a9/attempt_3/draft_v1/69b518eea69ca48d25a74720184b93b00aa0f7e322e90d1066beec31f8b62311.json"
assert uri.startswith("gs://")
b,o = uri[5:].split("/",1)
c = storage.Client(project=os.getenv("GCP_PROJECT_ID") or None)
data = c.bucket(b).blob(o).download_as_bytes()
obj = json.loads(data.decode("utf-8"))
print("title:", obj.get("draft",{}).get("title"))
print("draft_markdown preview:\n", (obj.get("draft",{}).get("draft_markdown","")[:1200]))
