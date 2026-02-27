import json
import requests
from getpass import getpass

url = "https://api.zerogpt.com/api/detect/detectText"

api_key = getpass("ZeroGPT ApiKey (input hidden): ").strip()
text = input("Text to analyze (leave empty for default): ").strip() or "This is a test paragraph"

payload = json.dumps({"input_text": text})
headers = {
    "ApiKey": api_key,
    "Content-Type": "application/json",
}

print("Request URL:", url)
print("Request headers:", {**headers, "ApiKey": "***redacted***"})
print("Request payload bytes:", len(payload.encode("utf-8")))

try:
    response = requests.request("POST", url, headers=headers, data=payload, timeout=(10, 60))
except Exception as e:
    raise SystemExit(f"Request failed: {type(e).__name__}: {e}")

print("\nResponse status:", response.status_code)
print("Response headers:")
for k, v in response.headers.items():
    print(f"  {k}: {v}")

ct = response.headers.get("content-type", "")
print("\nResponse body (first 2000 chars):")
if "application/json" in ct:
    try:
        print(json.dumps(response.json(), indent=2)[:2000])
    except Exception:
        print(response.text[:2000])
else:
    print(response.text[:2000])
