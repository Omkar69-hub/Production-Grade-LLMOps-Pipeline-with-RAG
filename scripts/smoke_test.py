"""Quick end-to-end test: health check → upload → query."""
import json
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"

# 1. Health
req = urllib.request.urlopen(f"{BASE}/health")
print("HEALTH:", json.loads(req.read()))

# 2. Upload via multipart
boundary = "boundary1234"
with open("data/sample.txt", "rb") as f:
    content = f.read()

body = (
    b"--" + boundary.encode() + b"\r\n"
    b"Content-Disposition: form-data; name=\"file\"; filename=\"sample.txt\"\r\n"
    b"Content-Type: text/plain\r\n\r\n"
    + content
    + b"\r\n--" + boundary.encode() + b"--\r\n"
)
req2 = urllib.request.Request(
    f"{BASE}/upload",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
resp = urllib.request.urlopen(req2)
print("UPLOAD:", json.loads(resp.read()))

# Wait for background indexing
print("Waiting 5s for background indexing…")
time.sleep(5)

# 3. List docs
req3 = urllib.request.urlopen(f"{BASE}/documents")
print("DOCUMENTS:", json.loads(req3.read()))

# 4. Ask
payload = json.dumps({"query": "What are the benefits of RAG?"}).encode()
req4 = urllib.request.Request(
    f"{BASE}/ask",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp4 = urllib.request.urlopen(req4)
result = json.loads(resp4.read())
print("ANSWER:", result["answer"][:300])
print("SOURCES:", len(result["source_documents"]), "chunks retrieved")
