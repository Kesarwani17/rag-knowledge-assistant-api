"""Seed the RAG knowledge base with official OWASP Cheat Sheet documents."""
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "http://127.0.0.1:8000"
TENANT = "acme_corp"
MAX_WORKERS = max(1, min(3, int(os.getenv("SEED_WORKERS", "2"))))
SOURCE_BASE = "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets"
SOURCE_LICENSE = "https://creativecommons.org/licenses/by-sa/4.0/"

SOURCE_FILES = [
    "Authentication_Cheat_Sheet.md",
    "Authorization_Cheat_Sheet.md",
    "Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.md",
    "Cryptographic_Storage_Cheat_Sheet.md",
    "File_Upload_Cheat_Sheet.md",
    "Input_Validation_Cheat_Sheet.md",
    "Injection_Prevention_Cheat_Sheet.md",
    "JSON_Web_Token_Cheat_Sheet.md",
    "Logging_Cheat_Sheet.md",
    "Password_Storage_Cheat_Sheet.md",
    "Query_Parameterization_Cheat_Sheet.md",
    "RAG_Security_Cheat_Sheet.md",
    "REST_Security_Cheat_Sheet.md",
    "Secrets_Management_Cheat_Sheet.md",
    "Session_Management_Cheat_Sheet.md",
    "Threat_Modeling_Cheat_Sheet.md",
    "Transport_Layer_Security_Cheat_Sheet.md",
]


def load_real_docs() -> list[dict[str, str]]:
    """Download curated source documents from the official OWASP repository."""
    docs = []
    for filename in SOURCE_FILES:
        source_url = f"{SOURCE_BASE}/{filename}"
        response = requests.get(source_url, timeout=30)
        response.raise_for_status()
        title = filename.removesuffix(".md").replace("_", " ")
        content = (
            f"Source: {source_url}\n"
            f"License: OWASP Cheat Sheet Series, CC BY-SA 4.0 ({SOURCE_LICENSE})\n\n"
            f"{response.text}"
        )
        docs.append({"title": title, "content": content})
    return docs

def wait_for(doc_id: int, timeout: int = 180) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status = requests.get(f"{API}/documents/{doc_id}/status").json()["status"]
            if status in ("completed", "failed"):
                return status
        except Exception:
            pass
        time.sleep(2)
    return "timeout"

def process_one(doc: dict, existing_titles: set) -> str:
    if doc["title"] in existing_titles:
        return f"     SKIP: {doc['title']} (already seeded)"
    
    try:
        res = requests.post(f"{API}/documents", json={**doc, "tenant_id": TENANT})
        res.raise_for_status()
        doc_id = res.json()["id"]
        
        requests.post(f"{API}/documents/{doc_id}/process")
        status = wait_for(doc_id)
        return f"{status:>9}: {doc['title']} (id={doc_id})"
    except Exception as e:
        return f"    ERROR: {doc['title']} - {e}"

def main() -> None:
    print("Checking API connection...")
    try:
        docs = load_real_docs()
        print(f"Loaded {len(docs)} official OWASP source documents.")
        res = requests.get(f"{API}/documents")

        if res.status_code != 200:
            print(f"❌ API returned status {res.status_code}")
            print(f"Server response: {res.text[:500]}")
            return

        try:
            existing_titles = {d["title"] for d in res.json()}
        except Exception:
            print("❌ API returned invalid JSON (likely an HTML error page).")
            print(f"Server response: {res.text[:500]}")
            return

        print(f"Found {len(existing_titles)} existing documents. Starting ingestion with {MAX_WORKERS} worker(s)...")

        # Keep the ingestion workload small enough for local PostgreSQL + embedding processing.
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_one, doc, existing_titles) for doc in docs]
            try:
                for future in as_completed(futures):
                    print(future.result())
            except KeyboardInterrupt:
                print("\nSeed interrupted by user. Stopping gracefully after active jobs were drained or canceled.")
                for future in futures:
                    if not future.done():
                        future.cancel()
                raise

    except KeyboardInterrupt:
        print("\nSeeding stopped cleanly. No crash was raised by the application itself.")

if __name__ == "__main__":
    main()