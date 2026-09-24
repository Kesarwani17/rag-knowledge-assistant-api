"""Seed the RAG knowledge base with official OWASP Cheat Sheet documents."""
import os
import sys
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


def _progress_bar(current: int, total: int, prefix: str = "", width: int = 40) -> None:
    """Print a simple progress bar to stdout."""
    if total == 0:
        return
    pct = current / total
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {current}/{total} ({pct:.0%})")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def load_real_docs() -> list[dict[str, str]]:
    """Download curated source documents from the official OWASP repository."""
    docs = []
    total = len(SOURCE_FILES)
    print(f"[DOWNLOAD] Downloading {total} OWASP Cheat Sheets...")
    for i, filename in enumerate(SOURCE_FILES, 1):
        source_url = f"{SOURCE_BASE}/{filename}"
        _progress_bar(i, total, prefix="  Downloading")
        response = requests.get(source_url, timeout=30)
        response.raise_for_status()
        title = filename.removesuffix(".md").replace("_", " ")
        content = (
            f"Source: {source_url}\n"
            f"License: OWASP Cheat Sheet Series, CC BY-SA 4.0 ({SOURCE_LICENSE})\n\n"
            f"{response.text}"
        )
        docs.append({"title": title, "content": content})
    print(f"[OK] Downloaded {len(docs)} documents")
    return docs

def wait_for(doc_id: int, timeout: int = 180) -> str:
    deadline = time.time() + timeout
    last_status = ""
    print(f"  [WAIT] Waiting for doc {doc_id} to complete...", end="", flush=True)
    while time.time() < deadline:
        try:
            status = requests.get(f"{API}/documents/{doc_id}/status").json()["status"]
            if status != last_status:
                print(f" {status}", end="", flush=True)
                last_status = status
            if status in ("completed", "failed"):
                print(" [OK]")
                return status
        except Exception:
            print(".", end="", flush=True)
        time.sleep(2)
    print(" [TIMEOUT]")
    return "timeout"

def process_one(doc: dict, existing_titles: set) -> str:
    if doc["title"] in existing_titles:
        return f"     SKIP: {doc['title']} (already seeded)"
    
    try:
        res = requests.post(f"{API}/documents", json={**doc, "tenant_id": TENANT})
        res.raise_for_status()
        doc_id = res.json()["id"]
        
        process_response = requests.post(f"{API}/documents/{doc_id}/process")
        process_response.raise_for_status()
        status = wait_for(doc_id)
        return f"{status:>9}: {doc['title']} (id={doc_id})"
    except Exception as e:
        return f"    ERROR: {doc['title']} - {e}"

def main() -> None:
    print("[LINK] Checking API connection...")
    try:
        res = requests.get(f"{API}/health", timeout=5)
        if res.status_code != 200:
            print(f"❌ API health check failed: {res.status_code}")
            return
        print("[OK] API is reachable")

        docs = load_real_docs()
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

        new_docs = [d for d in docs if d["title"] not in existing_titles]
        skipped = len(docs) - len(new_docs)
        print(f"[STATS] Found {len(existing_titles)} existing, {skipped} skipped, {len(new_docs)} to process")
        if not new_docs:
            print("[OK] Nothing new to seed")
            return

        print(f"[START] Starting ingestion with {MAX_WORKERS} worker(s)...\n")

        # Keep the ingestion workload small enough for local PostgreSQL + embedding processing.
        completed = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_one, doc, existing_titles): doc for doc in new_docs}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    completed += 1
                    if "ERROR" in result or "failed" in result.lower():
                        failed += 1
                    print(f"  [{completed}/{len(new_docs)}] {result}")
            except KeyboardInterrupt:
                print("\n[WARN]  Seed interrupted by user. Stopping gracefully...")
                for future in futures:
                    if not future.done():
                        future.cancel()
                raise

        print(f"\n[DONE] Seeding complete: {completed - failed} succeeded, {failed} failed, {skipped} skipped")

    except KeyboardInterrupt:
        print("\n[STOP] Seeding stopped cleanly. No crash was raised by the application itself.")

if __name__ == "__main__":
    main()