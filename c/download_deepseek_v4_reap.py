#!/usr/bin/env python3
"""Resumable sequential downloader for DeepSeek-V4-Flash-0731-reap-150b into Colibrì.
Downloads files directly into the target directory without duplicating cache.
"""

import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "puwaer/DeepSeek-V4-Flash-0731-reap-150b"
TARGET_DIR = Path("/home/upc/every_thing_claude/colibri/c/deepseek_v4_reap150b")

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    
    print(f"Fetching file list for {REPO_ID}...", flush=True)
    repo_files = api.list_repo_files(REPO_ID)
    
    # Priority order: metadata first, then safetensors shards
    meta_files = [f for f in repo_files if not f.endswith('.safetensors') and not f.startswith('.')]
    shards = sorted([f for f in repo_files if f.endswith('.safetensors')])
    
    all_files = meta_files + shards
    total_files = len(all_files)
    print(f"Total files to verify/download: {total_files} ({len(shards)} shards)", flush=True)
    
    for idx, filename in enumerate(all_files, start=1):
        target_file = TARGET_DIR / filename
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{idx}/{total_files}] Processing: {filename}...", flush=True)
        start_time = time.time()
        
        # Download with retry
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                dest = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    local_dir=str(TARGET_DIR),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                elapsed = time.time() - start_time
                size_mb = Path(dest).stat().st_size / (1024 * 1024)
                print(f"✓ [{idx}/{total_files}] Done: {filename} ({size_mb:.1f} MB in {elapsed:.1f}s)", flush=True)
                break
            except Exception as e:
                print(f"⚠️ Error downloading {filename} (attempt {attempt}/{max_retries}): {e}", flush=True)
                if attempt == max_retries:
                    print(f"❌ Failed to download {filename} after {max_retries} attempts.", flush=True)
                    return 1
                time.sleep(5)
                
    print("\n🎉 ALL FILES DOWNLOADED SUCCESSFULLY!", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
