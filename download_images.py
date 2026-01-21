import os
import hashlib
import requests
import pandas as pd
import warnings
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Settings ---
PARQUET_PATH = "data/photo-concept-bucket.parquet"
CACHE_DIR = Path("data/images")
NUM_WORKERS = 16  # Reduced slightly to avoid 429s
TIMEOUT = 10

CACHE_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(PARQUET_PATH).dropna(subset=["url"])

# Setup a session with automatic exponential backoff for 429 errors
session = requests.Session()
retries = Retry(
    total=5, 
    backoff_factor=2, # Waits 2s, 4s, 8s, 16s, 32s between retries
    status_forcelist=[429, 500, 502, 503, 504],
    raise_on_status=False
)
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

def get_cache_path(url, cache_dir):
    """Your provided logic for generating paths"""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = "jpg"
    try:
        url_clean = url.split('?')[0]
        if '.' in url_clean:
            potential_ext = url_clean.split('.')[-1].lower()
            if potential_ext in ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp']:
                ext = potential_ext
    except:
        pass
    return cache_dir / f"{url_hash}.{ext}"

# 1. Map expected paths to URLs
print("Mapping dataset...")
expected_paths_to_urls = {str(get_cache_path(u, CACHE_DIR)): u for u in df["url"]}
expected_paths_set = set(expected_paths_to_urls.keys())

# 2. Cleanup: Remove extra, wrong, or corrupted files
print("Cleaning folder...")
for file_path in CACHE_DIR.glob("*"):
    str_path = str(file_path)
    if str_path not in expected_paths_set:
        print(f"Removing (Not in dataset/Wrong extension): {file_path.name}")
        file_path.unlink()
    else:
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception:
            print(f"Removing (Corrupted): {file_path.name}")
            file_path.unlink()

# 3. Synchronized Download with Retry Logic
def download_missing(url):
    target_path = get_cache_path(url, CACHE_DIR)
    if target_path.exists():
        return True

    try:
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            # Verify it's actually an image before saving
            Image.open(BytesIO(r.content)).verify()
            with open(target_path, "wb") as f:
                f.write(r.content)
            return True
        elif r.status_code == 429:
            warnings.warn(f"Rate limited (429) after retries: {url}")
        else:
            warnings.warn(f"Access failed (Status {r.status_code}): {url}")
    except Exception as e:
        warnings.warn(f"Error downloading {url}: {e}")
    
    return False

print("Downloading missing images...")
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(download_missing, url) for url in expected_paths_to_urls.values()]
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass