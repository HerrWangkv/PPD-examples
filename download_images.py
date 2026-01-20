import os
import hashlib
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PARQUET_PATH = "data/photo-concept-bucket/photo-concept-bucket.parquet"
OUT_DIR = "/mrtstorage/users/kwang/photo-concept-bucket/"
NUM_WORKERS = 32
TIMEOUT = 10

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(PARQUET_PATH)

def filename_from_url(url):
    h = hashlib.sha1(url.encode()).hexdigest()
    return f"{h}.jpg"

def download(row):
    url = row["url"]
    fname = filename_from_url(url)
    path = os.path.join(OUT_DIR, fname)

    if os.path.exists(path):
        return True

    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass

    return False


with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(download, row) for _, row in df.iterrows()]
    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass
