"""
Download first frame of each SURREAL test MP4 via ffmpeg HTTP streaming.
Credentials from env vars SURREAL_USER and SURREAL_PASSWORD.
"""

import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

URL_LIST = "surreal/download/files/files_cmu_test.mp4.txt"
DEFAULT_OUT = "/mrtstorage/datasets_tmp/surreal"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only first N clips (for testing)")
    return parser.parse_args()


def extract_first_frame(url, out_path, user, password):
    """Download MP4 to temp file, extract first frame as JPEG, delete temp."""
    import tempfile
    import requests

    with requests.get(url, auth=(user, password), timeout=30, stream=True) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            for chunk in r.iter_content(chunk_size=65536):
                tmp.write(chunk)

    try:
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-i", tmp_path,
            "-frames:v", "1", "-update", "1",
            "-q:v", "2",
            "-y", out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode()[:200])
    finally:
        os.remove(tmp_path)


def main():
    args = parse_args()
    user = os.environ.get("SURREAL_USER", "")
    password = os.environ.get("SURREAL_PASSWORD", "")
    if not user or not password:
        raise EnvironmentError("Set SURREAL_USER and SURREAL_PASSWORD env vars")

    os.makedirs(args.output_dir, exist_ok=True)

    with open(URL_LIST) as f:
        urls = [l.strip() for l in f if l.strip()]

    if args.limit:
        urls = urls[:args.limit]

    # Build (url, out_path) pairs, skip already done
    todo = []
    for url in urls:
        clip_name = os.path.splitext(os.path.basename(url))[0]
        out_path = os.path.join(args.output_dir, clip_name + ".jpg")
        if not os.path.exists(out_path):
            todo.append((url, out_path))

    print(f"Total clips: {len(urls)} | Already done: {len(urls) - len(todo)} | Remaining: {len(todo)}")

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(extract_first_frame, url, out, user, password): url
                   for url, out in todo}
        with tqdm(total=len(todo)) as pbar:
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    errors.append((url, str(e)))
                pbar.update(1)

    print(f"Done. {len(todo) - len(errors)} saved, {len(errors)} errors.")
    if errors:
        for url, err in errors[:5]:
            print(f"  ERROR {os.path.basename(url)}: {err}")


if __name__ == "__main__":
    main()
