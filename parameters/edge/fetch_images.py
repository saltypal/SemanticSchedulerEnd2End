"""Image fetchers for the Kaggle dataset (stdlib + Pillow only, network injectable).

``opener(url, headers) -> bytes`` is the only network seam; tests pass a fake. Fetchers are
resumable (existing non-empty files are skipped), validate every image with Pillow, retry with
exponential backoff on 429/5xx/network errors and return failures instead of raising.

* CrisisMMD: HuggingFace pinned-commit URL per ``image_path``.
* Places365: ``val_256.tar`` downloaded once (Range-resume), only selected members extracted.
* Open Images: OriginalURL then Thumbnail300KURL; re-encoded to JPEG q95, longest side <= 1600.
"""
import io
import json
import os
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CRISISMMD_URL = ("https://huggingface.co/datasets/QCRI/CrisisMMD/resolve/"
                 "10a5626ba112ab9c50369c473a2062bcf205d708/%s")
PLACES_TAR_URL = "https://data.csail.mit.edu/places/places365/val_256.tar"
MAX_SIDE = 1600
JPEG_QUALITY = 95
_UA = {"User-Agent": "scheduler-dataset-builder/1.0"}


def default_opener(url, headers=None):
    req = urllib.request.Request(url, headers={**_UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def check_image(data):
    """Raise ValueError unless ``data`` is a decodable image (rejects HTML error pages)."""
    from PIL import Image
    if not data:
        raise ValueError("empty payload")
    if data.lstrip()[:1] == b"<":
        raise ValueError("HTML/XML payload, not an image")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            return im.format, im.size
    except Exception as exc:
        raise ValueError("not a decodable image: %s" % exc) from exc


def _retryable(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, OSError, TimeoutError, ConnectionError))


def get_with_retry(opener, url, retries=4, base_delay=1.0, headers=None, sleep=time.sleep):
    """Call opener; exponential backoff on 429/5xx/network errors; other errors raise at once."""
    for attempt in range(retries + 1):
        try:
            return opener(url, headers or {})
        except Exception as exc:
            if attempt == retries or not _retryable(exc):
                raise
            sleep(base_delay * (2 ** attempt))


def _write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _flush_progress(progress_path, done, failures, total):
    if progress_path:
        Path(progress_path).write_text(json.dumps(
            {"done": done, "total": total, "failed": len(failures), "failures": failures[-50:]}),
            encoding="utf-8")


def _map_threads(fn, items, workers, progress_path=None, every=200):
    """Run fn(item)->failure-or-None over a thread pool; returns list of failures."""
    failures, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for failure in pool.map(fn, items):
            done += 1
            if failure:
                failures.append(failure)
            if done % every == 0:
                _flush_progress(progress_path, done, failures, len(items))
    _flush_progress(progress_path, done, failures, len(items))
    return failures


def _ok_existing(path):
    return path.is_file() and path.stat().st_size > 0


def fetch_crisismmd(rows, out_dir, opener=default_opener, workers=8, retries=4, base_delay=1.0,
                    sleep=time.sleep, progress_path=None):
    """Download ``images/crisismmd/<image_path>`` for each row. Returns failure dicts."""
    root = Path(out_dir) / "images" / "crisismmd"

    def one(row):
        rel = row["image_path"]
        target = root / rel
        if _ok_existing(target):
            return None
        try:
            data = get_with_retry(opener, CRISISMMD_URL % rel, retries, base_delay, sleep=sleep)
            check_image(data)
            _write_atomic(target, data)
            return None
        except Exception as exc:
            return {"source": "crisismmd", "key": rel, "reason": str(exc)[:200]}
    return _map_threads(one, list(rows), workers, progress_path)


def download_with_resume(dest, url=PLACES_TAR_URL, retries=8, base_delay=2.0, sleep=time.sleep,
                         chunk=1 << 20, urlopen=None):
    """Stream ``url`` to ``dest``; resume a partial file with an HTTP Range request."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    urlopen = urlopen or (lambda req: urllib.request.urlopen(req, timeout=60))
    for attempt in range(retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = dict(_UA)
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            with urlopen(urllib.request.Request(url, headers=headers)) as resp:
                status = getattr(resp, "status", 200)
                if have and status != 206:        # server ignored Range: restart
                    have = 0
                with open(dest, "ab" if have else "wb") as out:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        out.write(block)
            return dest
        except urllib.error.HTTPError as exc:
            if exc.code == 416:                    # already complete
                return dest
            if attempt == retries or not _retryable(exc):
                raise
        except Exception as exc:
            if attempt == retries or not _retryable(exc):
                raise
        sleep(base_delay * (2 ** min(attempt, 6)))
    return dest


def fetch_places365(items, out_dir, tar_source, keep=False):
    """Extract only the selected ``items`` (dicts with ``id`` = file name) from val_256.tar.

    ``tar_source``: path of an existing tar, or a callable ``dest_path -> None`` that downloads
    it (e.g. ``download_with_resume``). Returns failure dicts for items missing from the tar."""
    root = Path(out_dir) / "images" / "places365"
    root.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not _ok_existing(root / it["id"])]
    failures = []
    if todo:
        downloaded = False
        if callable(tar_source):
            tar_path = Path(out_dir).parent / "_val_256.tar.part"
            tar_source(tar_path)
            downloaded = True
        else:
            tar_path = Path(tar_source)
        wanted = {"val_256/" + it["id"]: it for it in todo}
        seen = set()
        with tarfile.open(tar_path) as tar:
            for member in tar:
                if member.name in wanted:
                    data = tar.extractfile(member).read()
                    try:
                        check_image(data)
                        _write_atomic(root / wanted[member.name]["id"], data)
                        seen.add(member.name)
                    except ValueError as exc:
                        failures.append({"source": "places365", "key": wanted[member.name]["id"],
                                         "reason": str(exc)[:200]})
                        seen.add(member.name)
                    if len(seen) == len(wanted):
                        break
        for name, it in wanted.items():
            if name not in seen:
                failures.append({"source": "places365", "key": it["id"], "reason": "missing from tar"})
        if downloaded and not keep:
            tar_path.unlink(missing_ok=True)
    return failures


def normalize_jpeg(data):
    """-> (jpeg_bytes, original_size). Re-encodes unless already JPEG, RGB and within MAX_SIDE."""
    from PIL import Image
    fmt, size = check_image(data)
    if fmt == "JPEG" and max(size) <= MAX_SIDE:
        with Image.open(io.BytesIO(data)) as im:
            if im.mode == "RGB":
                return data, size
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        if max(im.size) > MAX_SIDE:
            scale = MAX_SIDE / max(im.size)
            im = im.resize((max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))),
                           Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue(), size


def fetch_openimages_one(item, out_dir, opener=default_opener, thumb_urls=None, retries=4,
                         base_delay=1.0, sleep=time.sleep):
    """Fetch one Open Images item. Returns ``{"id", "orig_size", "url_used"}`` or ``None``;
    on failure ``item["_failure"]`` is set (thread-safe: per-item dict)."""
    target = Path(out_dir) / "images" / "openimages" / ("%s.jpg" % item["id"])
    if _ok_existing(target):
        return {"id": item["id"], "orig_size": item.get("orig_size"), "url_used": "cached"}
    urls = [u for u in (item.get("url"), (thumb_urls or {}).get(item["id"])) if u]
    errors = []
    for url in urls:
        try:
            data = get_with_retry(opener, url, retries, base_delay, sleep=sleep)
            jpeg, orig = normalize_jpeg(data)
            _write_atomic(target, jpeg)
            return {"id": item["id"], "orig_size": list(orig), "url_used": url}
        except Exception as exc:
            errors.append("%s: %s" % (url, str(exc)[:120]))
    item["_failure"] = {"source": "openimages_people", "key": item["id"],
                        "reason": "; ".join(errors) or "no url"}
    return None


def fetch_openimages(items, out_dir, opener=default_opener, thumb_urls=None, workers=8, retries=4,
                     base_delay=1.0, sleep=time.sleep, progress_path=None):
    """Fetch all items. Returns (results_by_id, failures)."""
    results = {}

    def one(item):
        res = fetch_openimages_one(item, out_dir, opener, thumb_urls, retries, base_delay, sleep)
        if res:
            results[item["id"]] = res
            return None
        return item.pop("_failure")
    failures = _map_threads(one, list(items), workers, progress_path)
    return results, failures


def fill_to_count(candidates, n, fetch_fn, workers=8):
    """Take candidates in order until ``n`` succeed. ``fetch_fn(item) -> result|None``.

    Returns (successes [(item, result)] in candidate order, failed_items)."""
    successes, failed, pos = [], [], 0
    cands = list(candidates)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while len(successes) < n and pos < len(cands):
            chunk = cands[pos:pos + (n - len(successes))]
            pos += len(chunk)
            for item, res in zip(chunk, pool.map(fetch_fn, chunk)):
                if res is None:
                    failed.append(item)
                else:
                    successes.append((item, res))
    return successes, failed


def verify_images(rows, root):
    """Return (missing, corrupt) lists of row ``file`` paths under ``root``."""
    missing, corrupt = [], []
    for row in rows:
        path = Path(root) / row["file"]
        if not _ok_existing(path):
            missing.append(row["file"])
            continue
        try:
            check_image(path.read_bytes())
        except ValueError:
            corrupt.append(row["file"])
    return missing, corrupt
