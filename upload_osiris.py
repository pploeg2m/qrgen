#!/usr/bin/env python3
"""Upload a local course tree to osiris via the Jupyter contents API.

  python upload_osiris.py <local_root> <remote_path> <token>
e.g. upload data4/osiris_course/OCR-Klausur-WiMa-SS26 -> courses/OCR/OCR-Klausur-WiMa-SS26
"""
import base64, json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://osiris.nrw/user/cid8683jgf/api/contents"
LOCAL = Path(sys.argv[1]); REMOTE = sys.argv[2].strip("/"); TOK = sys.argv[3]
TEXT_EXT = {".py", ".csv", ".txt", ".ipynb", ".md", ".json"}


def put(remote_path, body):
    req = urllib.request.Request(
        f"{BASE}/{remote_path}", data=json.dumps(body).encode(),
        method="PUT", headers={"Authorization": f"token {TOK}",
                               "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status < 400
        except urllib.error.HTTPError as e:
            if e.code in (409, 400) and attempt < 3:
                time.sleep(0.5); continue
            if attempt == 3:
                return f"HTTP {e.code}: {e.read()[:120]}"
            time.sleep(0.6)
        except Exception as e:
            if attempt == 3:
                return str(e)[:120]
            time.sleep(0.8)
    return False


def mkdir(rel):
    return put(rel, {"type": "directory"})


def upload_file(local: Path, rel: str):
    if local.suffix in TEXT_EXT:
        body = {"type": ("notebook" if local.suffix == ".ipynb" else "file"),
                "format": ("json" if local.suffix == ".ipynb" else "text"),
                "content": (json.loads(local.read_text(encoding="utf-8"))
                            if local.suffix == ".ipynb"
                            else local.read_text(encoding="utf-8"))}
    else:
        body = {"type": "file", "format": "base64",
                "content": base64.b64encode(local.read_bytes()).decode()}
    return put(rel, body)


def main():
    # 1. directories, shallow → deep
    dirs = sorted({p.relative_to(LOCAL.parent) for p in LOCAL.rglob("*") if p.is_dir()}
                  | {LOCAL.relative_to(LOCAL.parent)},
                  key=lambda p: len(p.parts))
    root_prefix = "/".join(REMOTE.split("/")[:-1])
    def remote_of(rel):  # rel is relative to LOCAL.parent (starts with LOCAL.name)
        parts = list(rel.parts); parts[0] = REMOTE.split("/")[-1]
        return (root_prefix + "/" + "/".join(parts)).strip("/")
    print(f"creating {len(dirs)} dirs …")
    for d in dirs:
        r = mkdir(remote_of(d))
        if r is not True:
            print("  dir fail", remote_of(d), r)
    # 2. files, concurrent
    files = [p for p in LOCAL.rglob("*") if p.is_file()]
    print(f"uploading {len(files)} files …")
    ok = 0; fails = []
    def work(p):
        return p, upload_file(p, remote_of(p.relative_to(LOCAL.parent)))
    with ThreadPoolExecutor(max_workers=10) as ex:
        for i, (p, res) in enumerate(ex.map(work, files), 1):
            if res is True:
                ok += 1
            else:
                fails.append((str(p), res))
            if i % 200 == 0:
                print(f"  {i}/{len(files)} (ok={ok}, fail={len(fails)})")
    print(f"\nDONE: {ok}/{len(files)} uploaded, {len(fails)} failed")
    for f, r in fails[:20]:
        print("  FAIL", f, r)


if __name__ == "__main__":
    main()
