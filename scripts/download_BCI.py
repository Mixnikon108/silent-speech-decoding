#!/usr/bin/env python3
import pathlib, requests, sys
from collections import defaultdict
from tqdm import tqdm

TRACK3_ID = "5e947fd8f1353503a7d55758"
ROOT_API  = f"https://api.osf.io/v2/nodes/pq7vb/files/osfstorage/{TRACK3_ID}/"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent      # …/proyecto/
DEST_DIR     = PROJECT_ROOT / "data" / "raw" / "BCI2020"
DEST_DIR.mkdir(parents=True, exist_ok=True)


session = requests.Session()

def get(url):
    r = session.get(url, timeout=30) 
    r.raise_for_status()
    return r.json()

def paged_list(url):
    while url:
        data = get(url)
        for item in data["data"]:
            yield item
        url = data["links"].get("next")

def walk(folder_url):
    for item in paged_list(folder_url):
        if item["attributes"]["kind"] == "file" and item["attributes"]["name"].endswith(".mat"):
            yield {
                "remote_path": item["attributes"]["materialized_path"],
                "download_url": item["links"]["download"],
                "expected_size": int(item["attributes"].get("size", 0))
            }
        elif item["attributes"]["kind"] == "folder":
            sub = item["relationships"]["files"]["links"]["related"]["href"]
            yield from walk(sub)

def remote2local(rpath: str) -> pathlib.Path:
    parts = rpath.strip("/").split("/")[1:]
    parts = [p.replace(" ", "_").lower() for p in parts]
    return DEST_DIR.joinpath(*parts)

def preview(files):
    tree = defaultdict(list)
    for entry in files:
        r = entry["remote_path"]
        rel = remote2local(r).relative_to(DEST_DIR)
        tree[rel.parts[0]].append((rel.name, entry["expected_size"]))
    print("\nPrevisualización:\n")
    for folder in sorted(tree):
        print(f"{folder}/")
        for f, size in sorted(tree[folder]):
            size_mb = size / (1024 * 1024)
            print(f"    {f} — {size_mb:.2f} MB")
    print()

def download(entry, bar: tqdm):
    url  = entry["download_url"]
    dest = remote2local(entry["remote_path"])
    size = entry["expected_size"]

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(url, stream=True, timeout=60); r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
                bar.set_description(f"[{entry['remote_path'].split('/')[-2].lower()}] {dest.name[:35]}")

    # Verificación post-descarga
    actual = dest.stat().st_size
    if actual != size:
        print(f"\n❌ Archivo corrupto (esperado {size}, recibido {actual}): {dest}")
        dest.unlink()

# ────────────── MAIN ───────────────────
if __name__ == "__main__":
    print("Buscando archivos .mat en Track #3…")
    files = list(walk(ROOT_API))
    print(f"🔍 {len(files)} archivos encontrados\n")

    preview(files)
    if input("¿Descargar? [y/N] ").strip().lower() != "y":
        sys.exit("❌ Cancelado.")

    total_bytes = sum(f["expected_size"] for f in files if not remote2local(f["remote_path"]).exists())

    with tqdm(total=total_bytes, unit="B", unit_scale=True, ascii=True) as bar:
        for i, entry in enumerate(files, 1):
            local = remote2local(entry["remote_path"])
            if local.exists():
                bar.update(entry["expected_size"])
                continue
            download(entry, bar)

    print("\n✅ Descarga completa y verificada.")