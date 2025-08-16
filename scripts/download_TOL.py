#!/usr/bin/env python3
"""
OpenNeuro Downloader with Rich progress bar and optional 9-minute time-slice.

• Lee un .sh con líneas 'curl -o …', construye árbol y tamaños (MiB/GiB).
• Puede cortarse tras 9 min (540 s) para colas SLURM limitadas.
• Reanuda archivos parciales con cabecera HTTP Range si se relanza.
• Muestra barra global tipo “pip” con Rich.
"""

import re
import pathlib
import requests
import sys
import time
import argparse
from collections import defaultdict

from rich.progress import (
    Progress, BarColumn, DownloadColumn,
    TransferSpeedColumn, TimeRemainingColumn
)
from rich.console import Console

# ───── Configuración de rutas ──────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT_SH    = pathlib.Path(__file__).resolve().parent / "ds003626-2.1.2.sh"
DEST_ROOT    = PROJECT_ROOT / "data" / "raw" / "TOL"
DEST_ROOT.mkdir(parents=True, exist_ok=True)

# ───── Parámetros básicos ──────────────────────────────────────────────
TIME_SLICE_SEC  = 9 * 60
ETA_SPEED_MIB_s = 20
TIMEOUT         = 60
session         = requests.Session()

# ───── CLI para activar o desactivar el time-slice ─────────────────────
parser = argparse.ArgumentParser(
    description="Descarga dataset OpenNeuro con barra de progreso Rich y time-slice opcional de 9 min."
)
parser.add_argument("--full", action="store_true", help="Descarga completa sin cortar tras 9 min")
args = parser.parse_args()
USE_TIME_SLICE = not args.full

# ───── 1. Parsear .sh y extraer URLs/destinos ──────────────────────────
pat = re.compile(r"curl\s+--create-dirs\s+(?P<url>\S+)\s+-o\s+(?P<dest>\S+)")
entries = []
with open(SCRIPT_SH) as f:
    for line in f:
        m = pat.search(line)
        if m:
            entries.append({"url": m["url"], "dest": pathlib.Path(m["dest"])})

if not entries:
    sys.exit("❌  No se encontraron líneas «curl -o» en el script")

print(f"🔍  Detectados {len(entries)} archivos en {SCRIPT_SH}")

# ───── 2. HEAD para tamaños y progreso ya descargado ───────────────────
print("Consultando tamaños remotos…")
total_bytes_required = 0
already_have_bytes   = 0

for e in entries:
    head = session.head(e["url"], allow_redirects=True, timeout=TIMEOUT)
    head.raise_for_status()
    e["size"] = int(head.headers.get("content-length", 0))

    dest_path    = DEST_ROOT / e["dest"]
    on_disk      = dest_path.stat().st_size if dest_path.exists() else 0
    e["on_disk"] = on_disk

    total_bytes_required += e["size"]
    already_have_bytes   += min(on_disk, e["size"])

remaining_bytes = total_bytes_required - already_have_bytes

# ───── 3. Imprimir árbol de previsualización ───────────────────────────
def add(tree, parts, size, done):
    head, *tail = parts
    if tail:
        add(tree.setdefault(head, {}), tail, size, done)
    else:
        tree.setdefault("__files__", []).append((head, size, done))

root = {}
for e in entries:
    add(root, list(e["dest"].parts), e["size"], e["on_disk"] >= e["size"])

def show(node, prefix=""):
    for k in sorted(c for c in node if c != "__files__"):
        print(f"{prefix}{k}/")
        show(node[k], prefix + "    ")
    for fname, sz, done in sorted(node.get("__files__", [])):
        flag = "✓" if done else " "
        print(f"{prefix}    [{flag}] {fname:<40} — {sz/1024/1024:6.2f} MB")

print("\n📂  Previsualización (✓ = completo):\n")
show(root)
print(f"\nTOTAL: {total_bytes_required/1024/1024/1024:.2f} GiB "
      f"(pendiente {remaining_bytes/1024/1024/1024:.2f} GiB)")

eta_min = remaining_bytes / (ETA_SPEED_MIB_s * 1024 * 1024) / 60
print("⏳  ETA aprox. "
      f"{eta_min:.1f} min a {ETA_SPEED_MIB_s} MiB/s "
      + ("(se detendrá tras 9 min)" if USE_TIME_SLICE
         else "(sin límite de tiempo)"))

if input("\n¿Descargar ahora? [y/N] ").strip().lower() != "y":
    sys.exit("❌  Cancelado.")

# ───── 4. Descarga con Rich ProgressBar ────────────────────────────────
console  = Console()
progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(bar_width=None, style="cyan", complete_style="bold cyan", finished_style="green"),
    DownloadColumn(),
    TransferSpeedColumn(),
    TimeRemainingColumn(),
    console=console, transient=False, expand=True
)

task_id = progress.add_task("iniciando…", total=remaining_bytes)
progress.start()
start = time.time()

for e in entries:
    dest_path = DEST_ROOT / e["dest"]
    expected  = e["size"]
    have      = dest_path.stat().st_size if dest_path.exists() else 0

    if have >= expected:
        progress.update(task_id, advance=expected - min(have, expected), description=str(e["dest"])[:50])
        continue

    headers = {"Range": f"bytes={have}-"} if have > 0 else {}
    mode    = "ab" if have > 0 else "wb"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    progress.update(task_id, description=str(e["dest"])[:50])

    with session.get(e["url"], stream=True, headers=headers, timeout=TIMEOUT) as r:
        r.raise_for_status()
        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    progress.update(task_id, advance=len(chunk))
                if USE_TIME_SLICE and time.time() - start >= TIME_SLICE_SEC:
                    progress.stop()
                    print("\n⏸  Tiempo límite alcanzado (9 min). Ejecuta de nuevo para continuar.")
                    sys.exit(0)

    if dest_path.stat().st_size != expected:
        print(f"\n❌  Tamaño incorrecto en {dest_path.name}; se reanudará la próxima vez.")

progress.stop()
(DEST_ROOT / "derivatives").rename(DEST_ROOT / "derivatives_og")
print("\n✅  Todos los archivos descargados o tiempo agotado sin errores.")
