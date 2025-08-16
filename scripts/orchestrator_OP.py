#!/usr/bin/env python3
"""
Orquestador de experimentos (SLURM + srun) para Silent Speech

- Crea datasets con etiqueta ORCH via scripts/preprocess_BCI_mod_OP.py
- Lanza Diff-E (DiffE_AT | DiffE_mod | DiffE_TF) y Other Models con main_OP*.py
- Genera un script sbatch por experimento y (opcionalmente) lo envía
- Guarda logs y resultados en <project>/results/
- Agrega todas las líneas [RESULT_JSON] a results/all_results.jsonl

Formas de elegir combinaciones:
1) Config file (JSON/YAML) con grids (recomendado):
   python scripts/orchestrator_OP.py --config configs/experimentos.yaml --submit
2) Flags rápidos (listas separadas por comas):
   --pre-grid bandpass=1 notch=1 car=1,0 ica=0 baseline=1,0 residual=0 l_freq=0.05,1.0 h_freq=,40
   --diffe variants=TF,AT epochs=50 alpha=0.1,1.0 encoder_dim=256 fc_dim=512 ddpm_dim=128 n_T=1000
   --other models=eegnet,deepconvnet epochs=100

Requisitos:
- Estructura de proyecto:
  scripts/preprocess_BCI_mod_OP.py
  models/DiffE_TF/main_OP.py  (o DiffE_mod/main_OP.py, DiffE_AT/main_OP.py)
  models/otherModels/main_OP_other.py

Autor: tú + ChatGPT
"""
from __future__ import annotations

import argparse, itertools, json, os, re, shlex, subprocess, sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- Paths por defecto ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
DATA_PROC    = PROJECT_ROOT / "data" / "processed" / "BCI2020"

# ---------- Mapas de ejecutables ----------
DIFFE_SCRIPTS = {
    "TF":  MODELS_DIR / "DiffE_TF"  / "main_OP.py",
    "MOD": MODELS_DIR / "DiffE_mod" / "main_OP.py",
    "AT":  MODELS_DIR / "DiffE_AT"  / "main_OP.py",
}
OTHER_SCRIPT  = MODELS_DIR / "otherModels" / "main_OP_other.py"
PREPROC_SCRIPT = SCRIPTS_DIR / "preprocess_BCI_mod_OP.py"

# ---------- Utilidades ----------
def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def short_hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_bool(v: str) -> Optional[bool]:
    if v in ("", "None", "null"): return None
    return v.lower() in ("1","true","t","yes","y")

def parse_kv_list(kvs: List[str]) -> Dict[str, List[str]]:
    """
    Convierte ["a=1,2", "b=,40", "flag=1"] -> {"a":["1","2"], "b":["", "40"], "flag":["1"]}
    """
    out: Dict[str, List[str]] = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Argumento inválido (esperado k=v1,v2,...): {kv}")
        k, v = kv.split("=", 1)
        out[k.strip()] = [x.strip() for x in v.split(",")] if v != "" else [""]
    return out

def try_load_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path: return None
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(p)
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # opcional
        except Exception as e:
            raise RuntimeError("Instala PyYAML o usa .json") from e
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

# ---------- Estructuras ----------
@dataclass
class SlurmSpec:
    job_name: str
    partition: str = "standard-gpu"
    gres: str = "gpu:a100:1"
    ntasks: int = 1
    cpus_per_task: int = 4
    mem: str = "16G"
    time: str = "06:00:00"
    account: Optional[str] = None
    qos: Optional[str] = None
    extra: List[str] = field(default_factory=list)

@dataclass
class PreprocCfg:
    preprocess: bool = True
    enable_bandpass: bool = True
    l_freq: float = 0.05
    h_freq: Optional[float] = None
    enable_notch: bool = True
    enable_car: bool = True
    enable_ica: bool = False
    enable_baseline: bool = True
    residual: bool = False

    def tag(self) -> str:
        # ORCH + minitag reproducible
        parts = [
            "ORCH",
            f"BP{int(self.enable_bandpass)}",
            f"N{int(self.enable_notch)}",
            f"CAR{int(self.enable_car)}",
            f"ICA{int(self.enable_ica)}",
            f"BL{int(self.enable_baseline)}",
            f"R{int(self.residual)}",
            f"L{self.l_freq:g}",
            f"H{('None' if self.h_freq is None else str(self.h_freq))}",
        ]
        return "_".join(parts)

    def to_preproc_cli(self, out_npz: Path) -> str:
        args = [
            f"-o {shlex.quote(str(out_npz))}",
            f"--{'preprocess' if self.preprocess else 'no-preprocess'}",
            f"--{'bandpass' if self.enable_bandpass else 'no-bandpass'}",
            f"--{'notch' if self.enable_notch else 'no-notch'}",
            f"--{'car' if self.enable_car else 'no-car'}",
            f"--{'ica' if self.enable_ica else 'no-ica'}",
            f"--{'baseline' if self.enable_baseline else 'no-baseline'}",
            f"--l-freq {self.l_freq}",
            f"--h-freq {self.h_freq}" if self.h_freq is not None else "",
            f"--residual" if self.residual else "",
        ]
        return " ".join(a for a in args if a)

@dataclass
class DiffEGrid:
    variants: List[str] = field(default_factory=lambda: ["TF"])
    num_epochs: List[int] = field(default_factory=lambda: [100])
    batch_train: List[int] = field(default_factory=lambda: [250])
    batch_eval: List[int] = field(default_factory=lambda: [32])
    seed: List[int] = field(default_factory=lambda: [42])
    alpha: List[float] = field(default_factory=lambda: [0.1])
    num_classes: int = 5
    channels: int = 64
    subject_id: Optional[int] = None
    n_T: List[int] = field(default_factory=lambda: [1000])
    ddpm_dim: List[int] = field(default_factory=lambda: [128])
    encoder_dim: List[int] = field(default_factory=lambda: [256])
    fc_dim: List[int] = field(default_factory=lambda: [512])

@dataclass
class OtherGrid:
    models: List[List[str]] = field(default_factory=lambda: [["eegnet","shallowconvnet","deepconvnet"]])
    epochs: List[int] = field(default_factory=lambda: [150])
    lr: List[float] = field(default_factory=lambda: [1e-3])
    weight_decay: List[float] = field(default_factory=lambda: [0.0])
    batch_train: List[int] = field(default_factory=lambda: [64])
    batch_eval: List[int] = field(default_factory=lambda: [32])
    seed: List[int] = field(default_factory=lambda: [42])
    n_channels: int = 64
    n_classes: int = 5
    subject_id: Optional[int] = None

# ---------- Generadores de combinaciones ----------
def expand_preproc_grid(grid: Dict[str, List[Any]]) -> List[PreprocCfg]:
    # defaults si algo falta
    keys = ["preprocess","enable_bandpass","l_freq","h_freq","enable_notch",
            "enable_car","enable_ica","enable_baseline","residual"]
    values = [grid.get(k, [PreprocCfg().__dict__[k]]) for k in keys]
    combos = []
    for vals in itertools.product(*values):
        kwargs = dict(zip(keys, vals))
        combos.append(PreprocCfg(**kwargs))
    return combos

def expand_diffe_grid(g: DiffEGrid):
    for variant, ne, bt, be, seed, alpha, nT, dd, ed, fd in itertools.product(
        g.variants, g.num_epochs, g.batch_train, g.batch_eval, g.seed, g.alpha, g.n_T, g.ddpm_dim, g.encoder_dim, g.fc_dim
    ):
        combo = {
            "variant": variant,
            "num_epochs": ne,
            "batch_train": bt,
            "batch_eval": be,
            "seed": seed,
            "alpha": alpha,
            "n_T": nT,
            "ddpm_dim": dd,
            "encoder_dim": ed,
            "fc_dim": fd,
            # siempre propagamos estos:
            "channels": g.channels,
            "num_classes": g.num_classes,
        }
        # subject_id solo si se define (None => no pasar a srun)
        if g.subject_id is not None:
            combo["subject_id"] = g.subject_id
        yield combo

def expand_other_grid(g: OtherGrid):
    for models, ne, lr, wd, bt, be, seed in itertools.product(
        g.models, g.epochs, g.lr, g.weight_decay, g.batch_train, g.batch_eval, g.seed
    ):
        combo = {
            "models": models,
            "epochs": ne,
            "lr": lr,
            "weight_decay": wd,
            "batch_train": bt,
            "batch_eval": be,
            "seed": seed,
            # siempre propagamos estos:
            "n_channels": g.n_channels,
            "n_classes": g.n_classes,
        }
        # subject_id solo si se define (None => no pasar a srun)
        if g.subject_id is not None:
            combo["subject_id"] = g.subject_id
        yield combo

# ---------- Constructor de sbatch ----------
def build_sbatch(
    job: SlurmSpec,
    venv_activate: str,
    project_root: Path,
    lines: List[str],
    out_dir: Path,
) -> str:
    out_dir = ensure_dir(out_dir)
    log_out = out_dir / "slurm.out"
    log_err = out_dir / "slurm.err"

    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job.job_name}",
        f"#SBATCH --partition={job.partition}",
        f"#SBATCH --gres={job.gres}",
        f"#SBATCH --ntasks={job.ntasks}",
        f"#SBATCH --cpus-per-task={job.cpus_per_task}",
        f"#SBATCH --mem={job.mem}",
        f"#SBATCH --time={job.time}",
        f"#SBATCH --output={log_out}",
        f"#SBATCH --error={log_err}",
    ]
    if job.account: header.append(f"#SBATCH --account={job.account}")
    if job.qos:     header.append(f"#SBATCH --qos={job.qos}")
    header.extend(job.extra)

    body = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(project_root))}",
        f"source {shlex.quote(venv_activate)}",
        "export PYTHONUNBUFFERED=1",
        "export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK",
        "",
    ]

    # añadimos las líneas de ejecución (cada una con tee para capturar logs)
    body.extend(lines)

    return "\n".join(header + ["##---------------------"] + body) + "\n"

# ---------- Orquestación ----------
def main():
    ap = argparse.ArgumentParser(description="Orquestador SLURM para preprocesado + modelos")
    ap.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    ap.add_argument("--venv-activate", type=str, required=True,
                    help="Ruta al 'bin/activate' del entorno (e.g., /.../venv/bin/activate)")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dataset", type=str, default="BCI", choices=["BCI","TOL"])

    # selección por config
    ap.add_argument("--config", type=str, default=None, help="YAML/JSON con grids")

    # flags rápidos (opcionales)
    ap.add_argument("--pre-grid", nargs="*", default=[], help="k=v1,v2... para preprocesado")
    ap.add_argument("--diffe", nargs="*", default=[], help="variants=TF,AT alpha=0.1,1.0 ...")
    ap.add_argument("--other", nargs="*", default=[], help="models=eegnet,deepconvnet epochs=50 ...")

    # slurm
    ap.add_argument("--partition", default="standard-gpu")
    ap.add_argument("--gres", default="gpu:a100:1")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem", default="16G")
    ap.add_argument("--time", default="06:00:00")
    ap.add_argument("--account", default=None)
    ap.add_argument("--qos", default=None)

    # control
    ap.add_argument("--submit", action="store_true", help="Enviar los jobs (sbatch)")
    ap.add_argument("--dry-run", action="store_true", help="Solo generar scripts, no enviar")
    ap.add_argument("--name", type=str, default=None, help="Prefijo amigable para el job")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    results_root = ensure_dir(project_root / "results")
    data_proc    = ensure_dir(project_root / "data" / "processed" / "BCI2020")
    all_jsonl    = results_root / "all_results.jsonl"

    # 1) Cargar config (si hay)
    cfg = try_load_config(args.config)

    # 2) Expandir grid de preprocesado
    if cfg and "preprocess_grid" in cfg:
        pre_list = expand_preproc_grid(cfg["preprocess_grid"])
    else:
        # desde flags
        if args.pre_grid:
            kv = parse_kv_list(args.pre_grid)
            # normalizar booleanos y numéricos donde aplica
            def norm_list(k, vals):
                out=[]
                for v in vals:
                    if k in ("h_freq",) and v in ("", "None", "null"): out.append(None)
                    elif k in ("preprocess","enable_bandpass","enable_notch","enable_car","enable_ica","enable_baseline","residual"):
                        out.append(to_bool(v))
                    elif k in ("l_freq", "h_freq"):
                        out.append(None if v in ("", "None","null") else float(v))
                    else:
                        out.append(float(v) if re.match(r"^-?\d+(\.\d+)?$", v) else v)
                return out
            normalized = {k: norm_list(k, vals) for k, vals in kv.items()}
            pre_list = expand_preproc_grid(normalized)
        else:
            pre_list = [PreprocCfg()]  # default

    # 3) Expandir DiffE y Other
    if cfg and "models" in cfg:
        diffe_cfg = cfg["models"].get("diffe", {})
        other_cfg = cfg["models"].get("other", {})
        diffe_grid = DiffEGrid(
            variants=diffe_cfg.get("variants", ["TF"]),
            num_epochs=diffe_cfg.get("num_epochs", [100]),
            batch_train=diffe_cfg.get("batch_train", [250]),
            batch_eval=diffe_cfg.get("batch_eval", [32]),
            seed=diffe_cfg.get("seed", [42]),
            alpha=diffe_cfg.get("alpha", [0.1]),
            num_classes=diffe_cfg.get("num_classes", 5),
            channels=diffe_cfg.get("channels", 64),
            n_T=diffe_cfg.get("n_T", [1000]),
            ddpm_dim=diffe_cfg.get("ddpm_dim", [128]),
            encoder_dim=diffe_cfg.get("encoder_dim", [256]),
            fc_dim=diffe_cfg.get("fc_dim", [512]),
        )
        other_grid = OtherGrid(
            models=other_cfg.get("models", [["eegnet","shallowconvnet","deepconvnet"]]),
            epochs=other_cfg.get("epochs", [150]),
            lr=other_cfg.get("lr", [1e-3]),
            weight_decay=other_cfg.get("weight_decay", [0.0]),
            batch_train=other_cfg.get("batch_train", [64]),
            batch_eval=other_cfg.get("batch_eval", [32]),
            seed=other_cfg.get("seed", [42]),
        )
    else:
        # desde flags
        diffe_kv = parse_kv_list(args.diffe) if args.diffe else {}
        other_kv = parse_kv_list(args.other) if args.other else {}
        diffe_grid = DiffEGrid(
            variants=diffe_kv.get("variants", ["TF"]),
            num_epochs=[int(x) for x in diffe_kv.get("epochs", [100])],
            batch_train=[int(x) for x in diffe_kv.get("batch_train", [250])],
            batch_eval=[int(x) for x in diffe_kv.get("batch_eval", [32])],
            seed=[int(x) for x in diffe_kv.get("seed", [42])],
            alpha=[float(x) for x in diffe_kv.get("alpha", [0.1])],
            n_T=[int(x) for x in diffe_kv.get("n_T", [1000])],
            ddpm_dim=[int(x) for x in diffe_kv.get("ddpm_dim", [128])],
            encoder_dim=[int(x) for x in diffe_kv.get("encoder_dim", [256])],
            fc_dim=[int(x) for x in diffe_kv.get("fc_dim", [512])],
        )
        other_grid = OtherGrid(
            models=[ [m.strip() for m in other_kv.get("models", ["eegnet,shallowconvnet,deepconvnet"])[0].split(",")] ],
            epochs=[int(x) for x in other_kv.get("epochs", [150])],
            lr=[float(x) for x in other_kv.get("lr", [1e-3])],
            weight_decay=[float(x) for x in other_kv.get("weight_decay", [0.0])],
            batch_train=[int(x) for x in other_kv.get("batch_train", [64])],
            batch_eval=[int(x) for x in other_kv.get("batch_eval", [32])],
            seed=[int(x) for x in other_kv.get("seed", [42])],
        )

    # 4) Por cada configuración de preprocesado, generamos dataset ORCH y un sbatch
    created_scripts = []
    for pre_cfg in pre_list:
        tag = pre_cfg.tag()
        dataset_name = f"BCI_{tag}_{short_hash(tag)}.npz"
        out_npz = data_proc / dataset_name  # etiqueta ORCH garantizada en nombre

        # carpeta de resultados para este dataset
        exp_id = f"{args.name or 'ORCH'}_{tag}_{now_id()}"
        exp_dir = ensure_dir(results_root / exp_id)

        # 4.1 línea para preprocesado
        pre_cmd = f"srun python -u {shlex.quote(str(PREPROC_SCRIPT))} {pre_cfg.to_preproc_cli(out_npz)} |& tee -a {shlex.quote(str(exp_dir / 'preprocess.log'))}"

        # 4.2 líneas para DiffE combos
        diffe_lines = []
        for h in expand_diffe_grid(diffe_grid):
            variant = h['variant'].upper()
            script = DIFFE_SCRIPTS.get(variant)
            if not script or not script.exists():
                raise FileNotFoundError(f"No encuentro main_OP.py de DiffE variante '{variant}' en {DIFFE_SCRIPTS.get(variant)}")
            diffe_run_dir = ensure_dir(exp_dir / f"DiffE_{variant}" /
                                       f"ne{h['num_epochs']}_bt{h['batch_train']}_be{h['batch_eval']}_a{h['alpha']}_ed{h['encoder_dim']}_fd{h['fc_dim']}_dd{h['ddpm_dim']}_nT{h['n_T']}")
            log = diffe_run_dir / "train.log"
            cmd = (
                f"srun python -u {shlex.quote(str(script))} "
                f"--exp-dir {shlex.quote(str(diffe_run_dir))} "
                f"--dataset_file {shlex.quote(str(out_npz))} --dataset {args.dataset} "
                f"--device {shlex.quote(args.device)} "
                f"--num_epochs {h['num_epochs']} --batch_train {h['batch_train']} --batch_eval {h['batch_eval']} "
                f"--seed {h['seed']} --alpha {h['alpha']} --num_classes 5 --channels 64 "
                f"--n_T {h['n_T']} --ddpm_dim {h['ddpm_dim']} --encoder_dim {h['encoder_dim']} --fc_dim {h['fc_dim']} "
                f"|& tee -a {shlex.quote(str(log))} ; "
                # extra: volcado de RESULT_JSON a un agregador global
                f"grep -a \"^\\[RESULT_JSON\\]\" {shlex.quote(str(log))} >> {shlex.quote(str(all_jsonl))}"
            )
            diffe_lines.append(cmd)

        # 4.3 líneas para Other Models combos (cada run entrena lista de modelos)
        other_lines = []
        for o in expand_other_grid(other_grid):
            other_run_dir = ensure_dir(exp_dir / "OtherModels" /
                                       f"ne{o['epochs']}_bt{o['batch_train']}_be{o['batch_eval']}_lr{o['lr']}_wd{o['weight_decay']}")
            log = other_run_dir / "benchmark.log"
            models_list = " ".join(o["models"])
            cmd = (
                f"srun python -u {shlex.quote(str(OTHER_SCRIPT))} "
                f"--exp-dir {shlex.quote(str(other_run_dir))} "
                f"--dataset_file {shlex.quote(str(out_npz))} --dataset {args.dataset} "
                f"--device {shlex.quote(args.device)} "
                f"--epochs {o['epochs']} --batch_train {o['batch_train']} --batch_eval {o['batch_eval']} "
                f"--lr {o['lr']} --weight_decay {o['weight_decay']} "
                f"--models {models_list} "
                f"--out_csv {shlex.quote(str(other_run_dir / 'benchmark_results.csv'))} "
                f"|& tee -a {shlex.quote(str(log))} ; "
                f"grep -a \"^\\[RESULT_JSON\\]\" {shlex.quote(str(log))} >> {shlex.quote(str(all_jsonl))}"
            )
            other_lines.append(cmd)

        # 4.4 ensamblar sbatch
        slurm = SlurmSpec(
            job_name=(args.name or "ORCH")[:12] + "_" + short_hash(tag),
            partition=args.partition, gres=args.gres,
            cpus_per_task=args.cpus, mem=args.mem, time=args.time,
            account=args.account, qos=args.qos
        )
        lines = [pre_cmd, ""] + diffe_lines + [""] + other_lines
        sbatch_text = build_sbatch(
            job=slurm,
            venv_activate=args.venv_activate,
            project_root=project_root,
            lines=lines,
            out_dir=exp_dir,
        )

        # 4.5 escribir script y (opcional) enviar
        sbatch_path = exp_dir / "run.sh"
        with open(sbatch_path, "w", encoding="utf-8") as f:
            f.write(sbatch_text)
        os.chmod(sbatch_path, 0o755)
        created_scripts.append(sbatch_path)

        if args.submit and not args.dry_run:
            print(f"[submit] sbatch {sbatch_path}")
            subprocess.run(["sbatch", str(sbatch_path)], check=False)

    # resumen
    print("\n# Orquestador listo.")
    for p in created_scripts:
        print(f"- {p}")

if __name__ == "__main__":
    main()



