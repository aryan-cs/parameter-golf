from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
TRAINER_SRC = ROOT / "records/track_non_record_16mb/2026-03-24_H200_LeakyReLU_LegalTTT_FlashFallback/train_gpt.py"

APP_NAME = "parameter-golf-h100"
DATA_VOLUME_NAME = "parameter-golf-data"
RUNS_VOLUME_NAME = "parameter-golf-runs"

REMOTE_TRAINER_DIR = Path("/root/parameter-golf")
REMOTE_TRAINER_PATH = REMOTE_TRAINER_DIR / "train_gpt.py"
REMOTE_DATA_ROOT = Path("/data")
REMOTE_DATASET_DIR = REMOTE_DATA_ROOT / "datasets" / "fineweb10B_sp1024"
REMOTE_TOKENIZER_PATH = REMOTE_DATA_ROOT / "tokenizers" / "fineweb_1024_bpe.model"
REMOTE_RUNS_ROOT = Path("/runs")

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install("zstandard")
    .add_local_file(str(TRAINER_SRC), str(REMOTE_TRAINER_PATH))
)


BASE_ENV = {
    "VOCAB_SIZE": "1024",
    "NUM_LAYERS": "11",
    "BIGRAM_VOCAB_SIZE": "1536",
    "XSA_LAST_N": "4",
    "ROPE_DIMS": "16",
    "LN_SCALE": "1",
    "VE_ENABLED": "1",
    "VE_DIM": "128",
    "VE_LAYERS": "9,10",
    "TRAIN_SEQ_LEN": "2048",
    "EVAL_SEQ_LEN": "2048",
    "TRAIN_BATCH_TOKENS": "786432",
    "VAL_BATCH_SIZE": "786432",
    "ITERATIONS": "9000",
    "MAX_WALLCLOCK_SECONDS": "600",
    "EVAL_STRIDE": "64",
    "EXTRA_STRIDE64_FINAL_EVAL": "0",
    "WARMDOWN_ITERS": "3500",
    "MUON_WD": "0.04",
    "ADAM_WD": "0.04",
    "MATRIX_LR": "0.025",
    "SCALAR_LR": "0.025",
    "TIED_EMBED_LR": "0.035",
    "MUON_MOMENTUM": "0.99",
    "MUON_MOMENTUM_WARMUP_START": "0.92",
    "MUON_MOMENTUM_WARMUP_STEPS": "1500",
    "TTT_ENABLED": "1",
    "TTT_LR": "0.002",
    "TTT_EPOCHS": "3",
    "TTT_CHUNK_TOKENS": "32768",
    "TTT_FREEZE_BLOCKS": "0",
    "TTT_MOMENTUM": "0.9",
    "TTT_BATCH_SEQS": "32",
    "TTT_GRAD_CLIP": "1.0",
}


CANDIDATE_ENVS = {
    "baseline": {},
    "vr1": {"VALUE_RESIDUAL": "1"},
    "bg3072": {"BIGRAM_VOCAB_SIZE": "3072"},
    "vr1_bg3072": {"VALUE_RESIDUAL": "1", "BIGRAM_VOCAB_SIZE": "3072"},
    "tttlr25": {"TTT_LR": "0.0025"},
    "vr1_bg3072_tttlr25": {
        "VALUE_RESIDUAL": "1",
        "BIGRAM_VOCAB_SIZE": "3072",
        "TTT_LR": "0.0025",
    },
}


def _build_env(candidate: str, seed: int, run_id: str) -> dict[str, str]:
    if candidate not in CANDIDATE_ENVS:
        raise ValueError(f"unknown candidate '{candidate}', expected one of {sorted(CANDIDATE_ENVS)}")
    env = dict(os.environ)
    env.update(BASE_ENV)
    env.update(CANDIDATE_ENVS[candidate])
    env["SEED"] = str(seed)
    env["RUN_ID"] = run_id
    env["DATA_PATH"] = str(REMOTE_DATASET_DIR)
    env["TOKENIZER_PATH"] = str(REMOTE_TOKENIZER_PATH)
    return env


@app.function(
    image=image,
    gpu="H100!:8",
    cpu=16,
    memory=65536,
    timeout=4 * 60 * 60,
    startup_timeout=30 * 60,
    volumes={
        str(REMOTE_DATA_ROOT): data_volume,
        str(REMOTE_RUNS_ROOT): runs_volume,
    },
)
def train(candidate: str = "baseline", seed: int = 1337) -> dict[str, str | float]:
    run_id = f"modal_{candidate}_seed{seed}_{int(time.time())}"
    workdir = REMOTE_RUNS_ROOT / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REMOTE_TRAINER_PATH, workdir / "train_gpt.py")

    env = _build_env(candidate, seed, run_id)
    launch_metadata = {
        "candidate": candidate,
        "seed": seed,
        "run_id": run_id,
        "trainer_path": str(workdir / "train_gpt.py"),
        "data_path": env["DATA_PATH"],
        "tokenizer_path": env["TOKENIZER_PATH"],
        "gpu": "H100!:8",
    }
    (workdir / "launch_metadata.json").write_text(json.dumps(launch_metadata, indent=2) + "\n", encoding="utf-8")
    (workdir / "launch_env.json").write_text(
        json.dumps({k: env[k] for k in sorted(BASE_ENV | CANDIDATE_ENVS[candidate] | {"SEED": "", "RUN_ID": "", "DATA_PATH": "", "TOKENIZER_PATH": ""})}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    runs_volume.commit()

    t0 = time.perf_counter()
    subprocess.run(
        [
            "bash",
            "-lc",
            "set -o pipefail; torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train.log",
        ],
        cwd=workdir,
        env=env,
        check=True,
    )
    elapsed_s = time.perf_counter() - t0

    result = {
        "candidate": candidate,
        "seed": str(seed),
        "run_id": run_id,
        "workdir": str(workdir),
        "elapsed_s": round(elapsed_s, 2),
        "log_path": str(workdir / "train.log"),
    }
    print(json.dumps(result, indent=2))
    runs_volume.commit()
    return result


@app.local_entrypoint()
def upload_data(
    local_dataset_dir: str = "data/datasets/fineweb10B_sp1024",
    local_tokenizer_path: str = "data/tokenizers/fineweb_1024_bpe.model",
) -> None:
    dataset_dir = Path(local_dataset_dir).resolve()
    tokenizer_path = Path(local_tokenizer_path).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    with data_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(dataset_dir), str(REMOTE_DATASET_DIR))
        batch.put_file(str(tokenizer_path), str(REMOTE_TOKENIZER_PATH))

    print(f"Uploaded dataset to volume '{DATA_VOLUME_NAME}' at {REMOTE_DATASET_DIR}")
    print(f"Uploaded tokenizer to volume '{DATA_VOLUME_NAME}' at {REMOTE_TOKENIZER_PATH}")


@app.local_entrypoint()
def launch(candidate: str = "baseline", seed: int = 1337) -> None:
    result = train.remote(candidate=candidate, seed=seed)
    print(json.dumps(result, indent=2))
