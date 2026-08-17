from pathlib import Path

# --------------------------------------------------------------------------- #
# Shared, cross-node storage (NFS on Anyscale). Falls back to a local dir when
# running off-Anyscale so the code is still importable anywhere.
# --------------------------------------------------------------------------- #
_CLUSTER_STORAGE = Path("/mnt/cluster_storage")
DATA_ROOT = (_CLUSTER_STORAGE if _CLUSTER_STORAGE.is_dir() else Path.home()) / "tfm_ray"

RAW_DIR = DATA_ROOT / "raw"  # synthetic / real raw parquet
SPLIT_DIR = DATA_ROOT / "temporal_split"  # train/val/test parquet
TOKENIZED_DIR = DATA_ROOT / "tokenized"  # NB02 output (parquet, replaces corpus.txt)
MODEL_DIR = DATA_ROOT / "models"  # NB03 checkpoints (HF format)
EMBED_DIR = DATA_ROOT / "embeddings"  # NB04 output (parquet)
OUTPUT_DIR = DATA_ROOT / "outputs"  # NB05 plots / metrics

for _d in (RAW_DIR, SPLIT_DIR, TOKENIZED_DIR, MODEL_DIR, EMBED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Single shared GPU runtime_env. Installed once per worker node, reused by all
# notebooks. Pinned to the exact versions validated on this cluster.
# --------------------------------------------------------------------------- #
GPU_PIP = [
    "torch==2.12.0",
    "transformers==5.9.0",
    "cudf-cu12==25.10.*",
    "kvikio-cu12==25.10.*",
    # Back-half GPU groupby/map_groups via ShuffleStrategy.GPU_SHUFFLE (rapidsmpf+UCXX).
    "rapidsmpf-cu12==25.10.*",
    "ucxx-cu12==0.46.*",
    "cupy-cuda12x",
    "xgboost==3.0.*",
    "scikit-learn",
    "pandas",
    "pyarrow>=17,<20",
]

# Absolute path to the `src` package dir.
_SRC_RAY_DIR = str(Path(__file__).resolve().parent)

# JOB-level env (pass to `ray.init`): ships the src package to every node so
# `import src.tokenizer` works on workers. Code only — the CPU head installs
# nothing here. `py_modules` (local-dir upload) is ONLY valid at the job level.
JOB_RUNTIME_ENV = {
    "py_modules": [_SRC_RAY_DIR],
    "env_vars": {"PIP_EXTRA_INDEX_URL": "https://pypi.nvidia.com"},
}

# PER-OP env (pass to map_batches): installs the GPU wheels on the worker node.
# Merged with the job env, so workers get both the pip deps and src code.
GPU_RUNTIME_ENV = {
    "pip": GPU_PIP,
    "env_vars": {"PIP_EXTRA_INDEX_URL": "https://pypi.nvidia.com"},
}

# Ray Train workers inherit the JOB runtime_env, so for training notebooks we
# pass this combined env to `ray.init` (code + pip). The CPU head driver does not
# import torch, so its (cached) install is harmless.
TRAIN_JOB_ENV = {
    "py_modules": [_SRC_RAY_DIR],
    "pip": GPU_PIP,
    "env_vars": {"PIP_EXTRA_INDEX_URL": "https://pypi.nvidia.com"},
}

# Model / tokenizer constants shared across notebooks (match the original blueprint).
MERCHANT_HASH_SIZE = 2000
# Match the original blueprint exactly: 315 transactions packed into a
# 4096-token causal-LM sequence (~13 tok/txn incl. separators). Same ~64K train
# sequences and same model context window as the original NB02/03.
SEQ_LENGTH = 4096  # tokens per training sequence (NB02 -> NB03)
SEQ_CHUNK_SIZE = 315  # transactions per sequence (~13 tok/txn fills 4096)
EMB_MAX_LENGTH = 128  # per-transaction encode length for embeddings (NB04)
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
SEP_TOKEN_ID = 3
UNK_TOKEN_ID = 4

# Llama decoder config (the ~29M-param foundation model from the original NB03).
MODEL_CONFIG = dict(
    vocab_size=6251,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    intermediate_size=1408,
    max_position_embeddings=8192,
    rope_theta=500000.0,
    hidden_act="silu",
    rms_norm_eps=1.0e-5,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    bos_token_id=BOS_TOKEN_ID,
    eos_token_id=EOS_TOKEN_ID,
    pad_token_id=PAD_TOKEN_ID,
)
