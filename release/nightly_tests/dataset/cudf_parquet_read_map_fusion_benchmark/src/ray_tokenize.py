# SPDX-License-Identifier: Apache-2.0
"""
Ray Data tokenization building blocks (run on GPU workers).

Module-level imports are head-safe (numpy only); cuDF and the cuDF-based
tokenizer are imported lazily inside the actor so this module can be referenced
from the CPU head and shipped to workers via `py_modules`.
"""

from __future__ import annotations

import numpy as np

from . import ray_common as C


class GPUTokenizer:
    """Stateful Ray Data actor: cuDF financial tokenizer over a batch of raw
    transactions. Emits, per transaction, the (data-independent) field token
    IDs plus grouping/order keys and the fraud label.

    Returned arrays are plain numpy so the CPU head can handle them.
    """

    def __init__(
        self,
        merchant_hash_size: int = C.MERCHANT_HASH_SIZE,
        carry_cols=None,
        summary_only: bool = False,
    ):
        import cudf  # lazy: worker-only
        import cupy as cp  # lazy: worker-only
        from src.tokenizer import FinancialTokenizerPipeline

        self._cudf = cudf
        self._cp = cp
        self.pipeline = FinancialTokenizerPipeline(
            merchant_hash_size=merchant_hash_size,
            use_streams=False,
        )
        self._fitted = False
        # Profiling aid: when True, do all GPU work but return only a tiny scalar
        # summary instead of the full token matrix (isolates compute from the
        # host-side output-block serialization cost). Not used in production.
        self.summary_only = summary_only
        # Optional raw columns to pass through, aligned with each tokenized row
        # (used by NB04 so the embeddings table also holds raw features for the
        # NB05 "combined" model). Names are the original TabFormer column names.
        self.carry_cols = list(carry_cols) if carry_cols else []

    @staticmethod
    def _fraud_label(proc, n, cp):
        # Device-side: compare on the GPU, single D2H of an int64 column.
        for col in ("is_fraud?", "is_fraud", "fraud"):
            if col in proc.columns:
                s = proc[col].astype("str")
                yes = ((s == "Yes") | (s == "1")).values.astype("int64")
                return cp.asnumpy(yes)
        return np.zeros(n, dtype="int64")

    def __call__(self, batch):
        cudf = self._cudf
        cp = self._cp
        # `batch` is a cudf.DataFrame already resident in VRAM: either decoded
        # straight from Parquet by the fused GPU reader (disk -> VRAM, no host
        # block), or handed to us by Ray via `batch_format="cudf"`. Everything
        # below runs on the GPU; only compact token-id arrays cross back to host.
        gdf = batch if isinstance(batch, cudf.DataFrame) else cudf.DataFrame(batch)
        # Fast path: integer-only date math, no sort, no time-delta groupby (all
        # unused by the 12-field vocab / read+tokenize stage).
        proc = self.pipeline.preprocess(
            gdf, sort=False, compute_time_delta=False, fast_dates=True
        )
        if not self._fitted:
            # vocab is data-independent (fixed bins/hash/ranges) -> consistent IDs.
            self.pipeline.fit(proc)
            self._fitted = True
        if self.summary_only:
            ids_dev = self.pipeline.transform_ids(proc)  # stays on device
            return {"chk": np.array([int(ids_dev.sum())], dtype="int64")}
        # On-device int32 (n, n_fields) token-ID matrix, then ONE D2H copy.
        token_ids = cp.asnumpy(self.pipeline.transform_ids(proc))
        user = proc["user"].astype("int64").values
        card = proc["card"].astype("int64").values
        uc_key = cp.asnumpy(user * 100 + card)  # unique (user, card) key
        ts = cp.asnumpy(proc["ts_epoch"].values)  # epoch s (integer date math)
        out = {
            "uc_key": uc_key,
            "ts": ts,
            "token_ids": token_ids,
            "label": self._fraud_label(proc, len(token_ids), cp),
        }
        for col in self.carry_cols:  # raw features, row-aligned
            lc = col.strip().replace(" ", "_").lower()
            out[col] = proc[lc].to_pandas().to_numpy()
        return out


def build_sequences(
    group, seq_length: int = C.SEQ_LENGTH, chunk_size: int = C.SEQ_CHUNK_SIZE
):
    """Ray Data `map_groups` fn: turn one (user, card)'s transactions into
    fixed-length causal-LM sequences:  <bos> txn1 <sep> txn2 ... <eos> <pad>...
    """
    ts, tok = group["ts"], group["token_ids"]
    order = np.argsort(ts, kind="stable")
    tok = tok[order]
    seqs = []
    for start in range(0, len(tok), chunk_size):
        chunk = tok[start : start + chunk_size]  # (m, n_fields)
        seq = [C.BOS_TOKEN_ID]
        for i, row in enumerate(chunk):
            seq.extend(int(x) for x in row)
            if i < len(chunk) - 1:
                seq.append(C.SEP_TOKEN_ID)
        seq.append(C.EOS_TOKEN_ID)
        seq = seq[:seq_length]
        arr = np.full(seq_length, C.PAD_TOKEN_ID, dtype="int64")
        arr[: len(seq)] = seq
        seqs.append(arr)
    if not seqs:
        return {"input_ids": np.zeros((0, seq_length), dtype="int64")}
    return {"input_ids": np.stack(seqs, axis=0)}
