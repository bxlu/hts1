#!/usr/bin/env python3
"""
make_retrieval_train.py -- build retrieval+selection datasets.

Augments each record's description with a candidate-heading block retrieved
by dense kNN over the training corpus (primary) plus dense catalog search
(tail insurance). The TARGET is untouched -- existing gated rationales are
reused -- so input format is the only variable vs the closed-book model.

Because candidates ride inside the 'description' field, train_hts_classifier,
bakeoff_eval, vote_eval and the audit tooling all work UNCHANGED.

Correctness rules baked in:
  * SELF-EXCLUSION: a training row never retrieves itself from the index.
  * GOLD INJECTION (train/val mode only): if retrieval missed the gold
    heading, it is injected so the target is learnable. NEVER in eval mode --
    eval candidates are pure retrieval, so the ~93% ceiling manifests
    honestly instead of leaking the answer.
  * POSITION SHUFFLE: candidates are shuffled with a per-row deterministic
    seed so gold's position carries no signal.
  * The frozen golden file is never modified; eval mode emits a DERIVED
    input-formatted artifact (same ids, same codes, new filename).

Usage:
  python make_retrieval_train.py --mode train \
      --in data/hts_train_20k_freeform.jsonl \
      --train-index data/hts_train_20k_freeform.jsonl \
      --catalog data/hts_processed.json \
      --out data/hts_train_20k_rk20.jsonl --embed-backend vertex

  (same for --in data/hts_val_20k_freeform.jsonl -> hts_val_20k_rk20.jsonl)

  python make_retrieval_train.py --mode eval \
      --in data/hts_golden_v2c.jsonl \
      --train-index data/hts_train_20k_freeform.jsonl \
      --catalog data/hts_processed.json \
      --out data/hts_golden_v2c_rk20.jsonl --embed-backend vertex
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------

def digits(c):
    return re.sub(r"\D", "", str(c or ""))


def load_jsonl(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        raise SystemExit(f"ERROR: {path} empty")
    return rows


def _sha(texts):
    h = hashlib.sha1()
    for t in texts:
        h.update(str(t).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _char_batches(texts, max_chars=16000, max_n=16, clip=1500):
    batch, budget, out = [], 0, []
    for t in texts:
        t = str(t)[:clip]
        if batch and (budget + len(t) > max_chars or len(batch) >= max_n):
            out.append(batch)
            batch, budget = [], 0
        batch.append(t)
        budget += len(t)
    if batch:
        out.append(batch)
    return out


class VertexEmbedder:
    name = "vertex"

    def __init__(self, model):
        import json as _json
        import vertexai
        from vertexai.language_models import TextEmbeddingModel
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        sa = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        creds = None
        if sa and os.path.exists(sa):
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                sa, scopes=["https://www.googleapis.com/auth/cloud-platform"])
            project = project or _json.load(open(sa)).get("project_id")
        vertexai.init(project=project,
                      location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                      credentials=creds)
        self.m = TextEmbeddingModel.from_pretrained(model)

    def encode(self, texts, task):
        import numpy as np
        from vertexai.language_models import TextEmbeddingInput
        out = []
        batches = _char_batches(texts)
        for bi, batch in enumerate(batches, 1):
            inputs = [TextEmbeddingInput(t, task) for t in batch]
            for e in self.m.get_embeddings(inputs):
                out.append(e.values)
            if bi % 25 == 0:
                print(f"  embedded {bi}/{len(batches)} batches")
        v = np.asarray(out, dtype="float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v


class LocalEmbedder:
    """Swap seam for a self-contained production encoder (sideloaded path)."""
    name = "local"

    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)

    def encode(self, texts, task):
        import numpy as np
        v = self.m.encode(list(texts), batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)
        return np.asarray(v, dtype="float32")


def make_embedder(args):
    model = args.embed_model
    if args.embed_backend == "vertex" and model == "BAAI/bge-small-en-v1.5":
        model = "text-embedding-005"
    return VertexEmbedder(model) if args.embed_backend == "vertex" else LocalEmbedder(model)


def embed_corpus(texts, embedder, cache_dir, tag, task):
    import numpy as np
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}_{embedder.name}_{_sha(texts)}.npy")
    if os.path.exists(path):
        print(f"{tag}: loaded cached embeddings")
        return np.load(path)
    print(f"{tag}: embedding {len(texts)} texts ...")
    v = embedder.encode(texts, task)
    np.save(path, v)
    return v


# ---------------------------------------------------------------------------

def load_heading_names(path):
    """{4-digit heading: its own one-line catalog description} -- the short
    text shown per candidate (path_string docs are too long for 20-candidate
    prompts). Falls back to the first row seen under a heading."""
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw.get("data") if isinstance(raw, dict) else raw
    names = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        code = digits(r.get("hts_code") or r.get("htsno") or r.get("code") or "")
        desc = str(r.get("description") or "").strip()
        if len(code) < 4 or not desc or code[:2] in ("98", "99"):
            continue
        h = code[:4]
        if len(code) == 4:                 # the heading's own row: authoritative
            names[h] = desc
        else:
            names.setdefault(h, desc)      # fallback: first child seen
    return names


def build_candidate_block(cands, names):
    lines = [f"{h}: {names.get(h, '')}".rstrip(": ").strip() for h in cands]
    return ("\n\nCandidate HTS headings:\n" + "\n".join(lines) +
            "\n\nSelect the correct heading from the candidates and provide "
            "the full 10-digit HTS code with a brief justification.")


def topk_rows(mat, qv, exclude_idx, pool):
    import numpy as np
    s = mat @ qv
    if exclude_idx is not None and 0 <= exclude_idx < len(s):
        s[exclude_idx] = -1e9
    return np.argsort(-s)[:pool]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True,
                    help="train: gold injected if retrieval missed it; "
                         "eval: pure retrieval, no injection")
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--train-index", required=True,
                    help="corpus the kNN retrieves from (the training file)")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--cat-k", type=int, default=5,
                    help="catalog candidates merged in (tail insurance)")
    ap.add_argument("--embed-backend", choices=["vertex", "local"], default="vertex")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--embed-cache", default="data/emb_cache")
    args = ap.parse_args()

    records = load_jsonl(args.infile)
    index_rows = load_jsonl(args.train_index)
    names = load_heading_names(args.catalog)
    print(f"catalog: {len(names)} heading names")

    idx_texts = [str(r.get("description", "")) for r in index_rows]
    idx_heads = [digits(r.get("code") or r.get("hts_code"))[:4] for r in index_rows]
    idx_ids = {r.get("id"): i for i, r in enumerate(index_rows)}

    cat_heads = sorted(names)
    cat_texts = [f"{h}: {names[h]}" for h in cat_heads]

    embedder = make_embedder(args)
    E_idx = embed_corpus(idx_texts, embedder, args.embed_cache,
                         "train", "RETRIEVAL_DOCUMENT")
    E_cat = embed_corpus(cat_texts, embedder, args.embed_cache,
                         "catnames", "RETRIEVAL_DOCUMENT")
    q_texts = [str(r.get("description", "")) for r in records]
    q_tag = "trainq" if args.infile == args.train_index else \
            os.path.splitext(os.path.basename(args.infile))[0] + "_q"
    E_q = embed_corpus(q_texts, embedder, args.embed_cache,
                       q_tag, "RETRIEVAL_QUERY")

    n_inject = n_missing = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for qi, rec in enumerate(records):
            gold_h = digits(rec.get("code") or rec.get("hts_code"))[:4]
            self_idx = idx_ids.get(rec.get("id"))
            # kNN candidates (dedupe headings, order = similarity)
            seen, cands = set(), []
            for i in topk_rows(E_idx, E_q[qi], self_idx, pool=300):
                h = idx_heads[i]
                if h and h not in seen:
                    seen.add(h)
                    cands.append(h)
                    if len(cands) >= args.k:
                        break
            # catalog tail insurance
            for i in topk_rows(E_cat, E_q[qi], None, pool=args.cat_k):
                h = cat_heads[i]
                if h not in seen:
                    seen.add(h)
                    cands.append(h)
            cands = cands[:args.k + args.cat_k]

            injected = False
            if gold_h and gold_h not in seen:
                if args.mode == "train":
                    cands[-1] = gold_h
                    injected = True
                    n_inject += 1
                else:
                    n_missing += 1
            # deterministic per-row shuffle kills position signal
            rng = random.Random(int(hashlib.sha1(
                str(rec.get("id")).encode()).hexdigest()[:8], 16))
            rng.shuffle(cands)

            out = dict(rec)
            out["description_raw"] = rec.get("description", "")
            out["description"] = (rec.get("description", "")
                                  + build_candidate_block(cands, names))
            out["candidates"] = cands
            out["gold_in_candidates"] = bool(gold_h and gold_h in cands)
            out["injected"] = injected
            f.write(json.dumps(out) + "\n")

    print(f"\nwrote {len(records)} rows -> {args.out}")
    if args.mode == "train":
        print(f"gold injected (retrieval miss): {n_inject} "
              f"({n_inject/len(records):.1%})")
    else:
        print(f"gold ABSENT from candidates (honest ceiling): {n_missing} "
              f"({n_missing/len(records):.1%})  <- these items cannot be won")


if __name__ == "__main__":
    main()
