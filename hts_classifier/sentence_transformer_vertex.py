#!/usr/bin/env python3
"""
retrieval_recall.py -- measure the CEILING of retrieval+selection before
building any of it. In a selection architecture the model can only pick a
heading that retrieval surfaced, so recall@k on the frozen golden IS the
accuracy ceiling at candidate-list size k. Gate: union recall@20 >= ~90%
-> proceed to the retrain; below -> improve the retriever first.

Two candidate sources, reported separately and as a union:
  TRAIN-KNN : BM25 over the 8.5K labeled training descriptions; retrieved
              rows vote their headings (product vocabulary matches product
              vocabulary -- usually the stronger source).
  CATALOG   : BM25 over official heading text from the USITC HTS JSON
              (covers tail headings the training data has never seen).

Usage:
  python retrieval_recall.py --train data/hts_train_20k_freeform.jsonl \
      --golden data/hts_golden_v2c.jsonl \
      [--catalog data/hts_2026_rev4.json] \
      [--dense --embed-backend vertex]   (or --embed-backend local
       --embed-model <hub-name-or-local-path>)  --k 5 10 20 30

BM25 arms are stdlib. --dense adds embedding-based arms (semantic matching;
needs `pip install sentence-transformers`; ~130MB model download on first
run; CPU works, GPU faster). The UNION row merges every enabled arm --
lexical and semantic candidates are complementary, and the table decides
which arm earns its place.
"""
import argparse, json, math, re, sys
from collections import Counter, defaultdict

STOP = set("""the a an of and or for with from to in on by is are was were be
been this that other others including included except not whether such per
item items product products merchandise style no number made make use used
""".split())


def digits(c):
    return re.sub(r"\D", "", str(c or ""))


def toks(text):
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(t) >= 3 and t not in STOP]


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


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avg = sum(self.lens) / max(len(self.lens), 1)
        self.inv = defaultdict(list)
        for i, c in enumerate(self.docs):
            for t in c:
                self.inv[t].append(i)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(ix) + 0.5) / (len(ix) + 0.5))
                    for t, ix in self.inv.items()}

    def search(self, query_tokens, topn):
        scores = defaultdict(float)
        for t in query_tokens:
            if t not in self.inv:
                continue
            idf = self.idf[t]
            for i in self.inv[t]:
                tf = self.docs[i][t]
                dl = self.lens[i]
                scores[i] += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avg))
        return sorted(scores.items(), key=lambda kv: -kv[1])[:topn]


import hashlib
import os


def _sha(texts):
    h = hashlib.sha1()
    for t in texts:
        h.update(str(t).encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _char_batches(texts, max_chars=60000, max_n=40, clip=8000):
    """Split texts into request batches under Vertex token/count limits."""
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


class LocalEmbedder:
    """sentence-transformers; --embed-model may be a hub name or a LOCAL PATH
    (sideloaded snapshot) for networks where huggingface.co is blocked."""
    name = "local"

    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model)

    def encode(self, texts, task):
        import numpy as np
        v = self.m.encode(list(texts), batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True)
        return np.asarray(v, dtype="float32")


class VertexEmbedder:
    """Google Vertex text embeddings -- same credentials/network path as the
    project's Gemini calls. Default model: text-embedding-005."""
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
            if bi % 20 == 0:
                print(f"  embedded {bi}/{len(batches)} batches")
        v = np.asarray(out, dtype="float32")
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v


def make_embedder(args):
    model = args.embed_model
    if args.embed_backend == "vertex" and model == "BAAI/bge-small-en-v1.5":
        model = "text-embedding-005"
    return VertexEmbedder(model) if args.embed_backend == "vertex" else LocalEmbedder(model)


def embed_corpus(texts, embedder, cache_dir, tag, task):
    """Disk-cached corpus embedding: same texts + backend -> load, not re-embed."""
    import numpy as np
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}_{embedder.name}_{_sha(texts)}.npy")
    if os.path.exists(path):
        print(f"{tag}: loaded cached embeddings ({path})")
        return np.load(path)
    print(f"{tag}: embedding {len(texts)} texts ...")
    v = embedder.encode(texts, task)
    np.save(path, v)
    return v


def load_catalog_headings(path):
    """USITC HTS JSON -> {4-digit heading: concatenated text}.
    Handles both the raw export schema (htsno/description) and the processed
    schema (hts_code/description/path_string). path_string is preferred as
    doc text when present -- it carries full hierarchical context, which the
    raw file's bare fragments ('Males', 'Other') lack; this matters most for
    dense encoders. Ch 98/99 special provisions are excluded (never labels)."""
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw.get("data") if isinstance(raw, dict) else raw
    heads = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        code = digits(r.get("hts_code") or r.get("htsno") or r.get("hts8")
                      or r.get("code") or "")
        text = str(r.get("path_string") or r.get("description")
                   or r.get("desc") or "").strip()
        if len(code) >= 4 and text and code[:2] not in ("98", "99"):
            heads[code[:4]].append(text)
    return {h: " ".join(parts)[:6000] for h, parts in heads.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--catalog", default=None, help="USITC HTS JSON (optional)")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--dense", action="store_true",
                    help="add embedding-based retrieval arms (semantic)")
    ap.add_argument("--embed-backend", choices=["local", "vertex"], default="local",
                    help="local = sentence-transformers (hub name or sideloaded "
                         "path); vertex = Google text-embedding-005 via the "
                         "project's existing GCP credentials (for networks "
                         "where huggingface.co is blocked)")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                    help="local model name/path, or vertex model id")
    ap.add_argument("--embed-cache", default="data/emb_cache",
                    help="embeddings cached here; re-runs load instead of re-embed")
    ap.add_argument("--pool", type=int, default=200,
                    help="raw rows retrieved before heading dedupe (train-kNN)")
    args = ap.parse_args()

    train = load_jsonl(args.train)
    golden = load_jsonl(args.golden)
    kmax = max(args.k)

    t_heads = [digits(r.get("code") or r.get("hts_code"))[:4] for r in train]
    t_texts = [str(r.get("description", "")) for r in train]
    t_docs = [toks(t) for t in t_texts]
    knn = BM25(t_docs)

    cat_bm, cat_heads, cat_texts = None, [], []
    if args.catalog:
        cat = load_catalog_headings(args.catalog)
        cat_heads = list(cat)
        cat_texts = [cat[h] for h in cat_heads]
        cat_bm = BM25([toks(v) for v in cat_texts])
        print(f"catalog: {len(cat_heads)} headings indexed")

    d_train = d_cat = d_gold = None
    if args.dense:
        embedder = make_embedder(args)
        d_train = embed_corpus(t_texts, embedder, args.embed_cache,
                               "train", "RETRIEVAL_DOCUMENT")
        if cat_texts:
            d_cat = embed_corpus(cat_texts, embedder, args.embed_cache,
                                 "catalog", "RETRIEVAL_DOCUMENT")
        g_texts = [str(r.get("description", "")) for r in golden]
        d_gold = embed_corpus(g_texts, embedder, args.embed_cache,
                              "golden", "RETRIEVAL_QUERY")
    print(f"train:   {len(train)} rows, {len(set(t_heads))} distinct headings\n")

    def dedupe_heads(hits, heads):
        seen, out = set(), []
        for i, _ in hits:
            h = heads[i]
            if h and h not in seen:
                seen.add(h)
                out.append(h)
                if len(out) >= kmax:
                    break
        return out

    def dense_hits(mat, qv, topn):
        import numpy as np
        s = mat @ qv
        idx = np.argsort(-s)[:topn]
        return [(int(i), float(s[i])) for i in idx]

    def arm_lists(rec, gi):
        q_toks = toks(rec.get("description", ""))
        arms = {}
        arms["knn-bm25"] = dedupe_heads(knn.search(q_toks, args.pool), t_heads)
        if cat_bm is not None:
            arms["cat-bm25"] = dedupe_heads(cat_bm.search(q_toks, kmax), cat_heads)
        if d_train is not None:
            arms["knn-dense"] = dedupe_heads(
                dense_hits(d_train, d_gold[gi], args.pool), t_heads)
        if d_cat is not None:
            arms["cat-dense"] = dedupe_heads(
                dense_hits(d_cat, d_gold[gi], kmax), cat_heads)
        return arms

    def union(lists):
        seen, out = set(), []
        for row in zip(*[l + [None] * kmax for l in lists]):
            for h in row:
                if h and h not in seen:
                    seen.add(h)
                    out.append(h)
        return out

    arm_names = None
    hits = {}
    misses = Counter()
    n = 0
    for gi, rec in enumerate(golden):
        gold_h = digits(rec.get("code") or rec.get("hts_code"))[:4]
        if not gold_h:
            continue
        n += 1
        arms = arm_lists(rec, gi)
        if arm_names is None:
            arm_names = list(arms) + ["UNION"]
            hits = {(a, k): 0 for a in arm_names for k in args.k}
        u = union(list(arms.values()))
        for k in args.k:
            for a, lst in arms.items():
                hits[(a, k)] += int(gold_h in lst[:k])
            hits[("UNION", k)] += int(gold_h in u[:k])
        if gold_h not in u[:kmax]:
            misses[gold_h[:2]] += 1

    print(f"RETRIEVAL RECALL on {n} golden items  (= selection accuracy ceiling)")
    header = f"{'k':>4}" + "".join(f"{a:>11}" for a in arm_names)
    print(header)
    print("-" * len(header))
    for k in args.k:
        row = f"{k:>4}" + "".join(f"{hits[(a, k)]/n:>11.1%}" for a in arm_names)
        print(row)
    if misses:
        print(f"\ncomplete misses @k={kmax}: {sum(misses.values())} "
              f"({sum(misses.values())/n:.1%}) -- by chapter: "
              f"{dict(misses.most_common(8))}")
    print("\ngate: UNION recall@20 >= ~90% -> build the selection retrain; "
          "\nbelow -> the retriever needs work before any GPU time is spent.")


if __name__ == "__main__":
    main()
