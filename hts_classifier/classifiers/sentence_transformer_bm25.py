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
      [--dense]  [--embed-model BAAI/bge-small-en-v1.5] --k 5 10 20 30

BM25 arms are stdlib. --dense adds embedding-based arms (semantic matching;
needs `pip install sentence-transformers`; ~130MB model download on first
run; CPU works, GPU faster). The UNION row merges every enabled arm --
lexical and semantic candidates are complementary, and the table decides
which arm earns its place.

python retrieval_recall.py --train data/hts_train_20k_freeform.jsonl --golden data/hts_golden_v2c.jsonl --catalog data/hts_processed.json --dense --k 5 10 20 30


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


def build_dense(texts, model_name):
    """Encode texts -> unit vectors. Lazy import so BM25-only runs need no torch."""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)
    return model, np.asarray(emb)


def dense_search(model, mat, query_text, topn):
    import numpy as np
    q = model.encode([query_text], normalize_embeddings=True)[0]
    scores = mat @ q
    idx = np.argsort(-scores)[:topn]
    return [(int(i), float(scores[i])) for i in idx]


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
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
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

    d_model = d_train = d_cat = None
    if args.dense:
        print(f"encoding with {args.embed_model} ...")
        d_model, d_train = build_dense(t_texts, args.embed_model)
        if cat_texts:
            _, d_cat = build_dense(cat_texts, args.embed_model)
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

    def arm_lists(rec):
        q_toks = toks(rec.get("description", ""))
        q_text = str(rec.get("description", ""))
        arms = {}
        arms["knn-bm25"] = dedupe_heads(knn.search(q_toks, args.pool), t_heads)
        if cat_bm is not None:
            arms["cat-bm25"] = dedupe_heads(cat_bm.search(q_toks, kmax), cat_heads)
        if d_train is not None:
            arms["knn-dense"] = dedupe_heads(
                dense_search(d_model, d_train, q_text, args.pool), t_heads)
        if d_cat is not None:
            arms["cat-dense"] = dedupe_heads(
                dense_search(d_model, d_cat, q_text, kmax), cat_heads)
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
    for rec in golden:
        gold_h = digits(rec.get("code") or rec.get("hts_code"))[:4]
        if not gold_h:
            continue
        n += 1
        arms = arm_lists(rec)
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
