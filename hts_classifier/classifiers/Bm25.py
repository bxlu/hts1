

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


def load_catalog_headings(path):
    """USITC HTS JSON -> {4-digit heading: concatenated description text}.
    Tolerates the common schemas: a list of rows with htsno/description, or
    {'data': [...]}. Rows without an htsno (chapter notes) are skipped."""
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw.get("data") if isinstance(raw, dict) else raw
    heads = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict):
            continue
        code = digits(r.get("htsno") or r.get("hts8") or r.get("code") or "")
        desc = str(r.get("description") or r.get("desc") or "").strip()
        if len(code) >= 4 and desc:
            heads[code[:4]].append(desc)
    return {h: " ".join(parts)[:4000] for h, parts in heads.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--catalog", default=None, help="USITC HTS JSON (optional)")
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20, 30])
    ap.add_argument("--pool", type=int, default=200,
                    help="raw rows retrieved before heading dedupe (train-kNN)")
    args = ap.parse_args()

    train = load_jsonl(args.train)
    golden = load_jsonl(args.golden)
    kmax = max(args.k)

    t_heads = [digits(r.get("code") or r.get("hts_code"))[:4] for r in train]
    t_docs = [toks(r.get("description", "")) for r in train]
    knn = BM25(t_docs)

    cat_bm, cat_heads = None, []
    if args.catalog:
        cat = load_catalog_headings(args.catalog)
        cat_heads = list(cat)
        cat_bm = BM25([toks(v) for v in cat.values()])
        print(f"catalog: {len(cat_heads)} headings indexed")
    print(f"train:   {len(train)} rows, {len(set(t_heads))} distinct headings\n")

    def knn_candidates(q):
        seen, out = set(), []
        for i, _ in knn.search(q, args.pool):
            h = t_heads[i]
            if h and h not in seen:
                seen.add(h)
                out.append(h)
                if len(out) >= kmax:
                    break
        return out

    def cat_candidates(q):
        if cat_bm is None:
            return []
        return [cat_heads[i] for i, _ in cat_bm.search(q, kmax)]

    def union(a, b):
        seen, out = set(), []
        for pair in zip(a + [None] * kmax, b + [None] * kmax):
            for h in pair:
                if h and h not in seen:
                    seen.add(h)
                    out.append(h)
        return out

    hits = {("knn", k): 0 for k in args.k}
    hits.update({("cat", k): 0 for k in args.k})
    hits.update({("union", k): 0 for k in args.k})
    misses = Counter()
    n = 0
    for rec in golden:
        gold_h = digits(rec.get("code") or rec.get("hts_code"))[:4]
        if not gold_h:
            continue
        n += 1
        q = toks(rec.get("description", ""))
        a, b = knn_candidates(q), cat_candidates(q)
        u = union(a, b)
        for k in args.k:
            hits[("knn", k)] += int(gold_h in a[:k])
            hits[("cat", k)] += int(gold_h in b[:k])
            hits[("union", k)] += int(gold_h in u[:k])
        if gold_h not in u[:kmax]:
            misses[gold_h[:2]] += 1

    print(f"RETRIEVAL RECALL on {n} golden items  (= selection accuracy ceiling)")
    print(f"{'k':>4}{'train-kNN':>12}{'catalog':>10}{'UNION':>9}")
    print("-" * 36)
    for k in args.k:
        c = f"{hits[('cat', k)]/n:.1%}" if cat_bm else "   --"
        print(f"{k:>4}{hits[('knn', k)]/n:>12.1%}{c:>10}{hits[('union', k)]/n:>9.1%}")
    if misses:
        print(f"\ncomplete misses @k={kmax}: {sum(misses.values())} "
              f"({sum(misses.values())/n:.1%}) -- by chapter: "
              f"{dict(misses.most_common(8))}")
    print("\ngate: UNION recall@20 >= ~90% -> build the selection retrain; "
          "\nbelow -> the retriever needs work before any GPU time is spent.")


if __name__ == "__main__":
    main()
