"""
gemini_baseline.py -- score Gemini Flash on the frozen golden set, emitting
the project-standard preds format ({id, gold, pred}) so collision_audit.py,
anyof_score.py, and the coverage tooling all work unchanged.

Purpose: the '85%' Gemini figure predates every data fix in this project and
was never measured on a leak-free benchmark. This defines the REAL target
for the 80-82% initiative.

Uses the same Vertex plumbing as add_rationales_v2.py (generate_text forces
JSON mime type; output is unwrapped the same way). Resumable by id.

Usage:
  python scripts/gemini_baseline.py --golden data/hts_golden_v2c.jsonl \
      --out data/preds_gemini_flash.jsonl --concurrency 8
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
try:
    from hts_classifier.core.simple_direct_classifier import (
        generate_text, load_credentials_from_file, get_project_id_from_file, SimpleSettings)
except ImportError:
    from simple_direct_classifier import (
        generate_text, load_credentials_from_file, get_project_id_from_file, SimpleSettings)

import vertexai
from dotenv import load_dotenv
from tqdm import tqdm

settings = SimpleSettings(
    google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id"),
    google_cloud_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

PROMPT = """You are an expert U.S. customs classifier.

Product description:
\"\"\"{description}\"\"\"

Determine the correct 10-digit HTS (Harmonized Tariff Schedule of the United
States) classification. Reason briefly, then give your final answer on its
own last line in exactly this format:

HTS: NNNN.NN.NN.NN"""

_HTS_ANCHOR = re.compile(r"HTS\s*:?\s*(\d{4}(?:[.\s]?\d{2}){1,3})", re.IGNORECASE)
_CODE_TOKEN = re.compile(r"\b\d{4}(?:\.\d{2}){1,3}\b")


def digits(c):
    return re.sub(r"\D", "", str(c or ""))


def extract_pred(text):
    """Last HTS-anchored code wins (the answer line); fallback last code token."""
    t = str(text or "")
    anchored = _HTS_ANCHOR.findall(t)
    if anchored:
        return digits(anchored[-1])
    toks = _CODE_TOKEN.findall(t)
    return digits(toks[-1]) if toks else ""


def unwrap(t):
    s = (t or "").strip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("text", "answer", "response", "rationale"):
                    if isinstance(obj.get(k), str):
                        s = obj[k]
                        break
        except (json.JSONDecodeError, ValueError):
            pass
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            d = json.loads(s)
            if isinstance(d, str):
                s = d
        except (json.JSONDecodeError, ValueError):
            s = s[1:-1]
    return s.replace('\\"', '"').replace("\\n", "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    project_root = Path(__file__).parent.parent
    if (project_root / ".env").exists():
        load_dotenv(project_root / ".env")
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        if sa_path and os.path.exists(sa_path):
            creds = load_credentials_from_file(sa_path)
            vertexai.init(project=get_project_id_from_file(sa_path),
                          location=settings.google_cloud_location, credentials=creds)
        else:
            vertexai.init(project=settings.google_cloud_project,
                          location=settings.google_cloud_location)
    except Exception as e:
        print(f"Vertex init failed: {e}")
        return

    golden = [json.loads(l) for l in open(args.golden, encoding="utf-8") if l.strip()]
    if args.limit:
        golden = golden[:args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(json.loads(line).get("id"))
            except Exception:
                pass
        if done:
            print(f"resuming: {len(done)} already scored")

    pending = [r for r in golden if r.get("id") not in done and r.get("description")]
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def score(rec):
        async with sem:
            try:
                result = await generate_text(
                    PROMPT.format(description=rec["description"][:2000]),
                    thinking_level="low")
                pred = extract_pred(unwrap(result.text))
                return rec, pred, None
            except Exception as e:
                return rec, "", e

    n = ch = hd = errs = 0
    with open(args.out, "a", encoding="utf-8") as f:
        tasks = [asyncio.ensure_future(score(r)) for r in pending]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            rec, pred, err = await fut
            if err is not None:
                errs += 1
                print(f"\nerror on {rec.get('id')}: {err}")
                continue
            gold = digits(rec.get("code") or rec.get("hts_code"))
            n += 1
            ch += int(bool(pred) and pred[:2] == gold[:2])
            hd += int(bool(pred) and pred[:4] == gold[:4])
            f.write(json.dumps({"id": rec.get("id"), "gold": gold, "pred": pred,
                                "raw": None,
                                "correct_4": int(bool(pred) and pred[:4] == gold[:4])
                                }) + "\n")
            f.flush()

    if n:
        print(f"\nGemini Flash on {args.golden} ({n} scored, {errs} errors):")
        print(f"  chapter (2-digit): {ch/n:.1%}")
        print(f"  heading (4-digit): {hd/n:.1%}")
        print(f"\nnext: python collision_audit.py --train data/hts_train_20k_freeform.jsonl "
              f"--golden {args.golden} --preds {args.out}")
        print("(the CLEAN-strict line from that audit is the apples-to-apples "
              "number against the 3B's 64.7%)")


if __name__ == "__main__":
    asyncio.run(main())
