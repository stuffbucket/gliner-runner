"""Accuracy evaluation of the GLiNER Runner ONNX models on labelled public datasets.

Three steps, each reading what the previous one wrote:

    sample  pick a fixed, seeded sample of every dataset (ids only) -> <out>/samples.json
    run     send each sampled example to a running ``gliner-runner serve`` over HTTP,
            one request at a time, and keep every answer   -> <work>/predictions/<run>.jsonl
    score   compute accuracy, F1, calibration, latency and baselines
            -> <out>/results.json and <out>/predictions.jsonl.gz (answers only, no text)

The datasets are the normalised copies written by ``fetch.py``. Requests have the shape the
site uses (``classify()`` in site/scripts/runner.mjs): one classification task per request,
named after the dataset's own field, with ``min_labels: 0, max_labels: null`` for several-label
tasks. A 429 (queue full) is retried with backoff. Standard library only; run with ``python -I``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SEED = 20261021

# Every task: where its rows are, the field holding the gold label, the request's task name,
# whether it is several-label, and how it is sampled. ``subsample`` is a smaller stratified draw
# from the sample (size, strata) for runs that would take too long on the full sample.
TASKS: dict[str, dict[str, Any]] = {
    "banking77/intent": {"field": "intent", "size": 770, "strata": ("intent",),
                         "subsample": (200, ("intent",))},
    "news_category/topic": {"field": "topic", "size": 1000, "strata": ("topic",)},
    "amazon_polarity/sentiment": {"field": "sentiment", "size": 1000, "strata": ("sentiment",)},
    "civil_comments/toxicity": {"field": "toxicity", "size": 1000, "strata": ("toxicity",)},
    "go_emotions/emotions": {"field": "emotions", "size": 1000, "tune": 500, "several": True},
    "massive/scenario": {"field": "scenario", "size": 100, "parallel": True},
    "massive/language": {"field": "language", "size": 100, "parallel": True},
    "support_tickets/queue": {"field": "queue", "size": 1000, "strata": ("group", "queue")},
    "support_tickets/priority": {"field": "priority", "size": 1000, "strata": ("group", "queue")},
    "fast_decisions": {"all": True, "subsample": (200, ("group",))},
}


# ---------------------------------------------------------------- data


def read_rows(data: Path, dataset: str) -> list[dict[str, Any]]:
    with (data / dataset / "rows.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def stratum(row: dict[str, Any], keys: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(row[k] if k in row else row["task"][k]) for k in keys)


def stratified(rows: list[dict[str, Any]], keys: tuple[str, ...], size: int,
               rng: random.Random) -> list[str]:
    """Proportional allocation (largest remainder), then a seeded draw within each stratum."""
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in rows:
        groups[stratum(row, keys)].append(row["id"])
    total = len(rows)
    quotas = {key: size * len(ids) / total for key, ids in groups.items()}
    counts = {key: math.floor(q) for key, q in quotas.items()}
    leftover = size - sum(counts.values())
    for key in sorted(quotas, key=lambda k: (counts[k] - quotas[k], k))[:leftover]:
        counts[key] += 1
    chosen: list[str] = []
    for key in sorted(groups):
        chosen.extend(rng.sample(sorted(groups[key]), counts[key]))
    return sorted(chosen)


def make_samples(data: Path, seed: int) -> dict[str, Any]:
    samples: dict[str, Any] = {"seed": seed, "tasks": {}}
    for name, spec in TASKS.items():
        dataset = name.split("/")[0]
        rows = read_rows(data, dataset)
        rng = random.Random(f"{seed}/{dataset}")
        entry: dict[str, Any]
        if spec.get("all"):
            entry = {"method": "every row", "eval": [r["id"] for r in rows]}
        elif spec.get("parallel"):
            # The same utterances in every locale: stratify on the English rows, keep the ids.
            english = [r for r in rows if r["group"] == "en-US"]
            picked = stratified(english, ("scenario",), spec["size"], rng)
            suffixes = [i.removeprefix("en-US-") for i in picked]
            locales = sorted({r["group"] for r in rows})
            entry = {"method": f"{spec['size']} utterances stratified by scenario on en-US, "
                               f"the same utterances in each of {len(locales)} locales",
                     "eval": sorted(f"{loc}-{s}" for loc in locales for s in suffixes)}
        elif spec.get("several"):
            test = [r["id"] for r in rows if r["split"] == "test"]
            tune = [r["id"] for r in rows if r["split"] == "validation"]
            entry = {"method": "simple random sample of test; tuning sample from validation",
                     "eval": sorted(rng.sample(sorted(test), spec["size"])),
                     "tune": sorted(rng.sample(sorted(tune), spec["tune"])),
                     "tune_population": len(tune)}
            rows = [r for r in rows if r["split"] == "test"]
        else:
            entry = {"method": "stratified by " + ", ".join(spec["strata"]) +
                               ", proportional allocation",
                     "eval": stratified(rows, spec["strata"], spec["size"], rng)}
        entry["population"] = len(rows)
        if "subsample" in spec:
            # A smaller stratified draw from the sample, for runs too slow to do in full.
            size, keys = spec["subsample"]
            in_sample = set(entry["eval"])
            sub_rng = random.Random(f"{seed}/{dataset}/sub{size}")
            entry["subsamples"] = {f"sub{size}": stratified(
                [r for r in rows if r["id"] in in_sample], keys, size, sub_rng)}
        samples["tasks"][name] = entry
    # Tasks on the same dataset with the same sampling share one sample.
    t = samples["tasks"]
    t["massive/language"]["eval"] = t["massive/scenario"]["eval"]
    t["support_tickets/priority"]["eval"] = t["support_tickets/queue"]["eval"]
    return samples


# ---------------------------------------------------------------- requests


def post(base: str, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
    data = json.dumps(body).encode("utf-8")
    delay = 0.5
    for _attempt in range(12):
        request = urllib.request.Request(f"{base}/v1/infer", data=data,
                                         headers={"content-type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=600) as reply:
                payload = json.load(reply)
            return payload, (time.perf_counter() - started) * 1000
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise RuntimeError(f"HTTP {error.code}: {error.read()[:500]!r}") from error
            retry = error.headers.get("retry-after")
            time.sleep(float(retry) if retry else delay)
            delay = min(delay * 2, 30)
    raise RuntimeError("queue stayed full (429) after 12 attempts")


def request_body(model: str, text: str, task_name: str, labels: list[str], several: bool,
                 descriptions: dict[str, str] | None) -> dict[str, Any]:
    task: dict[str, Any] = {"labels": labels}
    if descriptions:
        task["label_descriptions"] = {label: descriptions[label] for label in labels}
    if several:
        task.update(min_labels=0, max_labels=None)
    return {"model": model, "backend": "onnx", "precision": "fp32", "operation": "classify",
            "text": text, "schema": {"kind": "classification", "tasks": {task_name: task}}}


def jobs_for(name: str, rows: dict[str, dict[str, Any]], ids: list[str],
             descriptions: dict[str, Any] | None) -> Iterable[dict[str, Any]]:
    """One job per request: id, text, task name, labels, several, descriptions, gold."""
    if name == "fast_decisions":
        for row_id in ids:
            row = rows[row_id]
            for head, spec in row["heads"].items():
                key = f"fast_decisions/{row['group']}/{head}"
                yield {"id": f"{row_id}/{head}", "key": key, "group": row["group"],
                       "text": row["text"],
                       "task": head, "labels": spec["labels"], "several": spec["several"],
                       "descriptions": descriptions[key]["labels"] if descriptions else None,
                       "gold": spec["gold"]}
        return
    spec = TASKS[name]
    field = spec["field"]
    labels = sorted({
        value
        for row in rows.values()
        for value in (row["task"][field] if isinstance(row["task"][field], list)
                      else [row["task"][field]])
    })
    for row_id in ids:
        gold = rows[row_id]["task"][field]
        yield {"id": row_id, "key": name, "group": rows[row_id].get("group"),
               "text": rows[row_id]["text"], "task": field,
               "labels": labels, "several": bool(spec.get("several")),
               "descriptions": descriptions[name]["labels"] if descriptions else None,
               "gold": gold if isinstance(gold, list) else [gold]}


def run(args: argparse.Namespace) -> None:
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    all_descriptions = json.loads((HERE / "descriptions.json").read_text(encoding="utf-8"))
    for name in args.tasks:
        dataset = name.split("/")[0]
        # A non-commercial set's descriptions are kept with its data, not here (see fetch.py).
        local = args.data / dataset / "descriptions.json"
        if local.exists():
            all_descriptions.update(json.loads(local.read_text(encoding="utf-8")))
        rows = {r["id"]: r for r in read_rows(args.data, dataset)}
        sample = samples["tasks"][name]
        for split in ["eval", "tune"] if "tune" in sample else ["eval"]:
            descriptions = all_descriptions if args.descriptions else None
            if args.descriptions and name != "fast_decisions" and name not in all_descriptions:
                raise SystemExit(f"{name} has no descriptions")
            mode = "desc" if args.descriptions else "plain"
            run_id = f"{name.replace('/', '-')}--{args.model}--{mode}"
            ids = sample[split]
            if args.subsample:
                if split == "tune":
                    raise SystemExit("subsamples are for single-split tasks")
                ids = sample["subsamples"][args.subsample]
                run_id += f"--{args.subsample}"
            if split == "tune":
                run_id += "--tune"
            target = args.work / "predictions" / f"{run_id}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            done = set()
            if target.exists():
                with target.open(encoding="utf-8") as handle:
                    done = {json.loads(line)["id"] for line in handle}
            jobs = [j for j in jobs_for(name, rows, ids, descriptions)
                    if j["id"] not in done]
            print(f"{run_id}: {len(done)} done, {len(jobs)} to go", file=sys.stderr, flush=True)
            started = time.perf_counter()
            with target.open("a", encoding="utf-8") as out:
                for index, job in enumerate(jobs):
                    body = request_body(args.model, job["text"], job["task"], job["labels"],
                                        job["several"], job["descriptions"])
                    reply, client_ms = post(args.base, body)
                    answer = reply["output"][job["task"]]
                    value = answer["value"]
                    record = {
                        "id": job["id"], "key": job["key"], "group": job["group"],
                        "gold": job["gold"],
                        "several": job["several"],
                        "pred": value if isinstance(value, list) else
                        ([] if value is None else [value]),
                        "confidence": answer.get("confidence"),
                        "probabilities": answer.get("probabilities"),
                        "inference_ms": reply["timing"]["inference_ms"],
                        "queue_ms": reply["timing"]["queue_ms"],
                        "client_ms": client_ms,
                        "input_tokens": reply.get("usage", {}).get("inputTokens"),
                    }
                    out.write(json.dumps(record) + "\n")
                    if index % 100 == 99:
                        out.flush()
                        rate = (time.perf_counter() - started) / (index + 1)
                        print(f"  {index + 1}/{len(jobs)} ({rate:.2f} s each)", file=sys.stderr,
                              flush=True)
            print(f"{run_id}: finished in {time.perf_counter() - started:.0f} s", file=sys.stderr,
                  flush=True)


# ---------------------------------------------------------------- metrics


def wilson(hits: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def f1_table(pairs: list[tuple[set[str], set[str]]]) -> dict[str, Any]:
    """Per-label precision/recall/F1 over (gold set, predicted set) pairs."""
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for gold, pred in pairs:
        for label in pred & gold:
            tp[label] += 1
        for label in pred - gold:
            fp[label] += 1
        for label in gold - pred:
            fn[label] += 1
    labels = sorted(set(tp) | set(fp) | set(fn))
    per: dict[str, dict[str, float]] = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[label] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                      "support": tp[label] + fn[label]}
    micro_tp, micro_fp, micro_fn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    mr = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    return {
        "macro_f1": round(statistics.fmean(v["f1"] for v in per.values()), 4) if per else 0.0,
        "micro_f1": round(2 * mp * mr / (mp + mr), 4) if mp + mr else 0.0,
        "micro_precision": round(mp, 4), "micro_recall": round(mr, 4), "per_label": per,
    }


def reliability(points: list[tuple[float, bool]], bins: int = 10) -> dict[str, Any]:
    """Expected calibration error over equal-width confidence bins, and the bins themselves."""
    table = [{"low": i / bins, "high": (i + 1) / bins, "n": 0, "confidence": 0.0,
              "accuracy": 0.0} for i in range(bins)]
    for confidence, correct in points:
        b = table[min(int(confidence * bins), bins - 1)]
        b["n"] += 1
        b["confidence"] += confidence
        b["accuracy"] += float(correct)
    ece = 0.0
    for b in table:
        if b["n"]:
            b["confidence"] = round(b["confidence"] / b["n"], 4)
            b["accuracy"] = round(b["accuracy"] / b["n"], 4)
            ece += b["n"] / len(points) * abs(b["confidence"] - b["accuracy"])
    return {"ece": round(ece, 4), "bins": table}


def latency(records: list[dict[str, Any]]) -> dict[str, Any]:
    def q(values: list[float], f: float) -> float:
        values = sorted(values)
        return round(values[min(len(values) - 1, int(f * len(values)))], 1)

    inf = [r["inference_ms"] for r in records]
    cli = [r["client_ms"] for r in records]
    tok = [r["input_tokens"] for r in records if r["input_tokens"] is not None]
    return {"requests": len(records), "inference_ms_median": q(inf, 0.5),
            "inference_ms_p90": q(inf, 0.9), "client_ms_median": q(cli, 0.5),
            "client_ms_p90": q(cli, 0.9), "client_ms_total": round(sum(cli)),
            "input_tokens_median": q([float(t) for t in tok], 0.5) if tok else None}


def single_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    hits = sum(r["pred"] == r["gold"] for r in records)
    table = f1_table([(set(r["gold"]), set(r["pred"])) for r in records])
    points = []
    for r in records:
        probs = r["probabilities"] or {}
        label = r["pred"][0] if r["pred"] else max(probs, key=probs.get)
        points.append((probs.get(label, r["confidence"] or 0.0), r["pred"] == r["gold"]))
    return {"n": len(records), "accuracy": round(hits / len(records), 4),
            "accuracy_ci95": wilson(hits, len(records)), "macro_f1": table["macro_f1"],
            "per_label": table["per_label"], "calibration": reliability(points),
            **binary_auc(records)}


def binary_auc(records: Records) -> dict[str, Any]:
    """For two-label tasks: ROC AUC of the rarer label's probability (how well it ranks)."""
    labels = {label for r in records for label in (r["probabilities"] or {})}
    if len(labels) != 2:
        return {}
    counts = Counter(r["gold"][0] for r in records)
    positive = min(sorted(labels), key=lambda label: counts[label])
    scored = sorted((r["probabilities"][positive], r["gold"][0] == positive) for r in records)
    n_pos = sum(is_pos for _, is_pos in scored)
    n_neg = len(scored) - n_pos
    if not n_pos or not n_neg:
        return {}
    # Mann-Whitney U with average ranks for ties.
    rank_sum = 0.0
    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        rank_sum += average * sum(is_pos for _, is_pos in scored[i:j + 1])
        i = j + 1
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"auc": {"positive_label": positive, "value": round(auc, 4)}}


Records = list[dict[str, Any]]


def at_threshold(records: Records, threshold: float) -> list[tuple[set[str], set[str]]]:
    return [(set(r["gold"]), {label for label, p in r["probabilities"].items() if p >= threshold})
            for r in records]


def several_metrics(records: Records, tune: Records | None) -> dict[str, Any]:
    pairs = [(set(r["gold"]), set(r["pred"])) for r in records]
    default = f1_table(pairs)
    exact = sum(g == p for g, p in pairs)
    points = [(p, label in r["gold"]) for r in records for label, p in r["probabilities"].items()]
    out: dict[str, Any] = {
        "n": len(records),
        "default": {"micro_f1": default["micro_f1"], "macro_f1": default["macro_f1"],
                    "micro_precision": default["micro_precision"],
                    "micro_recall": default["micro_recall"],
                    "exact_match": round(exact / len(records), 4),
                    "exact_match_ci95": wilson(exact, len(records)),
                    "empty_predictions": sum(not p for _, p in pairs),
                    "per_label": default["per_label"]},
        "calibration": reliability(points),
    }
    if tune:
        grid = [round(0.01 * i, 2) for i in range(1, 100)]
        scored = [(f1_table(at_threshold(tune, t))["micro_f1"], t) for t in grid]
        best_f1, best = max(scored, key=lambda s: (s[0], -abs(s[1] - 0.5)))
        chosen = f1_table(at_threshold(records, best))
        pairs_best = at_threshold(records, best)
        out["tuned"] = {"threshold": best, "tune_n": len(tune), "tune_micro_f1": best_f1,
                        "micro_f1": chosen["micro_f1"], "macro_f1": chosen["macro_f1"],
                        "micro_precision": chosen["micro_precision"],
                        "micro_recall": chosen["micro_recall"],
                        "exact_match": round(sum(g == p for g, p in pairs_best) / len(records), 4)}
    return out


def baselines(records: list[dict[str, Any]], labels_of: dict[str, list[str]], several: bool,
              seed: int, tune: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Majority class (from the scored sample itself) and a seeded random guess."""
    rng = random.Random(f"{seed}/baseline")
    if several:
        freq: Counter[str] = Counter(g for r in (tune or records) for g in r["gold"])
        n_ref = len(tune or records)
        top = freq.most_common(1)[0][0]
        majority = f1_table([(set(r["gold"]), {top}) for r in records])
        rand = f1_table([(set(r["gold"]), {c for c in labels_of[r["key"]]
                                           if rng.random() < freq[c] / n_ref}) for r in records])
        return {"majority": {"rule": f"always '{top}'", "micro_f1": majority["micro_f1"],
                             "macro_f1": majority["macro_f1"]},
                "random": {"rule": "each label independently, with its tuning-split frequency",
                           "micro_f1": rand["micro_f1"], "macro_f1": rand["macro_f1"]}}
    top = Counter(r["gold"][0] for r in records).most_common(1)[0][0]
    maj = [(set(r["gold"]), {top}) for r in records]
    rnd = [(set(r["gold"]), {rng.choice(labels_of[r["key"]])}) for r in records]
    return {
        "majority": {"rule": f"always '{top}'",
                     "accuracy": round(sum(g == p for g, p in maj) / len(records), 4),
                     "macro_f1": f1_table(maj)["macro_f1"]},
        "random": {"rule": "uniform over the candidate labels",
                   "accuracy": round(sum(g == p for g, p in rnd) / len(records), 4),
                   "macro_f1": f1_table(rnd)["macro_f1"]},
    }


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def score_fast_decisions(records: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_domain[r["key"].split("/")[1]].append(r)
        by_head[r["key"]].append(r)
    domains = {d: round(sum(sorted(r["pred"]) == sorted(r["gold"]) for r in rs) / len(rs), 4)
               for d, rs in sorted(by_domain.items())}
    heads = {h: round(sum(sorted(r["pred"]) == sorted(r["gold"]) for r in rs) / len(rs), 4)
             for h, rs in sorted(by_head.items())}
    hits = sum(sorted(r["pred"]) == sorted(r["gold"]) for r in records)
    single = [r for r in records if not r["several"]]
    points = []
    for r in single:
        probs = r["probabilities"] or {}
        label = r["pred"][0] if r["pred"] else max(probs, key=probs.get)
        points.append((probs.get(label, 0.0), r["pred"] == r["gold"]))
    # Baselines per head: majority gold answer within the head, and a uniform random label.
    rng = random.Random(f"{seed}/baseline")
    maj_hits: dict[str, int] = Counter()
    rnd_hits: dict[str, int] = Counter()
    for head, rs in by_head.items():
        top = Counter(tuple(sorted(r["gold"])) for r in rs).most_common(1)[0][0]
        domain = head.split("/")[1]
        for r in rs:
            maj_hits[domain] += tuple(sorted(r["gold"])) == top
            labels = sorted(r["probabilities"])
            guess = ([c for c in labels if rng.random() < 0.5] if r["several"]
                     else [rng.choice(labels)])
            rnd_hits[domain] += sorted(guess) == sorted(r["gold"])
    n_domain = {d: len(rs) for d, rs in by_domain.items()}
    return {
        "n_rows": len({r["id"].split("/")[0] for r in records}), "n_decisions": len(records),
        "domain_average": round(statistics.fmean(domains.values()), 4),
        "overall_accuracy": round(hits / len(records), 4),
        "overall_ci95": wilson(hits, len(records)),
        "domains": domains, "heads": heads,
        "single_label_calibration": reliability(points),
        "baselines": {
            "majority": {"rule": "most common gold answer per head",
                         "domain_average": round(statistics.fmean(
                             maj_hits[d] / n_domain[d] for d in n_domain), 4)},
            "random": {"rule": "uniform label per single-label head; each label with p=0.5 "
                               "on several-label heads",
                       "domain_average": round(statistics.fmean(
                           rnd_hits[d] / n_domain[d] for d in n_domain), 4)},
        },
    }


def score_run(name: str, model: str, mode: str, records: Records, tune: Records | None,
              seed: int) -> dict[str, Any]:
    entry: dict[str, Any] = {"task": name, "model": model, "descriptions": mode == "desc",
                             "latency": latency(records)}
    if name == "fast_decisions":
        entry.update(score_fast_decisions(records, seed))
        return entry
    labels_of = {r["key"]: sorted(r["probabilities"]) for r in records}
    if TASKS[name].get("several"):
        entry.update(several_metrics(records, tune))
        entry["baselines"] = baselines(records, labels_of, True, seed, tune)
        return entry
    entry.update(single_metrics(records))
    entry["baselines"] = baselines(records, labels_of, False, seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("group"):
            groups[r["group"]].append(r)
    if groups:
        entry["by_group"] = {g: round(sum(r["pred"] == r["gold"] for r in rs) / len(rs), 4)
                             for g, rs in sorted(groups.items())}
    return entry


def score(args: argparse.Namespace) -> None:
    """Score every run. A run on a subsample is also matched by every full run of the same task,
    restricted to the same ids, so the two are compared on the same examples."""
    samples = json.loads((args.out / "samples.json").read_text(encoding="utf-8"))
    pred_dir = args.work / "predictions"
    results: dict[str, Any] = {"seed": args.seed, "runs": {}}
    compact: list[str] = []
    args.out.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, tuple[str, str, str, str | None, Records]] = {}
    for path in sorted(pred_dir.glob("*.jsonl")):
        if path.name.endswith("--tune.jsonl"):
            continue
        task_part, model, mode, *rest = path.stem.split("--")
        name = task_part.replace("-", "/", 1) if task_part != "fast_decisions" else task_part
        subsample = rest[0] if rest else None
        records = load(path)
        if subsample:
            keep = set(samples["tasks"][name]["subsamples"][subsample])
            records = [r for r in records if r["id"].split("/")[0] in keep]
        if records:
            loaded[path.stem] = (name, model, mode, subsample, records)
    # Full runs restricted to each subsample that some run of the same task used.
    for run_id, (name, model, mode, subsample, records) in list(loaded.items()):
        if subsample:
            continue
        used = {s for n, _, _, s, _ in loaded.values() if n == name and s}
        for sub in sorted(used):
            if f"{run_id}--{sub}" not in loaded:
                keep = set(samples["tasks"][name]["subsamples"][sub])
                loaded[f"{run_id}--{sub}"] = (name, model, mode, sub, [
                    r for r in records if r["id"].split("/")[0] in keep])
    for run_id, (name, model, mode, subsample, records) in sorted(loaded.items()):
        tune_path = pred_dir / f"{run_id}--tune.jsonl"
        tune = load(tune_path) if tune_path.exists() else None
        entry = score_run(name, model, mode, records, tune, args.seed)
        if subsample:
            entry["subsample"] = subsample
        results["runs"][run_id] = entry
        if subsample and not (pred_dir / f"{run_id}.jsonl").exists():
            continue  # a restriction of a full run; its answers are already written
        # The model's answers only: ids point into samples.json, and no dataset text or gold
        # label is kept, so the committed file carries no data from the datasets themselves.
        for r in records:
            probs = r["probabilities"] or {}
            compact.append(json.dumps({
                "run": run_id, "id": r["id"], "pred": r["pred"],
                "p": round(max(probs.values()), 4) if probs else None,
                "ms": round(r["inference_ms"], 1)}, separators=(",", ":")))
        print(f"scored {run_id} ({len(records)})", file=sys.stderr)
    (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    with (args.out / "predictions.jsonl.gz").open("wb") as raw, \
            gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as packed:
        packed.write(("\n".join(compact) + "\n").encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("sample", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("/tmp/gr-work/datasets"))
        p.add_argument("--out", type=Path, required=True, help="committed results directory")
        p.add_argument("--work", type=Path, default=Path("/tmp/gr-work/accuracy"),
                       help="local directory for full predictions")
        p.add_argument("--seed", type=int, default=SEED)
        if name == "run":
            p.add_argument("--base", default="http://127.0.0.1:8102")
            p.add_argument("--model", required=True)
            p.add_argument("--descriptions", action="store_true")
            p.add_argument("--subsample", help="run on a named subsample, such as sub200")
            p.add_argument("tasks", nargs="+", choices=sorted(TASKS))
    args = parser.parse_args()
    if args.command == "sample":
        args.out.mkdir(parents=True, exist_ok=True)
        samples = make_samples(args.data, args.seed)
        (args.out / "samples.json").write_text(json.dumps(samples, indent=1) + "\n",
                                               encoding="utf-8")
        for name, entry in samples["tasks"].items():
            subs = {k: len(v) for k, v in entry.get("subsamples", {}).items()}
            print(f"{name}: {len(entry['eval'])} of {entry['population']} ({entry['method']})"
                  + (f" subsamples {subs}" if subs else ""))
    elif args.command == "run":
        run(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
