"""Download the accuracy-evaluation datasets and write a normalised copy of each.

Every dataset goes into its own new directory under ``--out`` (default
``/tmp/gr-work/datasets``): ``<id>/raw/`` holds the downloaded files exactly as
fetched, and ``<id>/rows.jsonl`` holds one ``{"id", "text", "task": {...}}``
object per example. Downloaded data is treated as untrusted: it is only parsed
(CSV, JSON lines, Parquet), never executed, and this script lives outside the
data directories.

Run with an interpreter that has ``pyarrow`` installed, in isolated mode:

    python -I benchmarks/accuracy/fetch.py --out /tmp/gr-work/datasets

Every file's SHA-256 is checked against ``manifest.json``. A file whose
manifest hash is ``null`` is reported with its hash so the manifest can be
completed; ``--strict`` turns that into an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"

# MASSIVE locale tag -> the language label used for the language task.
LANGUAGE_NAMES = {
    "en-US": "English",
    "es-ES": "Spanish",
    "fr-FR": "French",
    "de-DE": "German",
    "it-IT": "Italian",
    "pt-PT": "Portuguese",
    "nl-NL": "Dutch",
    "ru-RU": "Russian",
    "ja-JP": "Japanese",
    "zh-CN": "Chinese",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"user-agent": "gliner-runner-accuracy-fetch"})
    with urllib.request.urlopen(request, timeout=300) as reply, target.open("wb") as out:
        while block := reply.read(1 << 20):
            out.write(block)


def parquet_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Rows of a Parquet file and the ClassLabel names recorded in its Hugging Face metadata."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names: dict[str, list[str]] = {}
    meta = table.schema.metadata or {}
    if b"huggingface" in meta:
        features = json.loads(meta[b"huggingface"]).get("info", {}).get("features", {})
        for column, feature in features.items():
            if feature.get("_type") == "ClassLabel":
                names[column] = list(feature["names"])
            elif feature.get("_type") == "Sequence" and isinstance(feature.get("feature"), dict):
                inner = feature["feature"]
                if inner.get("_type") == "ClassLabel":
                    names[column] = list(inner["names"])
            elif isinstance(feature, list) and feature and feature[0].get("_type") == "ClassLabel":
                names[column] = list(feature[0]["names"])
    return table.to_pylist(), names


def clean(text: str) -> str:
    return " ".join(str(text).split())


def rows_banking77(raw: Path) -> Iterator[dict[str, Any]]:
    with (raw / "test.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            yield {"id": f"test-{index}", "text": clean(row["text"]),
                   "task": {"intent": row["category"]}}


def rows_fast_decisions(raw: Path, files: list[str]) -> Iterator[dict[str, Any]]:
    """One row per document. Unlike the other sets, each row carries its own heads and labels."""
    for name in files:
        domain = name.removesuffix(".jsonl")
        with (raw / name).open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                row = json.loads(line)
                heads = {
                    head["task"]: {
                        "labels": list(head["labels"]),
                        "gold": list(head["true_label"]),
                        "several": bool(head.get("multi_label", False)),
                    }
                    for head in row["output"]["classifications"]
                }
                # The input keeps its line breaks: the documents are laid out in fields.
                yield {"id": f"{domain}-{index}", "group": domain, "text": row["input"],
                       "heads": heads}


def rows_news_category(raw: Path) -> Iterator[dict[str, Any]]:
    with (raw / "data.json").open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            headline, summary = clean(row["headline"]), clean(row["short_description"])
            if not headline:
                continue
            text = f"{headline}. {summary}" if summary else headline
            yield {"id": f"row-{index}", "text": text, "task": {"topic": row["category"]}}


def rows_amazon_polarity(raw: Path) -> Iterator[dict[str, Any]]:
    rows, names = parquet_rows(raw / "test.parquet")
    for index, row in enumerate(rows):
        title, content = clean(row["title"]), clean(row["content"])
        text = f"{title}. {content}" if title else content
        yield {"id": f"test-{index}", "text": text,
               "task": {"sentiment": names["label"][row["label"]]}}


def rows_go_emotions(raw: Path) -> Iterator[dict[str, Any]]:
    for split in ("test", "validation"):
        rows, names = parquet_rows(raw / f"{split}.parquet")
        for row in rows:
            yield {"id": f"{split}-{row['id']}", "split": split, "text": clean(row["text"]),
                   "task": {"emotions": sorted(names["labels"][i] for i in row["labels"])}}


def rows_civil_comments(raw: Path) -> Iterator[dict[str, Any]]:
    rows, _ = parquet_rows(raw / "test.parquet")
    for index, row in enumerate(rows):
        text = clean(row["text"])
        if not text:
            continue
        label = "toxic" if row["toxicity"] >= 0.5 else "not toxic"
        yield {"id": f"test-{index}", "text": text, "task": {"toxicity": label},
               "score": row["toxicity"]}


def rows_massive(raw: Path, locales: list[str]) -> Iterator[dict[str, Any]]:
    for locale in locales:
        rows, names = parquet_rows(raw / f"{locale}.parquet")
        for row in rows:
            scenario = row["scenario"]
            if isinstance(scenario, int):
                scenario = names["scenario"][scenario]
            yield {"id": f"{locale}-{row['id']}", "group": locale, "text": clean(row["utt"]),
                   "task": {"scenario": scenario, "language": LANGUAGE_NAMES[locale]}}


def rows_support_tickets(raw: Path) -> Iterator[dict[str, Any]]:
    with (raw / "tickets.csv").open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            subject, body = clean(row.get("subject") or ""), clean(row.get("body") or "")
            if not body or not row.get("queue") or not row.get("priority"):
                continue
            text = f"{subject}. {body}" if subject else body
            yield {"id": f"row-{index}", "group": row["language"], "text": text,
                   "task": {"queue": row["queue"], "priority": row["priority"]}}


# The card's top priority is "3 (Critical)"; the data file labels it "high".
CARD_PRIORITY = {"1 (Low)": "low", "2 (Medium)": "medium", "3 (Critical)": "high"}


def card_section(card: str, heading: str) -> dict[str, str]:
    """The ``- <icon> **Name:** definition`` items under ``### <heading>`` in a dataset card."""
    section = card.split(f"### {heading}\n", 1)[1].split("\n### ", 1)[0]
    return {m[1]: m[2].strip() for m in re.finditer(r"^- .*?\*\*(.+?):\*\* (.+)$", section, re.M)}


def descriptions_support_tickets(raw: Path) -> dict[str, Any]:
    """The card's queue and priority definitions, verbatim, for the runs "with descriptions".

    The set is non-commercial, so this text is read from the card at fetch time and kept only
    in the data directory, never in the repository.
    """
    card = (raw / "README.md").read_text(encoding="utf-8")
    priority = {CARD_PRIORITY[name]: text for name, text in card_section(card, "Priority").items()}
    return {
        "support_tickets/queue": {
            "source": "documentation",
            "rule": "Dataset card, section Queue, verbatim.",
            "labels": card_section(card, "Queue"),
        },
        "support_tickets/priority": {
            "source": "documentation",
            "rule": "Dataset card, section Priority, verbatim (the card's '3 (Critical)' is "
                    "'high' in the data).",
            "labels": priority,
        },
    }


def normalise(entry: dict[str, Any], raw: Path) -> Iterator[dict[str, Any]]:
    dataset = entry["id"]
    if dataset == "massive":
        return rows_massive(raw, entry["locales"])
    if dataset == "fast_decisions":
        return rows_fast_decisions(raw, [item["name"] for item in entry["files"]])
    readers = {
        "banking77": rows_banking77,
        "news_category": rows_news_category,
        "amazon_polarity": rows_amazon_polarity,
        "go_emotions": rows_go_emotions,
        "civil_comments": rows_civil_comments,
        "support_tickets": rows_support_tickets,
    }
    return readers[dataset](raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/tmp/gr-work/datasets"))
    parser.add_argument("--only", nargs="*", help="dataset ids to fetch (default: all)")
    parser.add_argument("--strict", action="store_true", help="fail on a missing manifest hash")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = 0
    for entry in manifest["datasets"]:
        if args.only and entry["id"] not in args.only:
            continue
        home = args.out / entry["id"]
        raw = home / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        for item in entry["files"]:
            target = raw / item["name"]
            if not target.exists():
                print(f"{entry['id']}: downloading {item['url']}", file=sys.stderr)
                download(item["url"], target)
            actual = sha256_of(target)
            if item["sha256"] is None:
                print(f"{entry['id']}: {item['name']} sha256 {actual} (not in manifest)")
                failures += int(args.strict)
            elif item["sha256"] != actual:
                print(f"{entry['id']}: {item['name']} sha256 mismatch: {actual}", file=sys.stderr)
                failures += 1
        count = 0
        with (home / "rows.jsonl").open("w", encoding="utf-8") as out:
            for row in normalise(entry, raw):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        print(f"{entry['id']}: {count} rows -> {home / 'rows.jsonl'}")
        if entry["id"] == "support_tickets":
            descriptions = descriptions_support_tickets(raw)
            (home / "descriptions.json").write_text(
                json.dumps(descriptions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
