"""Write ``descriptions.json``: one description per label, for the runs "with descriptions".

The descriptions are written once and kept in the repository, so every run uses
the same text. The customer-support-tickets set is non-commercial, so its card's
definitions are not kept here: ``fetch.py`` reads them from the pinned card into
the data directory. They come from one of two places, recorded per task:

- ``documentation``: the dataset's own definitions, quoted (GoEmotions paper
  appendix A, the Civil Comments paper).
- ``template``: a fixed sentence built from the label name and nothing else
  (underscores become spaces). No label gets hand-tuned wording.

    python -I benchmarks/accuracy/make_descriptions.py --data /tmp/gr-work/datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# GoEmotions, Demszky et al. 2020 (arXiv:2005.00547), appendix A "Emotion Definitions".
GOEMOTIONS = {
    "admiration": "Finding something impressive or worthy of respect.",
    "amusement": "Finding something funny or being entertained.",
    "anger": "A strong feeling of displeasure or antagonism.",
    "annoyance": "Mild anger, irritation.",
    "approval": "Having or expressing a favorable opinion.",
    "caring": "Displaying kindness and concern for others.",
    "confusion": "Lack of understanding, uncertainty.",
    "curiosity": "A strong desire to know or learn something.",
    "desire": "A strong feeling of wanting something or wishing for something to happen.",
    "disappointment": "Sadness or displeasure caused by the nonfulfillment of one's hopes or "
                      "expectations.",
    "disapproval": "Having or expressing an unfavorable opinion.",
    "disgust": "Revulsion or strong disapproval aroused by something unpleasant or offensive.",
    "embarrassment": "Self-consciousness, shame, or awkwardness.",
    "excitement": "Feeling of great enthusiasm and eagerness.",
    "fear": "Being afraid or worried.",
    "gratitude": "A feeling of thankfulness and appreciation.",
    "grief": "Intense sorrow, especially caused by someone's death.",
    "joy": "A feeling of pleasure and happiness.",
    "love": "A strong positive emotion of regard and affection.",
    "nervousness": "Apprehension, worry, anxiety.",
    "optimism": "Hopefulness and confidence about the future or the success of something.",
    "pride": "Pleasure or satisfaction due to ones own achievements or the achievements of "
             "those with whom one is closely associated.",
    "realization": "Becoming aware of something.",
    "relief": "Reassurance and relaxation following release from anxiety or distress.",
    "remorse": "Regret or guilty feeling.",
    "sadness": "Emotional pain, sorrow.",
    "surprise": "Feeling astonished, startled by something unexpected.",
    # The paper defines the 27 emotions only; this one follows the template rule.
    "neutral": "No emotion.",
}

# Borkan et al. 2019 (arXiv:1903.04561), abstract: "'Toxicity', defined as anything that is
# rude, disrespectful, or unreasonable that would make someone want to leave a conversation".
CIVIL = {
    "toxic": "Anything that is rude, disrespectful, or unreasonable that would make someone "
             "want to leave a conversation.",
    "not toxic": "Nothing that is rude, disrespectful, or unreasonable that would make someone "
                 "want to leave a conversation.",
}


def words(label: str) -> str:
    return label.replace("_", " ")


def labels_of(path: Path, field: str) -> list[str]:
    found: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)["task"][field]
            found.update(value if isinstance(value, list) else [value])
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("/tmp/gr-work/datasets"))
    args = parser.parse_args()
    out: dict[str, dict[str, object]] = {}

    def template(key: str, rule: str, labels: list[str], make: object) -> None:
        out[key] = {"source": "template", "rule": rule,
                    "labels": {label: make(label) for label in labels}}  # type: ignore[operator]

    template("banking77/intent", "The customer's message is about {label words}.",
             labels_of(args.data / "banking77/rows.jsonl", "intent"),
             lambda label: f"The customer's message is about {words(label)}.")
    template("news_category/topic", "A news story from the {Label In Title Case} section.",
             labels_of(args.data / "news_category/rows.jsonl", "topic"),
             lambda label: f"A news story from the {label.title()} section.")
    template("massive/scenario", "A request to a voice assistant about {label words}.",
             labels_of(args.data / "massive/rows.jsonl", "scenario"),
             lambda label: f"A request to a voice assistant about {words(label)}.")
    out["go_emotions/emotions"] = {
        "source": "documentation",
        "rule": "GoEmotions paper (arXiv:2005.00547) appendix A, verbatim; 'neutral' is not "
                "defined there and uses 'No emotion.'",
        "labels": GOEMOTIONS,
    }
    out["civil_comments/toxicity"] = {
        "source": "documentation",
        "rule": "Civil Comments paper (arXiv:1903.04561) abstract definition of toxicity; "
                "'not toxic' is its negation.",
        "labels": CIVIL,
    }
    heads: dict[str, set[str]] = {}
    with (args.data / "fast_decisions/rows.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for head, spec in row["heads"].items():
                heads.setdefault(f"fast_decisions/{row['group']}/{head}", set()).update(
                    spec["labels"])
    for key, labels in sorted(heads.items()):
        head = key.rsplit("/", 1)[1]
        template(key, "{head words}: {label words}", sorted(labels),
                 lambda label, head=head: f"{words(head)}: {words(label)}")

    for key, entry in out.items():
        labels = entry["labels"]
        assert isinstance(labels, dict)
        if key.startswith(("go_emotions", "civil_comments")):
            field = key.split("/")[1]
            expected = labels_of(args.data / f"{key.split('/')[0]}/rows.jsonl", field)
            assert sorted(labels) == expected, (key, sorted(labels), expected)
    target = HERE / "descriptions.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} label sets to {target}")


if __name__ == "__main__":
    main()
