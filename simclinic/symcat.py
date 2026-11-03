"""Parse the raw Symcat export (symcat-801-diseases.csv) into condition -> symptom probabilities.

The CSV is a flattened web scrape: every row holds one (condition, symptom, percentage) link or one
demographic odds value, and the column a field lands in shifts between rows. We therefore anchor on the
Symcat URLs instead of column positions: the cell before a symptom URL is the symptom name and the cell
after it is the percentage.
"""
import csv
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/teliov/symcat-to-synthea/master/symcat/"
FILES = ("symcat-801-diseases.csv", "symcat-474-symptoms.csv")

CONDITION_URL = re.compile(r"http://www\.symcat\.com/conditions/([^?/]+)$")
SYMPTOM_URL = re.compile(r"http://www\.symcat\.com/symptoms/(.+)$")
DEMOGRAPHIC_URL = re.compile(r"http://www\.symcat\.com/demographics/(age|sex)-(.+)$")


@dataclass
class Condition:
    name: str
    slug: str
    symptoms: dict = field(default_factory=dict)   # symptom name -> P(symptom | condition), in [0, 1]
    synonyms: list = field(default_factory=list)
    age_odds: dict = field(default_factory=dict)   # "15-29 years" -> odds ratio
    sex_odds: dict = field(default_factory=dict)   # "Male" / "Female" -> odds ratio


def download(raw_dir: Path) -> None:
    """Fetch the two Symcat CSVs into raw_dir if they are missing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (raw_dir / name).exists():
            urllib.request.urlretrieve(SOURCE_URL + name, raw_dir / name)


def _synonyms(cell: str) -> list:
    # "Also known as Streptococcal Pharyngitis and Streptococcal Sore Throat"
    body = cell.removeprefix("Also known as").strip()
    return [s.strip() for s in re.split(r",\s*(?:and\s+)?|\s+and\s+", body) if s.strip()]


def parse_conditions(csv_path: Path) -> dict:
    """Return {condition name: Condition}, symptoms ordered by percentage (highest first)."""
    conditions = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row = [c.strip() for c in row]
            cond, got_symptom = None, False
            for i, cell in enumerate(row):
                if cond is None and (m := CONDITION_URL.match(cell)):
                    name = row[i + 1] or row[i - 1]       # "<name>, <url>, <disorder name>"
                    cond = conditions.setdefault(name, Condition(name=name, slug=m[1].removesuffix("--2")))
                elif cond and cell.startswith("Also known as") and not cond.synonyms:
                    cond.synonyms = _synonyms(cell)
                elif cond and not got_symptom and SYMPTOM_URL.match(cell) and row[i + 1].isdigit():
                    cond.symptoms.setdefault(row[i - 1], int(row[i + 1]) / 100)
                    got_symptom = True
                elif m := DEMOGRAPHIC_URL.match(cell):
                    # "<group name>, <url>, <odds>x, <condition name>"
                    odds, name = row[i + 1].removesuffix("x"), row[i + 2]
                    try:
                        odds = float(odds)
                    except ValueError:
                        continue
                    target = conditions.setdefault(name, Condition(name=name, slug=""))
                    (target.age_odds if m[1] == "age" else target.sex_odds)[row[i - 1]] = odds
    for c in conditions.values():
        c.symptoms = dict(sorted(c.symptoms.items(), key=lambda kv: -kv[1]))
    return conditions


def passes_quality_filter(cond: Condition, max_shared=4, min_common=3, common_threshold=0.30) -> bool:
    """Proposal filter: no five symptoms share a percentage, and at least three are reported by >= 30%."""
    ps = list(cond.symptoms.values())
    if not ps:
        return False
    return max(Counter(ps).values()) <= max_shared and sum(p >= common_threshold for p in ps) >= min_common


def count_symptoms(csv_path: Path) -> int:
    """Number of distinct symptom pages in the symptom export (474 in the audit)."""
    slugs = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            for cell in row:
                if (m := SYMPTOM_URL.match(cell.strip())):
                    slugs.add(m[1])
                    break
    return len(slugs)
