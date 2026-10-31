"""Case cards: a hidden disease, the 3-6 symptoms the patient has, and a persona.

The condition set is about 60 conditions a Singapore GP often sees: 59 Symcat conditions that pass the
quality filter, plus dengue and HFMD, whose cards the team wrote (`team_conditions.json`). A card is
    {"card_id", "disease", "symptoms": [Symcat names, presenting complaint first], "persona": {...}, "split"}
"""
import dataclasses
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from .symcat import Condition, parse_conditions

TEAM_CONDITIONS_PATH = Path(__file__).with_name("team_conditions.json")

# Common GP conditions that pass the Symcat quality filter. Croup was left out because the personas are adults.
SYMCAT_CONDITIONS = [
    "Flu", "Common cold", "Strep throat", "Acute sinusitis", "Acute otitis media",
    "Otitis externa (swimmer's ear)", "Tonsillitis", "Laryngitis", "Acute bronchitis", "Pneumonia",
    "Asthma", "Seasonal allergies (hay fever)", "Ear wax impaction", "Conjunctivitis", "Stye",
    "Mononucleosis", "Infectious gastroenteritis", "Gastroesophageal reflux disease (GERD)", "Gastritis",
    "Indigestion", "Irritable bowel syndrome", "Chronic constipation", "Hemorrhoids", "Appendicitis",
    "Gallstone", "Lactose intolerance", "Urinary tract infection", "Kidney stone", "Pyelonephritis",
    "Vaginal yeast infection", "Benign prostatic hyperplasia (BPH)", "Lumbago", "Sprain or strain", "Gout",
    "Osteoarthritis", "Carpal tunnel syndrome", "Muscle spasm", "Sciatica",
    "Lateral epicondylitis (tennis elbow)", "Migraine", "Tension headache",
    "Benign paroxysmal positional vertigo (BPPV)", "Eczema", "Contact dermatitis",
    "Fungal infection of the skin", "Chickenpox", "Shingles (herpes zoster)", "Impetigo", "Cold sore",
    "Insect bite", "Athlete's foot", "Herpangina", "Anxiety", "Depression", "Primary insomnia",
    "Panic attack", "Aphthous ulcer", "Heat exhaustion", "Viral exanthem",
]
TEAM_CONDITIONS = ["Dengue fever", "Hand, foot and mouth disease (HFMD)"]
CONDITIONS = SYMCAT_CONDITIONS + TEAM_CONDITIONS
# Symcat misspells BPPV. The slug, and so every card id, keeps Symcat's spelling.
RENAMED = {"Benign paroxysmal positional vertical (BPPV)": "Benign paroxysmal positional vertigo (BPPV)"}


def load_conditions(symcat_csv: Path, team_path: Path = TEAM_CONDITIONS_PATH) -> dict:
    """All parsed Symcat conditions, with the team cards added (the team's dengue card replaces Symcat's)."""
    conditions = {RENAMED.get(n, n): dataclasses.replace(c, name=RENAMED.get(n, n))
                  for n, c in parse_conditions(symcat_csv).items()}
    for name, card in json.loads(Path(team_path).read_text(encoding="utf-8")).items():
        if not name.startswith("_"):
            conditions[name] = Condition(name=name, slug=re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                                         symptoms=dict(sorted(card["symptoms"].items(), key=lambda kv: -kv[1])),
                                         synonyms=card.get("synonyms", []))
    return conditions


# Names a clinician (or an LLM) commonly uses that Symcat does not list as synonyms.
EXTRA_SYNONYMS = {
    "Common cold": ["Upper respiratory tract infection", "Upper respiratory infection", "URTI", "URI"],
    "Urinary tract infection": ["UTI"],
    "Acute otitis media": ["Middle ear infection", "Ear infection"],
    "Strep throat": ["Strep"],
    "Seasonal allergies (hay fever)": ["Allergic rhinitis"],
    "Infectious gastroenteritis": ["Gastroenteritis", "Viral gastroenteritis", "Stomach flu"],
    "Gastroesophageal reflux disease (GERD)": ["Acid reflux", "GORD"],
    "Irritable bowel syndrome": ["IBS"],
    "Tension headache": ["Tension-type headache"],
    "Primary insomnia": ["Insomnia"],
    "Fungal infection of the skin": ["Ringworm", "Tinea corporis"],
}

# Singapore resident population shares (rounded, Census 2020).
ETHNICITIES = {"Chinese": 0.74, "Malay": 0.135, "Indian": 0.09, "Others": 0.035}
ADULT_AGE_BUCKETS = {"15-29 years": (18, 29), "30-44 years": (30, 44), "45-59 years": (45, 59),
                     "60-74 years": (60, 74), "75+ years": (75, 88)}
OCCUPATIONS = ["teacher", "nurse", "accountant", "software engineer", "retail assistant", "taxi driver",
               "student", "hawker", "cleaner", "civil servant", "construction worker", "retiree",
               "sales executive", "security guard", "chef", "office administrator", "delivery rider"]
TEMPERAMENTS = ["anxious", "terse", "chatty", "calm"]
HEALTH_LITERACY = ["low", "medium", "high"]
ONSETS = {
    "acute": ["this morning", "last night", "yesterday", "two days ago", "three days ago", "about five days ago"],
    "subacute": ["last week", "about ten days ago", "two weeks ago", "about three weeks ago"],
    "chronic": ["about a month ago", "a couple of months ago", "about three months ago", "about six months ago",
                "more than a year ago"],
}
# When symptoms usually started by the time the patient sees a GP, so a cold does not start "a month ago".
COURSE = {
    **dict.fromkeys([
        "Flu", "Common cold", "Strep throat", "Acute otitis media", "Otitis externa (swimmer's ear)", "Tonsillitis",
        "Laryngitis", "Pneumonia", "Conjunctivitis", "Stye", "Infectious gastroenteritis", "Appendicitis",
        "Gallstone", "Urinary tract infection", "Kidney stone", "Pyelonephritis", "Vaginal yeast infection",
        "Sprain or strain", "Gout", "Muscle spasm", "Migraine", "Tension headache", "Contact dermatitis",
        "Chickenpox", "Shingles (herpes zoster)", "Impetigo", "Cold sore", "Insect bite", "Herpangina",
        "Panic attack", "Aphthous ulcer", "Heat exhaustion", "Viral exanthem", *TEAM_CONDITIONS], ("acute",)),
    **dict.fromkeys([
        "Acute sinusitis", "Acute bronchitis", "Mononucleosis", "Lumbago", "Sciatica",
        "Benign paroxysmal positional vertigo (BPPV)", "Ear wax impaction", "Gastritis", "Indigestion",
        "Fungal infection of the skin"], ("acute", "subacute")),
    **dict.fromkeys([
        "Lateral epicondylitis (tennis elbow)", "Athlete's foot", "Eczema", "Hemorrhoids", "Asthma",
        "Seasonal allergies (hay fever)", "Carpal tunnel syndrome", "Anxiety", "Depression", "Primary insomnia"],
        ("subacute", "chronic")),
    **dict.fromkeys([
        "Gastroesophageal reflux disease (GERD)", "Irritable bowel syndrome", "Chronic constipation",
        "Lactose intolerance", "Benign prostatic hyperplasia (BPH)", "Osteoarthritis"], ("chronic",)),
}
SEVERITIES = ["mild", "moderate", "severe"]
# Symcat counts symptoms per condition, not per patient, so a few only fit one sex or reproductive age
# (chickenpox lists both pregnancy problems and testicular pain).
SEX_ONLY = {
    "F": {"Long menstrual periods", "Painful menstruation", "Problems during pregnancy", "Vaginal discharge",
          "Vaginal itching", "Vaginal pain", "Vaginal redness", "Vulvar irritation"},
    "M": {"Impotence", "Pain in testicles", "Swelling of scrotum", "Symptoms of prostate"},
}
REPRODUCTIVE_AGE_ONLY = {"Long menstrual periods", "Painful menstruation", "Problems during pregnancy"}
INFANT_ONLY = {"Irritable infant", "Diaper rash", "Pulling at ears"}  # every persona is an adult


def _weighted(rng, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def sample_persona(cond, rng: random.Random) -> dict:
    age_w = {b: cond.age_odds.get(b, 1.0) for b in ADULT_AGE_BUCKETS}
    if sum(age_w.values()) == 0:
        age_w = dict.fromkeys(ADULT_AGE_BUCKETS, 1.0)
    lo, hi = ADULT_AGE_BUCKETS[_weighted(rng, age_w)]
    sex_w = {s: cond.sex_odds.get(s, 1.0) for s in ("Male", "Female")}
    if sum(sex_w.values()) == 0:
        sex_w = {"Male": 1.0, "Female": 1.0}
    age = rng.randint(lo, hi)
    onsets = [o for course in COURSE.get(cond.name, ("acute", "subacute")) for o in ONSETS[course]]
    return {"age": age, "sex": _weighted(rng, sex_w)[0], "ethnicity": _weighted(rng, ETHNICITIES),
            "occupation": "retiree" if age >= 65 else rng.choice([o for o in OCCUPATIONS if o != "retiree"
                                                                   and (o != "student" or age <= 26)]),
            "temperament": rng.choice(TEMPERAMENTS), "health_literacy": rng.choice(HEALTH_LITERACY),
            "onset": rng.choice(onsets), "severity": rng.choice(SEVERITIES)}


def fits_persona(symptom: str, persona: dict) -> bool:
    other = "M" if persona["sex"] == "F" else "F"
    return (symptom not in SEX_ONLY[other] and symptom not in INFANT_ONLY
            and not (symptom in REPRODUCTIVE_AGE_ONLY and persona["age"] > 50))


def sample_symptoms(cond, rng: random.Random, lo=3, hi=6, tries=2000, allowed=lambda s: True) -> list:
    """The condition's most common symptom, which the patient presents with, then each other symptom
    independently with its Symcat probability, resampled until lo..hi symptoms.

    Always keeping the most common symptom avoids incoherent cases (primary insomnia without trouble sleeping)
    that the generator cannot play without mentioning a symptom off the card."""
    (top, _), *rest = [(s, p) for s, p in cond.symptoms.items() if allowed(s)]
    for _ in range(tries):
        picked = [top] + [s for s, p in rest if rng.random() < p]
        if lo <= len(picked) <= hi:
            return picked
    return [top] + [s for s, _ in rest[:lo - 1]]


def make_cards(conditions: dict, names: list, per_condition: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    cards = []
    for name in names:
        cond = conditions[name]
        for k in range(per_condition):
            card_id = f"{cond.slug}-{k:03d}"
            symptoms, persona = sample_symptoms(cond, rng), sample_persona(cond, rng)
            if not all(fits_persona(s, persona) for s in symptoms):
                # resample from the card's own stream, so the shared stream (and every other card) is unchanged
                symptoms = sample_symptoms(cond, random.Random(f"{seed}/{card_id}"),
                                           allowed=lambda s: fits_persona(s, persona))
            cards.append({"card_id": card_id, "disease": name, "symptoms": symptoms, "persona": persona})
    return cards


def split_cards(cards: list, ratios=(0.8, 0.1, 0.1), seed: int = 42) -> list:
    """Stratified split by disease, so every disease appears in every split.

    Dev and test sizes are allocated by largest remainder, so the totals match the ratios
    (2,006 cards -> 201 test) even when every disease has the same number of cards.
    """
    rng = random.Random(seed)
    by_disease = defaultdict(list)
    for c in cards:
        by_disease[c["disease"]].append(c)
    groups = list(by_disease.values())
    counts = {}
    for split, ratio in (("test", ratios[2]), ("dev", ratios[1])):
        exact = [len(g) * ratio for g in groups]
        n = [max(1, int(x)) for x in exact]
        order = sorted(range(len(groups)), key=lambda i: (-(exact[i] - int(exact[i])), rng.random()))
        for i in order[: max(0, round(sum(exact)) - sum(n))]:
            n[i] += 1
        counts[split] = n
    for i, group in enumerate(groups):
        rng.shuffle(group)
        n_test, n_dev = counts["test"][i], counts["dev"][i]
        for j, c in enumerate(group):
            c["split"] = "test" if j < n_test else "dev" if j < n_test + n_dev else "train"
    return cards


def disease_names(condition) -> list:
    """Canonical name, its parts ("Gastroesophageal reflux disease", "GERD") and Symcat synonyms."""
    name = condition.name
    names = [name, *re.findall(r"\(([^)]+)\)", name), re.sub(r"\s*\([^)]*\)", "", name)]
    return list(dict.fromkeys([*names, *condition.synonyms, *EXTRA_SYNONYMS.get(name, [])]))


def named_diseases(text: str, names: list) -> list:
    """The names in `names` that appear in text as words ("flu-like" describes a symptom, not the flu)."""
    low = text.lower()
    return [n for n in names if len(n) >= 3 and re.search(rf"\b{re.escape(n.lower())}(?:s|es)?\b(?![- ]like)", low)]


def dialogue_problems(dialogue: dict, card: dict, lexicon, names: list) -> list:
    """Quality filter from the proposal: the patient must not name its disease or claim off-card symptoms."""
    from .lexicon import revealed_symptoms

    problems = []
    patient_text = " ".join(t["text"] for t in dialogue["turns"] if t["role"] == "patient")
    leaks = [n for n in names if [(a, b) for a, b, _ in lexicon.find(n)] != [(0, len(n))]]   # "Depression" is also a symptom
    problems += [f"patient names disease: {n}" for n in named_diseases(patient_text, leaks)]
    off_card = [s for s, present in revealed_symptoms(dialogue["turns"], lexicon).items()
                if present and not lexicon.on_card(s, card["symptoms"])]
    if off_card:
        problems.append(f"off-card symptoms: {off_card}")
    return problems
