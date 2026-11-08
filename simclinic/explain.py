"""Grounded explanations for a differential, and the citation check.

Retrieval: for each condition in the doctor's top 3, look up its Symcat entry and pair it with what the
patient actually said. Every claim in the explanation cites a source:

    Flu (Symcat):
      + fever      91% of Flu patients   - patient, turn 2: "I've had a fever since Monday"
      + cough      87%                   - patient, turn 4: "Yeah, a dry one."
      - sore throat 63%, denied          - patient, turn 6: "No."

The citation check verifies each citation against the dialogue and Symcat, so the same check can score
any doctor's stated reasons, including the FACTS an LLM writes in chain of thought:
    grounded fact      the patient's turns state this symptom with the same polarity
    hallucinated fact  the patient never said it (or said the opposite)
"""
import re
from dataclasses import dataclass, field

from .lexicon import answer_polarity


@dataclass
class Citation:
    symptom: str
    present: bool              # True = the patient has it, False = denied
    percent: float | None      # Symcat fraction (0-1) of the condition's patients with this symptom (None = not listed)
    turn: int | None           # index of the patient turn that states it
    quote: str | None


@dataclass
class Explanation:
    condition: str
    support: list = field(default_factory=list)       # revealed symptoms Symcat lists for the condition
    against: list = field(default_factory=list)       # denied symptoms Symcat lists at >= 30%
    unexplained: list = field(default_factory=list)   # revealed symptoms Symcat does not list for it

    def render(self, lexicon) -> str:
        lines = [f"{self.condition}:"]
        for c in self.support:
            lines.append(f"  + {lexicon.lay(c.symptom)} ({100 * c.percent:.0f}% of patients) - you said: \"{c.quote}\"")
        for c in self.against:
            lines.append(f"  - no {re.sub(r'^(?:an?|some)\s+', '', lexicon.lay(c.symptom))} ({100 * c.percent:.0f}% of patients have it) - you said: \"{c.quote}\"")
        for c in self.unexplained:
            lines.append(f"  ? {lexicon.lay(c.symptom)} is not typical of {self.condition}")
        return "\n".join(lines)


def symptom_sources(turns: list, lexicon) -> dict:
    """{symptom: (polarity, patient turn index, quote)} for every symptom the patient states, with the turn
    that settled it. Mirrors lexicon.revealed_symptoms, which decides polarity the same way."""
    out, asked = {}, []
    for i, t in enumerate(turns):
        if t["role"] == "doctor":
            asked = list(dict.fromkeys(s for _, _, s in lexicon.find(t["text"])))
            continue
        for symptom, polarity in lexicon.mentions(t["text"]):
            if polarity is not None:
                out[symptom] = (polarity, i, t["text"])
        polarity = answer_polarity(t["text"])
        for symptom in asked:
            if symptom not in out and (polarity is False or (polarity is True and len(asked) == 1)):
                out[symptom] = (polarity, i, t["text"])
        asked = []
    return out


def explain(condition: str, turns: list, symcat_table: dict, lexicon, against_min: float = 0.30) -> Explanation:
    listed = symcat_table[condition].symptoms
    exp = Explanation(condition)
    for symptom, (present, i, quote) in symptom_sources(turns, lexicon).items():
        pct = _percent(symptom, listed, lexicon)
        c = Citation(symptom, present, pct, i, quote)
        if present and pct is not None:
            exp.support.append(c)
        elif present:
            exp.unexplained.append(c)
        elif pct is not None and pct >= against_min:
            exp.against.append(c)
    exp.support.sort(key=lambda c: -c.percent)
    exp.against.sort(key=lambda c: -c.percent)
    return exp


def _percent(symptom, listed: dict, lexicon):
    """Symcat percentage for the symptom, or for a listed symptom everyday speech does not tell apart from it."""
    if symptom in listed:
        return listed[symptom]
    for s, p in listed.items():
        if lexicon.on_card(symptom, [s]) or lexicon.on_card(s, [symptom]):
            return p
    return None


def check_citation(c: Citation, turns: list, condition: str, symcat_table: dict, lexicon) -> dict:
    """A citation is valid when (1) the cited turn is a patient turn, (2) the lexicon reads the cited symptom in
    the dialogue with the cited polarity, settled at that turn, and (3) the stated Symcat percentage is the
    condition's real one."""
    sources = symptom_sources(turns, lexicon)
    in_turn = c.turn is not None and 0 <= c.turn < len(turns) and turns[c.turn]["role"] == "patient"
    stated = sources.get(c.symptom)
    said = stated is not None and stated[0] == c.present and stated[1] == c.turn
    pct = _percent(c.symptom, symcat_table[condition].symptoms, lexicon)
    return {"turn is a patient turn": in_turn, "patient said it": said,
            "Symcat percentage correct": c.percent is None and pct is None or (pct is not None and c.percent == pct)}


def fact_grounding(facts_text: str, turns: list, lexicon) -> dict:
    """Score free-text facts (an LLM's FACTS section) against the dialogue.

    Each symptom the lexicon finds in the facts, with its polarity, is grounded if the patient's turns state it
    with the same polarity (allowing symptoms everyday speech does not tell apart). Returns the counts and the
    hallucinated facts."""
    sources = symptom_sources(turns, lexicon)
    facts = {}
    for line in facts_text.splitlines() or [facts_text]:
        for symptom, polarity in lexicon.mentions(line):
            if polarity is not None:
                facts[symptom] = polarity
    grounded, wrong = [], []
    for symptom, polarity in facts.items():
        match = [p for s, (p, _, _) in sources.items()
                 if s == symptom or lexicon.on_card(s, [symptom]) or lexicon.on_card(symptom, [s])]
        (grounded if polarity in match else wrong).append((symptom, polarity))
    return {"facts": len(facts), "grounded": len(grounded), "hallucinated": wrong}


def unsupported(condition: str, turns: list, symcat_table: dict, lexicon) -> bool:
    """True when nothing the patient said supports the condition: no revealed symptom is listed for it."""
    return not explain(condition, turns, symcat_table, lexicon).support
