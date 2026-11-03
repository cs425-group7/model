"""Rule-based patient that answers from a case card.

A stand-in for the fine-tuned patient model in auto mode: it understands questions only through the symptom
lexicon and a few history-taking patterns, and it never volunteers anything that is not on its card. Any
object with `reply(history) -> str` can replace it.
"""
import random
import re

ONSET_Q = re.compile(r"\b(when did|when was|when do|how long|start|started|began|begin|since when)\b")
SEVERITY_Q = re.compile(r"\b(how bad|severe|severity|scale|out of 10|out of ten|how much does it|intense|intensity)\b")
FACTORS_Q = re.compile(r"\b(worse|better|relieve|trigger|makes? it|helps?)\b")
ELSE_Q = re.compile(r"\b(anything else|any other|other symptoms|something else|else you)\b")
MEDS_Q = re.compile(r"\b(medication|medications|medicine|medicines|tablets|pills|taking anything|tried anything)\b")
HISTORY_Q = re.compile(r"\b(medical history|history of|before|previous|chronic|conditions?)\b")
SOCIAL_Q = re.compile(r"\b(smoke|smoking|alcohol|travel|travelled|traveled|contact|sick contacts|around you)\b")
CHARACTER_Q = re.compile(r"\b(describe|what kind|what type|feel like|sharp|dull|throbbing|constant|come and go)\b")
SITE_Q = re.compile(r"\b(where|which part|which side|point to|located|spread|radiate)\b")
# A yes/no question about something that is neither on the card nor in the lexicon: the patient does not have it.
YES_NO_Q = re.compile(r"^\s*(?:have|has|had|do|does|did|are|is|were|was|any|anything)\b")


def _join(items, word="and"):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} " + items[-1]


def _cap(text):
    return text[:1].upper() + text[1:]


class CardPatient:
    def __init__(self, card: dict, lexicon, seed: int = 0):
        self.card, self.lexicon = card, lexicon
        self.persona = card.get("persona", {})
        self.rng = random.Random(seed)
        self.disclosed = {}          # ground truth of what the patient has said: {symptom: True/False}

    def _flavour(self, text):
        temperament = self.persona.get("temperament")
        if temperament == "anxious" and self.rng.random() < 0.3:
            return text + self.rng.choice([" Is it something serious?", " Should I be worried?"])
        if temperament == "chatty" and self.rng.random() < 0.3:
            job = self.persona.get("occupation", "worker")
            return text + self.rng.choice([f" It's been hard to keep up with work as a {job}.",
                                           " I was hoping it would go away on its own."])
        return text

    def _opening(self):
        top = self.card["symptoms"][: self.rng.choice([1, 2, 2])]
        for s in top:
            self.disclosed[s] = True
        lay = [self.lexicon.lay(s) for s in top]
        onset = self.persona.get("onset", "a few days ago")
        temperament = self.persona.get("temperament")
        if temperament == "terse":
            return f"{_cap(_join(lay))}. Started {onset}."
        if temperament == "anxious":
            return f"Doctor, I'm quite worried. I've had {_join(lay)}, and it started {onset}."
        if temperament == "chatty":
            return self._flavour(f"Hi doctor. So it started {onset}. I've had {_join(lay)} and it's not getting better.")
        return f"I've had {_join(lay)}. It started {onset}."

    def _answer_symptoms(self, asked):
        pos = [s for s in asked if s in self.card["symptoms"]]
        neg = [s for s in asked if s not in self.card["symptoms"]]
        self.disclosed.update({s: True for s in pos} | {s: False for s in neg})
        lay_pos = [self.lexicon.lay(s) for s in pos]
        lay_neg = [self.lexicon.lay(s) for s in neg]
        terse = self.persona.get("temperament") == "terse"
        if pos and neg:
            return f"I've had {_join(lay_pos)}, but I haven't had {_join(lay_neg, 'or')}."
        if pos:
            if terse and len(pos) == 1:
                return self.rng.choice(["Yes.", "Yeah."])
            return self._flavour(self.rng.choice([f"Yes, I've had {_join(lay_pos)}.", f"Yeah, I have {_join(lay_pos)}."]))
        if terse or self.rng.random() < 0.3:
            return self.rng.choice(["No.", "Nope.", "Not really."])
        return f"No, I haven't had {_join(lay_neg, 'or')}."

    def reply(self, history: list) -> str:
        question = history[-1]["text"]
        if not any(t["role"] == "patient" for t in history):
            return self._opening()
        asked = list(dict.fromkeys(s for _, _, s in self.lexicon.find(question)))
        if asked:
            return self._answer_symptoms(asked)
        q = question.lower()
        severity = self.persona.get("severity", "moderate")
        if ELSE_Q.search(q):
            rest = [s for s in self.card["symptoms"] if s not in self.disclosed]
            if rest:
                self.disclosed[rest[0]] = True
                return f"Now that you mention it, I've also had {self.lexicon.lay(rest[0])}."
            return "No, I think that's everything."
        if SEVERITY_Q.search(q):
            return {"mild": "It's not too bad, maybe 3 out of 10.",
                    "moderate": "It's quite uncomfortable, maybe 5 or 6 out of 10.",
                    "severe": "It's really bad, about 8 out of 10."}[severity]
        if ONSET_Q.search(q):
            return f"It started {self.persona.get('onset', 'a few days ago')}."
        if MEDS_Q.search(q):
            return self.rng.choice(["Just some paracetamol, it didn't help much.", "Nothing regular, doctor."])
        if FACTORS_Q.search(q):
            return "Nothing really makes it better or worse."
        if SOCIAL_Q.search(q):
            return "No, nothing like that."
        if HISTORY_Q.search(q):
            return "No, I'm usually quite healthy."
        if CHARACTER_Q.search(q) or SITE_Q.search(q):
            return "It's hard to describe, it just feels uncomfortable."
        if YES_NO_Q.search(q):
            return "No, I don't think so."
        return "I'm not sure, doctor."
