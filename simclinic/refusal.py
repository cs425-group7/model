"""Refusal for doctor mode: messages the doctor bot must not try to handle.

The GPT-2 doctor only knows how to take a history and name one of 61 conditions. Some messages a real user
types are outside that: dosing and prescriptions, emergencies, test results, conditions it does not cover,
paperwork and non-medical requests. For those the safe reply is a short refusal that says where to go instead.
Emergencies end the consultation.

A gate classifies each patient message before the doctor sees it:

    gate = RefusalGate.fit(texts, categories)          # categories: "answer" or one of REFUSE_CATEGORIES
    doctor = GuardedDoctor(GPT2Doctor(...), gate)
    turn = doctor.next_turn(history)                   # a refusal turn if the last patient message is out of scope

Notebook 11 trains the gate and measures refusal precision and recall on 100 hand-written messages, half of
which should be refused.
"""
import re
from dataclasses import dataclass

from .doctors import DoctorTurn

ANSWER = "answer"
REFUSE_CATEGORIES = ["medication", "emergency", "test results", "out of scope", "admin or certainty", "non-medical"]

MESSAGES = {
    "medication": "I can't advise on medicines or doses here. Please ask a pharmacist or your own doctor before "
                  "you start, stop or change any medicine.",
    "emergency": "This sounds like it could be an emergency. Please call 995 or go to the nearest A&E now. "
                 "If you are thinking of harming yourself, call the Samaritans of Singapore on 1767.",
    "test results": "I can't interpret test results, scans or photos. Please go through them with the doctor "
                    "who ordered them.",
    "out of scope": "That is outside what this practice consultation covers. Please see a doctor about it.",
    "admin or certainty": "I can't do that here: this is a practice tool, not a real clinic, so I can't issue "
                          "certificates, referrals or guarantees.",
    "non-medical": "I can only help with this consultation. Shall we continue with your symptoms?",
}

# About 2% of the generated training consultations end with the doctor promising treatment ("I'll prescribe some
# medication"), and the fine-tuned doctor copies that in about 4% of test consultations. A practice tool must not
# promise treatment, so such closings are replaced.
TREATMENT = re.compile(r"\b(?:prescrib\w*|medicat\w*|medicines?|antibiotics?|tablets?|pills?|creams?|ointments?|"
                       r"treat(?:ment|ing|ed)?|dose)\b", re.I)
SAFE_CLOSING = ("Thank you. Based on what you've told me, these are the conditions I would consider. "
                "Please see a doctor or pharmacist before taking any treatment.")


class KeywordGate:
    """Hand-written patterns: the simple baseline a gate has to beat."""

    # Generic trigger words per category, written from the category definitions only (not from the test
    # messages), the way a quick production filter would be.
    PATTERNS = {
        "emergency": r"\b(?:emergency|ambulance|chest pain|heart attack|stroke|can'?t breathe|cannot breathe|"
                     r"suicid\w*|kill myself|end my life|overdose|unconscious|seizure|fit(?:ting)?|bleeding heavily|"
                     r"collapsed?)\b",
        "medication": r"\b(?:dose|dosage|prescri\w*|medicine|medication|pills?|tablets?|antibiotics?|mg|"
                      r"drug|painkillers?)\b",
        "test results": r"\b(?:test results?|blood test|x-?ray|scan|mri|ct|ecg|report|lab|photo|picture)\b",
        "out of scope": r"\b(?:pregnan\w*|cancer|tumou?r|diabetes|vaccin\w*|surgery|operation|diet|child|baby)\b",
        "admin or certainty": r"\b(?:mc|medical certificate|certificate|letter|insurance|referral|refer|appointment|"
                              r"book|guarantee|promise)\b",
        "non-medical": r"\b(?:weather|homework|essay|code|coding|stocks?|invest\w*|football|movie|recipe|joke|"
                       r"ignore (?:your|all|previous) instructions|system prompt)\b",
    }

    def __init__(self):
        self.compiled = {c: re.compile(p, re.I) for c, p in self.PATTERNS.items()}

    def category(self, text: str) -> str:
        for c, pattern in self.compiled.items():          # emergency first: it matters most
            if pattern.search(text):
                return c
        return ANSWER


class RefusalGate:
    """Word and character n-gram TF-IDF with a multinomial logistic regression over ANSWER + REFUSE_CATEGORIES.
    A message is refused when P(answer) < threshold."""

    def __init__(self, pipeline, threshold: float = 0.5):
        self.pipeline, self.threshold = pipeline, threshold

    @classmethod
    def build(cls, C: float = 4.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, make_pipeline

        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ])
        return make_pipeline(features, LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))

    @classmethod
    def fit(cls, texts: list, categories: list, C: float = 4.0, threshold: float = 0.5) -> "RefusalGate":
        pipeline = cls.build(C)
        pipeline.fit(texts, categories)
        return cls(pipeline, threshold)

    def p_refuse(self, texts: list):
        classes = list(self.pipeline.classes_)
        return 1 - self.pipeline.predict_proba(texts)[:, classes.index(ANSWER)]

    def categories(self, texts: list) -> list:
        """ANSWER, or the most likely refusal category for messages that are refused."""
        proba = self.pipeline.predict_proba(texts)
        classes = list(self.pipeline.classes_)
        ai = classes.index(ANSWER)
        out = []
        for row in proba:
            if 1 - row[ai] < self.threshold:
                out.append(ANSWER)
            else:
                out.append(max((p, c) for c, p in zip(classes, row) if c != ANSWER)[1])
        return out

    def category(self, text: str) -> str:
        return self.categories([text])[0]


@dataclass
class RefusalTurn(DoctorTurn):
    refused: str | None = None          # the refusal category
    ends: bool = False                  # an emergency ends the consultation


class GuardedDoctor:
    """Wraps any doctor: the gate reads the last patient message first, and the doctor only answers what the
    gate lets through. Refused exchanges stay in the user's history but are removed from the doctor's, so the
    doctor never conditions on them. A closing remark that promises treatment is replaced by SAFE_CLOSING."""

    def __init__(self, doctor, gate):
        self.doctor, self.gate = doctor, gate

    def next_turn(self, history, force_diagnosis=False):
        last = next((t["text"] for t in reversed(history) if t["role"] == "patient"), None)
        if last is not None and history and history[-1]["role"] == "patient":
            category = self.gate.category(last)
            if category != ANSWER:
                return RefusalTurn(MESSAGES[category], refused=category, ends=category == "emergency")
        kept = self._without_refused(history)
        turn = self.doctor.next_turn(kept, force_diagnosis=force_diagnosis)
        if turn.diagnosis is not None and TREATMENT.search(turn.text or ""):
            turn.raw, turn.text = turn.raw or turn.text, SAFE_CLOSING     # keep the original for logging
        return turn

    def _without_refused(self, history):
        """Drop refused exchanges (patient message + refusal) so the doctor sees a clean consultation."""
        refusals = set(MESSAGES.values())
        out = []
        for t in history:
            if t["role"] == "doctor" and t["text"] in refusals and out and out[-1]["role"] == "patient":
                out.pop()
                continue
            out.append(t)
        return out
