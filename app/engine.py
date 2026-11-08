"""Everything the app needs from the shared simclinic package, with no Streamlit code,
so it can be tested and reused by the evaluation scripts.

Files it looks for (all optional except the Symcat CSV, which is downloaded if missing):
    data/raw/symcat/symcat-801-diseases.csv   Symcat export (simclinic.symcat.download fetches it)
    data/processed/cards/cards.jsonl          the team's 2,013 cards (notebook 07)
    checkpoints/doctor/best/                  fine-tuned GPT-2 doctor (notebook 04)
    models/patient/best/                      fine-tuned GPT-2 patient (notebook 08; copy the chosen run here)
    checkpoints/doctor/refusal_gate.pkl       refusal gate for doctor mode (notebook 11)
Environment overrides: SIMCLINIC_CARDS, SIMCLINIC_DOCTOR_CKPT, SIMCLINIC_PATIENT_CKPT, SIMCLINIC_GATE,
SIMCLINIC_LOG_DIR (set it to data/human_sessions to log into the same sessions.jsonl as simclinic.session).
"""
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from simclinic.cards import CONDITIONS, disease_names, load_conditions, make_cards, split_cards
from simclinic.consult import run_consultation, score
from simclinic.doctors import GPT2Doctor, NBDoctor
from simclinic.lexicon import Lexicon, revealed_symptoms
from simclinic.naive_bayes import SymcatNaiveBayes
from simclinic.patients import CardPatient
from simclinic.symcat import download

ROOT = Path(__file__).resolve().parents[1]
SYMCAT_DIR = ROOT / "data/raw/symcat"
CARDS_PATH = Path(os.environ.get("SIMCLINIC_CARDS", ROOT / "data/processed/cards/cards.jsonl"))
DOCTOR_CKPT = Path(os.environ.get("SIMCLINIC_DOCTOR_CKPT", ROOT / "checkpoints/doctor/best"))
PATIENT_CKPT = Path(os.environ.get("SIMCLINIC_PATIENT_CKPT", ROOT / "models/patient/best"))
GATE_PATH = Path(os.environ.get("SIMCLINIC_GATE", ROOT / "checkpoints/doctor/refusal_gate.pkl"))
LOG_DIR = os.environ.get("SIMCLINIC_LOG_DIR")
EMERGENCY_NOTE = ("Practice tool only, not medical advice. In an emergency call 995. "
                  "Samaritans of Singapore (24h): 1767.")   # same wording as simclinic.session
RULE_PATIENT, GPT2_PATIENT = "Rule-based (card)", "GPT-2 (fine-tuned)"
PATIENT_GEN = {"temperature": 0.8, "top_p": 0.9, "max_new_tokens": 80}   # notebook 08's GEN, as evaluated
MAX_QUESTIONS = 10   # same default as consult.run_consultation


@dataclass
class Resources:
    conditions: dict
    nb: SymcatNaiveBayes
    lexicon: Lexicon
    cards: list
    cards_source: str
    gpt2_doctor: object = None          # GPT2Doctor, or None if no checkpoint
    doctor_error: str | None = None
    doctor_names: list = field(default_factory=list)
    gpt2_patient: object = None         # simclinic.patient.PatientSimulator, or None if no checkpoint
    patient_error: str | None = None
    patient_names: list = field(default_factory=list)
    gate: object = None                 # simclinic.refusal.RefusalGate, or None if no gate file
    gate_error: str | None = None

    @property
    def case_pool(self) -> list:
        """Held-out cards, so the app never shows a case the models trained on."""
        return [c for c in self.cards if c.get("split") == "test"] or self.cards


def load_resources(load_gpt2: bool = True) -> Resources:
    download(SYMCAT_DIR)
    conditions = load_conditions(SYMCAT_DIR / "symcat-801-diseases.csv")
    # Same construction as notebooks 03/04.
    nb = SymcatNaiveBayes({n: conditions[n].symptoms for n in CONDITIONS})
    # sorted, exactly as notebooks 03-08: the order decides which symptom claims a shared phrase.
    lexicon = Lexicon.load(symptoms=sorted({s for n in CONDITIONS for s in conditions[n].symptoms}))

    if CARDS_PATH.exists():
        cards = [json.loads(l) for l in CARDS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        source = f"{len(cards)} team cards ({CARDS_PATH.name})"
    else:
        cards = split_cards(make_cards(conditions, CONDITIONS, per_condition=10))
        source = f"{len(cards)} stand-in cards sampled from Symcat (cards.jsonl not found)"

    res = Resources(conditions, nb, lexicon, cards, source)
    if load_gpt2 and DOCTOR_CKPT.exists():
        try:
            res.gpt2_doctor = GPT2Doctor.from_pretrained(str(DOCTOR_CKPT), candidates=CONDITIONS)
        except Exception as e:  # show the reason in the app instead of crashing it
            res.doctor_error = f"{type(e).__name__}: {e}"
    res.doctor_names = ["Naive Bayes (rule-based)"] + (["GPT-2 (fine-tuned)"] if res.gpt2_doctor else [])

    if load_gpt2 and PATIENT_CKPT.exists():
        try:
            from simclinic.patient import PatientSimulator   # imports torch, so only when needed
            res.gpt2_patient = PatientSimulator(str(PATIENT_CKPT))
        except Exception as e:
            res.patient_error = f"{type(e).__name__}: {e}"
    if GATE_PATH.exists():
        try:
            import pickle
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    # sklearn minor-version warnings on unpickling
                with open(GATE_PATH, "rb") as f:
                    res.gate = pickle.load(f)
        except Exception as e:
            res.gate_error = f"{type(e).__name__}: {e}"
    # The fine-tuned patient is the default once it exists.
    res.patient_names = ([GPT2_PATIENT] if res.gpt2_patient else []) + [RULE_PATIENT]
    return res


def make_doctor(res: Resources, name: str, seed: int = 0, guarded: bool = False):
    """guarded=True wraps the doctor in the team's GuardedDoctor (refusal gate), as simclinic.session does
    for human sessions. Only doctor mode uses it: auto mode stays identical to run_consultation."""
    if name.startswith("GPT-2") and res.gpt2_doctor is not None:
        doctor = res.gpt2_doctor
    else:
        doctor = NBDoctor(res.nb, res.lexicon, max_questions=MAX_QUESTIONS, seed=seed)
    if guarded and res.gate is not None:
        from simclinic.refusal import GuardedDoctor
        doctor = GuardedDoctor(doctor, res.gate)
    return doctor


def explanations(res: Resources, conditions: list, turns: list) -> list:
    """The team's grounded explanations (simclinic.explain): for each condition, the revealed symptoms
    Symcat lists for it (with percentages and the patient's words), and the common ones the patient denied."""
    from simclinic.explain import explain
    return [explain(c, turns, res.conditions, res.lexicon) for c in conditions if c in res.conditions]


class ModelPatient:
    """Adapts PatientSimulator.reply(card, history) to the reply(history) interface that
    run_consultation and the app use (the same one CardPatient has)."""

    def __init__(self, sim, card: dict):
        self.sim, self.card = sim, card

    def reply(self, history: list) -> str:
        return self.sim.reply(self.card, history, **PATIENT_GEN) or "Sorry, could you ask that again?"


def make_patient(res: Resources, card: dict, seed: int = 0, name: str | None = None):
    if name != RULE_PATIENT and res.gpt2_patient is not None:
        return ModelPatient(res.gpt2_patient, card)
    return CardPatient(card, res.lexicon, seed=seed)


def auto_consult(res: Resources, doctor, card: dict, seed: int = 0, patient_name: str | None = None) -> tuple:
    record = run_consultation(doctor, make_patient(res, card, seed, patient_name), max_questions=MAX_QUESTIONS)
    return record, score(record, card, res.nb, res.lexicon)


def auto_steps(res: Resources, doctor, card: dict, seed: int = 0, patient_name: str | None = None):
    """consult.run_consultation, step by step, so the app can show each turn as it is generated.

    Yields turn dicts as they happen, then a final ("done", record, scores). The loop and the record
    are identical to run_consultation (tests/test_app.py checks this), so the app shows exactly
    what the evaluation measures."""
    patient = make_patient(res, card, seed, patient_name)
    history = []
    while True:
        n_questions = sum(t["role"] == "doctor" for t in history)
        forced = n_questions >= MAX_QUESTIONS
        turn = doctor.next_turn(history, force_diagnosis=forced)
        if turn.diagnosis is not None:
            closing = {"role": "doctor", "text": turn.text}
            record = {"turns": history + [closing], "diagnosis": turn.diagnosis, "n_questions": n_questions,
                      "forced": forced, "reasoning": turn.reasoning, "raw": turn.raw}
            yield closing
            yield ("done", record, score(record, card, res.nb, res.lexicon))
            return
        history.append({"role": "doctor", "text": turn.text})
        yield history[-1]
        history.append({"role": "patient", "text": patient.reply(history)})
        yield history[-1]


def user_diagnosis_score(guesses: list, card: dict) -> dict:
    return {"top1": guesses[:1] == [card["disease"]], "top3": card["disease"] in guesses[:3]}


def symptom_feedback(res: Resources, turns: list, card: dict) -> dict:
    """What the patient revealed, against the card. Uses the team's revealed_symptoms / on_card."""
    revealed = revealed_symptoms(turns, res.lexicon)
    # A card symptom counts as found if the patient revealed it or one the lexicon treats as the same complaint
    # (on_card, e.g. "Headache" and "Frontal headache"), so one natural sentence covers both.
    said = [r for r, present in revealed.items() if present]
    found = [s for s in card["symptoms"] if any(r == s or res.lexicon.on_card(r, [s]) for r in said)]
    missed = [s for s in card["symptoms"] if s not in found]
    off_card = [s for s, present in revealed.items() if present and not res.lexicon.on_card(s, card["symptoms"])]
    return {"found": found, "missed": missed, "off_card": off_card}


def disease_leaked(res: Resources, turns: list, card: dict) -> list:
    from simclinic.cards import named_diseases
    text = " ".join(t["text"] for t in turns if t["role"] == "patient")
    names = disease_names(res.conditions[card["disease"]])
    # Same exclusion as cards.dialogue_problems: names that are also symptoms ("Depression").
    names = [n for n in names if [(a, b) for a, b, _ in res.lexicon.find(n)] != [(0, len(n))]]
    return named_diseases(text, names)


def log_session(record: dict) -> None:
    """Append one human session to $SIMCLINIC_LOG_DIR/sessions.jsonl, in the same record format as
    simclinic.session (time, mode, player, seed, card, turns, diagnosis, top1, top3, in_character, minutes),
    so notebook 13 can read app and terminal sessions together. "source" tells them apart."""
    if not LOG_DIR:
        return
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    record = {"time": datetime.now().isoformat(timespec="seconds"), **record, "source": "app"}
    with open(Path(LOG_DIR) / "sessions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
