"""Checks for the explanation/citation module and the refusal gate. Run with `pytest tests/` from the model folder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simclinic.doctors import DoctorTurn
from simclinic.explain import Citation, check_citation, explain, fact_grounding, symptom_sources, unsupported
from simclinic.lexicon import Lexicon
from simclinic.refusal import ANSWER, MESSAGES, SAFE_CLOSING, GuardedDoctor, KeywordGate, RefusalGate
from simclinic.symcat import Condition

LEX = Lexicon.load()
TABLE = {
    "Flu": Condition(name="Flu", slug="flu", symptoms={"Fever": 0.91, "Cough": 0.87, "Sore throat": 0.63}),
    "Gout": Condition(name="Gout", slug="gout", symptoms={"Foot or toe pain": 0.8, "Ankle swelling": 0.4}),
}
TURNS = [{"role": "doctor", "text": "What brings you in today?"},
         {"role": "patient", "text": "I've had a fever since Monday."},
         {"role": "doctor", "text": "Any cough?"}, {"role": "patient", "text": "Yes."},
         {"role": "doctor", "text": "Any sore throat?"}, {"role": "patient", "text": "No."}]


def test_sources_cite_the_turn_that_settled_each_symptom():
    src = symptom_sources(TURNS, LEX)
    assert src["Fever"][:2] == (True, 1)
    assert src["Cough"][:2] == (True, 3)
    assert src["Sore throat"][:2] == (False, 5)


def test_explanation_support_and_against():
    e = explain("Flu", TURNS, TABLE, LEX)
    assert [c.symptom for c in e.support] == ["Fever", "Cough"]          # sorted by Symcat percentage
    assert [c.symptom for c in e.against] == ["Sore throat"]
    assert "91% of patients" in e.render(LEX)
    assert unsupported("Gout", TURNS, TABLE, LEX) and not unsupported("Flu", TURNS, TABLE, LEX)


def test_citation_check_catches_wrong_citations():
    ok = explain("Flu", TURNS, TABLE, LEX).support[0]
    assert all(check_citation(ok, TURNS, "Flu", TABLE, LEX).values())
    wrong_turn = Citation("Fever", True, 0.91, 3, "Yes.")
    assert not check_citation(wrong_turn, TURNS, "Flu", TABLE, LEX)["patient said it"]
    wrong_pct = Citation("Fever", True, 0.5, 1, "I've had a fever since Monday.")
    assert not check_citation(wrong_pct, TURNS, "Flu", TABLE, LEX)["Symcat percentage correct"]
    doctor_turn = Citation("Fever", True, 0.91, 0, "What brings you in today?")
    assert not check_citation(doctor_turn, TURNS, "Flu", TABLE, LEX)["turn is a patient turn"]


def test_fact_grounding_flags_unsaid_and_reversed_facts():
    f = fact_grounding("The patient has a fever and a sore throat. No cough. Denies headache.", TURNS, LEX)
    bad = dict(f["hallucinated"])
    assert bad.get("Sore throat") is True        # the patient denied it
    assert bad.get("Cough") is False             # the patient has it
    assert "Headache" in bad                     # never discussed
    assert f["grounded"] == 1                    # the fever


def test_keyword_gate():
    kw = KeywordGate()
    assert kw.category("I have crushing chest pain right now") == "emergency"
    assert kw.category("What dose of this medicine can I take?") == "medication"
    assert kw.category("I've had a cough for three days") == ANSWER


class _Echo:
    def next_turn(self, history, force_diagnosis=False):
        return DoctorTurn(f"seen {len(history)} turns")


def test_guarded_doctor_refuses_and_hides_refused_exchanges():
    texts = ["I have a cough", "my throat hurts", "fever since monday", "it started yesterday",
             "how much ibuprofen can I take", "can you prescribe antibiotics", "what dose of panadol",
             "I can't breathe and my lips are blue", "chest pain crushing right now", "I took an overdose"]
    cats = [ANSWER] * 4 + ["medication"] * 3 + ["emergency"] * 3
    gate = RefusalGate.fit(texts * 3, cats * 3, C=10)
    doctor = GuardedDoctor(_Echo(), gate)
    h = [{"role": "doctor", "text": "What brings you in today?"}, {"role": "patient", "text": "what dose of ibuprofen"}]
    turn = doctor.next_turn(h)
    assert turn.text == MESSAGES["medication"] and turn.refused == "medication" and not turn.ends
    h += [{"role": "doctor", "text": turn.text}, {"role": "patient", "text": "I have a cough"}]
    # the refused exchange is dropped before the doctor sees the history
    assert doctor.next_turn(h).text == "seen 2 turns"


class _Closer:
    def __init__(self, text):
        self.text = text

    def next_turn(self, history, force_diagnosis=False):
        return DoctorTurn(self.text, diagnosis=["Flu", "Common cold", "Strep throat"])


def test_guarded_doctor_replaces_treatment_promises():
    gate = RefusalGate.fit(["I have a cough", "fever since monday", "what dose of panadol", "prescribe antibiotics"] * 3,
                           [ANSWER, ANSWER, "medication", "medication"] * 3, C=10)
    h = [{"role": "doctor", "text": "What brings you in today?"}, {"role": "patient", "text": "I have a cough"}]
    turn = GuardedDoctor(_Closer("Thanks. I'll prescribe some medication for that."), gate).next_turn(h)
    assert turn.text == SAFE_CLOSING and "prescribe" in turn.raw
    assert GuardedDoctor(_Closer("Thanks for letting me know."), gate).next_turn(h).text == "Thanks for letting me know."
