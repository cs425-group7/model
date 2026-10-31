"""Fast checks for the shared doctor-model code. Run with `pytest tests/` from the model folder."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simclinic import cards as C
from simclinic.format import format_doctor_example
from simclinic.lexicon import Lexicon, revealed_symptoms
from simclinic.naive_bayes import SymcatNaiveBayes
from simclinic.prompts import DiseaseMatcher, parse_reply

LEX = Lexicon.load()


@pytest.mark.parametrize("text, expected", [
    ("I've had a cough but no fever", [("Cough", True), ("Fever", False)]),
    ("No, I haven't had a fever.", [("Fever", False)]),
    ("Nope, just the cough", [("Cough", True)]),
    ("I don't have a fever, cough or headache", [("Fever", False), ("Cough", False), ("Headache", False)]),
    ("I don't have a fever, I have a cough though", [("Fever", False), ("Cough", True)]),
    ("I have lower back pain and I've been throwing up blood", [("Low back pain", True), ("Vomiting blood", True)]),
    ("Not sure if I have a fever.", [("Fever", None)]),
    ("No fever, no swelling.", [("Fever", False)]),
    ("Yeah, sick to my stomach. Upper stomach mostly.", [("Nausea", True), ("Upper abdominal pain", True)]),
    ("it burns when I pee and I am peeing a lot", [("Painful urination", True), ("Frequent urination", True)]),
])
def test_mentions(text, expected):
    assert LEX.mentions(text) == expected


def test_yes_no_answers():
    turns = [{"role": "doctor", "text": "What brings you in today?"},
             {"role": "patient", "text": "I've had a sore throat since yesterday."},
             {"role": "doctor", "text": "Have you had any cough?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": "Any body aches?"}, {"role": "patient", "text": "Yeah, a bit."},
             {"role": "doctor", "text": "Any rash or ear pain?"}, {"role": "patient", "text": "Not really."},
             {"role": "doctor", "text": "Fever or chills?"}, {"role": "patient", "text": "Yes."}]
    assert revealed_symptoms(turns, LEX) == {"Sore throat": True, "Cough": False, "Ache all over": True,
                                             "Skin rash": False, "Ear pain": False}   # bare "Yes." to two is ambiguous


def test_naive_bayes_ranks_and_denials():
    nb = SymcatNaiveBayes({"A": {"x": 0.9, "y": 0.1}, "B": {"x": 0.9, "y": 0.8}})
    assert nb.rank({"x": True, "y": True})[0][0] == "B"
    assert nb.rank({"x": True, "y": False})[0][0] == "A"
    assert max(nb.information_gain({"x": True}), key=nb.information_gain({"x": True}).get) == "y"


def test_doctor_example_spans():
    turns = [{"role": "doctor", "text": "Hi?"}, {"role": "patient", "text": "Cough."},
             {"role": "doctor", "text": "Thanks."}]
    text, spans = format_doctor_example(turns, ["Flu", "Common cold", "Asthma"], "<|endoftext|>")
    assert text == "<|doctor|>Hi?<|patient|>Cough.<|doctor|>Thanks.<|diagnosis|>Flu; Common cold; Asthma<|endoftext|>"
    assert [text[a:b] for a, b in spans] == ["Hi?<|patient|>", "Thanks.<|diagnosis|>Flu; Common cold; Asthma<|endoftext|>"]


def test_parse_llm_reply():
    m = DiseaseMatcher({"Flu": ["Influenza"], "Common cold": ["URTI"], "Tension headache": ["Tension-type headache"]})
    r = parse_reply("FACTS: fever\nREASONING: ...\n**DIAGNOSIS:** 1. Influenza (Flu) 2. URTI 3. Pharyngitis", m)
    assert r.diagnosis == ["Flu", "Common cold"] and r.unmatched == ["Pharyngitis"] and r.reasoning.startswith("FACTS")
    assert parse_reply("QUESTION: Any cough?\n(extra)", m).question == "Any cough?"
    assert m("tension type headaches") == "Tension headache"
    # Only the first directive counts: a question followed by role-played answers and a diagnosis is a question.
    roleplay = parse_reply("QUESTION: Do you have a headache?\n[If yes ...]\nDIAGNOSIS: Flu; URTI", m)
    assert roleplay.question == "Do you have a headache?" and roleplay.diagnosis is None
    assert parse_reply("QUESTION: QUESTION: Any rash?", m).question == "Any rash?"
    assert parse_reply("REASONING: it could be several things", m).diagnosis == []   # reasoning cut off
    cot = parse_reply("Here's my diagnosis:\n\nFACTS: fever\nREASONING: ...\nDIAGNOSIS: Influenza; URTI", m)
    assert cot.diagnosis == ["Flu", "Common cold"] and cot.reasoning.startswith("Here's my")
    assert parse_reply("QUESTION: Any rash?[[[", m).question == "Any rash?"


def test_split_is_stratified_and_sized():
    cards = [{"card_id": f"{d}{i}", "disease": d} for d in "abcdefghij" for i in range(34)]
    C.split_cards(cards)
    n = {s: sum(c["split"] == s for c in cards) for s in ("train", "dev", "test")}
    assert n == {"train": 272, "dev": 34, "test": 34}
    assert all({c["split"] for c in cards if c["disease"] == d} == {"train", "dev", "test"} for d in "abcdefghij")


@pytest.mark.parametrize("text, expected", [
    ("my throat feels a bit sore", [("Sore throat", True)]),                       # degree word + spoken variant
    ("My eyes are really red and swollen", [("Eye redness", True), ("Swollen eye", True)]),
    ("pain around my bottom", [("Pain of the anus", True)]),                        # "the" / "my"
    ("My foot started hurting two days ago", [("Foot or toe pain", True)]),
    ("No chest tightness, just the stomach pain and vomiting",
     [("Chest tightness", False), ("Sharp abdominal pain", True), ("Vomiting", True)]),
    ("I'm anxious about what this could be", []),                                   # stop phrase: temperament
    ("It's worse when I sit for long periods", []),
    ("Not really, a bit anxious lately", [("Anxiety and nervousness", True)]),
    ("My neck has been hurting really badly", [("Neck pain", True)]),
    ("Sometimes I get sharp pains in my chest", [("Sharp chest pain", True)]),
])
def test_spoken_mentions(text, expected):
    assert LEX.mentions(text) == expected


def test_yes_no_answer_does_not_take_back_a_statement():
    turns = [{"role": "doctor", "text": "What brings you in?"}, {"role": "patient", "text": "A rash on my lip."},
             {"role": "doctor", "text": "Any rash elsewhere?"}, {"role": "patient", "text": "No, only the lip."}]
    assert revealed_symptoms(turns, LEX) == {"Skin rash": True}


def test_on_card():
    assert LEX.on_card("Fever", ["Fever"])
    assert LEX.on_card("Feeling hot", ["Fever", "Cough"])                 # generic word for a card symptom
    assert LEX.on_card("Lower abdominal pain", ["Suprapubic pain"])       # same complaint in lay speech
    assert not LEX.on_card("Fever", ["Cough"])
    assert not LEX.on_card("Feeling hot", ["Cough"])


def test_cards_are_coherent():
    from simclinic.symcat import Condition
    flu = Condition("Flu", "flu", symptoms={"Fever": 0.9, "Cough": 0.5, "Headache": 0.5, "Sore throat": 0.5,
                                           "Vomiting": 0.3, "Diarrhea": 0.2})
    knee = Condition("Osteoarthritis", "oa", symptoms={"Knee pain": 0.8, "Hip pain": 0.6, "Back pain": 0.5})
    cards = C.make_cards({"Flu": flu, "Osteoarthritis": knee}, ["Flu", "Osteoarthritis"], per_condition=200)
    assert all(c["symptoms"][0] == ("Fever" if c["disease"] == "Flu" else "Knee pain") for c in cards)
    assert all(3 <= len(c["symptoms"]) <= 6 for c in cards)
    assert {c["persona"]["onset"] for c in cards if c["disease"] == "Flu"} <= set(C.ONSETS["acute"])
    assert {c["persona"]["onset"] for c in cards if c["disease"] == "Osteoarthritis"} <= set(C.ONSETS["chronic"])
    assert not [c for c in cards if c["persona"]["occupation"] == "student" and c["persona"]["age"] > 26]


def test_cards_fit_the_persona():
    from simclinic.symcat import Condition
    pox = Condition("Chickenpox", "pox", symptoms={"Skin rash": 0.9, "Fever": 0.6, "Problems during pregnancy": 0.5,
                                                   "Pain in testicles": 0.5, "Irritable infant": 0.5, "Cough": 0.4})
    cards = C.make_cards({"Chickenpox": pox}, ["Chickenpox"], per_condition=300)
    def has(s, **who):
        return any(s in c["symptoms"] and all(c["persona"][k] == v for k, v in who.items()) for c in cards)
    assert has("Problems during pregnancy", sex="F") and has("Pain in testicles", sex="M")
    assert not has("Problems during pregnancy", sex="M") and not has("Pain in testicles", sex="F")
    assert not [c for c in cards if "Problems during pregnancy" in c["symptoms"] and c["persona"]["age"] > 50]
    assert not has("Irritable infant")


def test_generated_dialogue_repairs_and_checks():
    import json
    from simclinic import generate as G
    card = {"card_id": "x", "disease": "Flu", "symptoms": ["Fever", "Cough", "Headache"]}
    raw = json.dumps({"turns": [
        {"role": "doctor", "text": "Hello, Mrs. Tan, what brings you in today?"},
        {"role": "patient", "text": "I've had a fever and a cough. I feel hot all the time."},
        {"role": "doctor", "text": "Any headache?"}, {"role": "patient", "text": "Yes, a headache too."},
        {"role": "doctor", "text": "Any sore throat?"}, {"role": "patient", "text": "No."},
        {"role": "doctor", "text": "Any vomiting?"}, {"role": "patient", "text": "Nope."},
        {"role": "doctor", "text": "And how bad is it?"}]})
    turns, fixes = G.parse_turns(raw)
    assert turns[0]["text"] == "Hello, what brings you in today?"
    assert turns[-1]["text"] == G.CLOSING and fixes == ["removed name", "replaced unanswered last question"]
    _, problems = G.check_dialogue(turns, card, LEX, ["Flu", "Influenza"], ["Flu", "Influenza"])
    assert problems == []                          # "I feel hot" is covered by the fever on the card
    turns[3] = {"role": "patient", "text": "Yes, and I've been sweating. Is it the flu?"}
    _, problems = G.check_dialogue(turns, card, LEX, ["Flu", "Influenza"], ["Flu", "Influenza"])
    assert problems == ["patient names the disease", "off-card symptoms: ['Sweating']"]


def test_doctor_does_not_know_the_card():
    from simclinic.generate import doctor_problems
    def dialogue(*texts):
        return [{"role": "doctor" if i % 2 == 0 else "patient", "text": t} for i, t in enumerate(texts)]
    assert doctor_problems(dialogue("What brings you in?", "I've been bleeding from my bottom.",
                                    "Where exactly do you feel the pain?", "Around my bottom.", "Thanks."), LEX) \
        == ["doctor presumes a pain nobody mentioned"]
    assert doctor_problems(dialogue("What brings you in?", "My stomach hurts.",
                                    "Where exactly do you feel the pain?", "Lower down.", "Thanks."), LEX) == []
    assert doctor_problems(dialogue("What brings you in?", "My bowels are off.", "Any pain?", "Yes, a bit.",
                                    "Can you describe the pain?", "Crampy.", "Thanks."), LEX) == []
    assert doctor_problems(dialogue("What brings you in?", "I've had a fever and a cough.",
                                    "Where do you feel the cough the most?", "In my chest.",
                                    "Have there been times where you felt dizzy?", "No.", "Thanks."), LEX) \
        == ["doctor asks where: ['Cough']"]
    assert doctor_problems(dialogue("What brings you in?", "My ear feels blocked.",
                                    "You mentioned a cough and fever too. When did those start?", "Yesterday.",
                                    "You mentioned your ear feels blocked, any ringing?", "No.", "Thanks."), LEX) \
        == ["doctor says the patient mentioned ['Cough', 'Fever']"]
    assert doctor_problems(dialogue("What brings you in?", "I've got itching down there.",
                                    "Any problems with your pregnancy?", "Yes.", "Thanks."), LEX) \
        == ["doctor presumes a pregnancy nobody mentioned"]
    assert doctor_problems(dialogue("What brings you in?", "I've got itching down there.",
                                    "Are you currently pregnant, or any problems with the pregnancy?", "Yes.",
                                    "How is your pregnancy going?", "Fine.", "Thanks."), LEX) == []


def test_anxious_patient_sounds_worried():
    from simclinic import generate as G
    card = {"card_id": "x", "disease": "Flu", "symptoms": ["Fever", "Cough"],
            "persona": {"sex": "F", "age": 30, "temperament": "anxious"}}
    turns = [{"role": "doctor", "text": "What brings you in today?"},
             {"role": "patient", "text": "I've had a fever and a cough."},
             {"role": "doctor", "text": "Any sore throat?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": "Any vomiting?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": "Any rash?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": "Any headache?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": G.CLOSING}]
    never = ["anxious patient never sounds worried unprompted"]
    assert G.check_dialogue(turns, card, LEX, ["Flu"], ["Flu"])[1] == never
    turns[2:4] = [{"role": "doctor", "text": "Are you worried about what this could be?"},
                  {"role": "patient", "text": "Yes, I'm worried."}]
    assert G.check_dialogue(turns, card, LEX, ["Flu"], ["Flu"])[1] == never     # only when the doctor asks
    turns[5] = {"role": "patient", "text": "No. Is it something serious?"}
    assert G.check_dialogue(turns, card, LEX, ["Flu"], ["Flu"])[1] == []


def test_doctor_asks_only_what_can_apply():
    from simclinic import generate as G
    card = {"card_id": "x", "disease": "Flu", "symptoms": ["Fever", "Cough"],
            "persona": {"sex": "M", "age": 40, "temperament": "calm"}}
    turns = [{"role": "doctor", "text": "What brings you in today?"},
             {"role": "patient", "text": "I've had a fever and a cough."},
             {"role": "doctor", "text": "Any sore throat?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": "Any vomiting?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": "Any vaginal itching?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": "Could you be pregnant?"}, {"role": "patient", "text": "No."},
             {"role": "doctor", "text": G.CLOSING}]
    assert G.check_dialogue(turns, card, LEX, ["Flu"], ["Flu"])[1] == [
        "doctor asks about symptoms that cannot apply: ['Problems during pregnancy', 'Vaginal itching']"]
    card["persona"]["sex"] = "F"
    assert G.check_dialogue(turns, card, LEX, ["Flu"], ["Flu"])[1] == []


def test_negatives_and_label_fit_the_card():
    from simclinic import generate as G
    from simclinic.naive_bayes import SymcatNaiveBayes
    from simclinic.symcat import Condition
    conditions = {"Sprain": Condition("Sprain", "s", symptoms={"Knee pain": 0.9, "Ankle swelling": 0.5}),
                  "Gout": Condition("Gout", "g", symptoms={"Joint pain": 0.9, "Knee pain": 0.5, "Fever": 0.2,
                                                           "Pain in testicles": 0.1}),
                  "Flu": Condition("Flu", "f", symptoms={"Fever": 0.9, "Cough": 0.8, "Headache": 0.5})}
    nb = SymcatNaiveBayes({n: c.symptoms for n, c in conditions.items()})
    card = {"disease": "Sprain", "symptoms": ["Knee pain", "Ankle swelling"], "persona": {"sex": "F", "age": 30}}
    negatives = G.negative_symptoms(card, conditions, nb, LEX, __import__("random").Random(0))
    assert "Joint pain" not in negatives and "Fever" in negatives   # knee pain is a joint pain
    assert "Pain in testicles" not in negatives                     # and she has no testicles
    turns = [{"role": "doctor", "text": "What brings you in?"}, {"role": "patient", "text": "My knee hurts."},
             {"role": "doctor", "text": "Any cough or fever?"}, {"role": "patient", "text": "No cough, no fever."},
             {"role": "doctor", "text": "Thanks."}]
    assert G.diagnosis_label(turns, card, nb, LEX, conditions) == ["Sprain", "Gout"]   # flu explains no knee pain


def test_bppv_is_spelled_vertigo():
    assert "Benign paroxysmal positional vertigo (BPPV)" in C.CONDITIONS
    assert not [n for n in C.CONDITIONS if "vertical" in n]


def test_prompt_example_is_a_card_symptom():
    from simclinic import generate as G
    card = {"disease": "Tonsillitis", "symptoms": ["Sore throat", "Fever", "Difficulty in swallowing"],
            "persona": {"age": 30, "sex": "F", "ethnicity": "Chinese", "occupation": "teacher", "temperament": "terse",
                        "health_literacy": "low", "onset": "two days ago", "severity": "mild"}}
    prompt = G.build_prompt(card, LEX, ["my throat hurts"], ["Cough"])
    line = next(l for l in prompt.splitlines() if l.startswith("- Names each symptom"))
    assert {s for s, _ in LEX.mentions(line)} == {"Difficulty in swallowing"}  # models copy this example verbatim
