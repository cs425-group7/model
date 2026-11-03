"""Auto mode: a doctor interviews a patient, and the result is scored against the hidden case card."""
from .lexicon import revealed_symptoms


def run_consultation(doctor, patient, max_questions: int = 10) -> dict:
    """Alternate doctor and patient turns until the doctor diagnoses or runs out of questions."""
    history = []
    while True:
        n_questions = sum(t["role"] == "doctor" for t in history)
        forced = n_questions >= max_questions
        turn = doctor.next_turn(history, force_diagnosis=forced)
        if turn.diagnosis is not None:
            return {"turns": history + [{"role": "doctor", "text": turn.text}], "diagnosis": turn.diagnosis,
                    "n_questions": n_questions, "forced": forced, "reasoning": turn.reasoning, "raw": turn.raw}
        history.append({"role": "doctor", "text": turn.text})
        history.append({"role": "patient", "text": patient.reply(history)})


def diagnose_transcript(doctor, turns: list) -> list:
    """Diagnosis from a fixed dialogue (its closing doctor remark removed): tests reasoning without asking."""
    history = turns[:-1] if turns and turns[-1]["role"] == "doctor" else turns
    return doctor.next_turn(history, force_diagnosis=True).diagnosis or []


def score(record: dict, card: dict, nb, lexicon) -> dict:
    """Diagnostic accuracy plus naive Bayes on the symptoms this dialogue revealed.

    nb_top1/nb_top3 show how well the evidence the doctor collected *could* be used, so comparing them with
    top1/top3 separates how well the doctor asks from how well it reasons.
    """
    dx, true = record["diagnosis"], card["disease"]
    evidence = revealed_symptoms(record["turns"], lexicon)
    nb_top = [d for d, _ in nb.rank(evidence, k=3)]
    on_card = set(card["symptoms"])
    return {"top1": dx[:1] == [true], "top3": true in dx[:3], "n_questions": record["n_questions"],
            "forced": record["forced"], "nb_top1": nb_top[:1] == [true], "nb_top3": true in nb_top,
            "symptom_recall": sum(bool(evidence.get(s)) for s in on_card) / len(on_card),
            "n_evidence": len(evidence), "no_diagnosis": not dx}


def summarise(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    cols = {"top1": "top-1", "top3": "top-3", "nb_top1": "NB top-1 (same evidence)",
            "nb_top3": "NB top-3 (same evidence)", "n_questions": "questions", "symptom_recall": "card symptoms found",
            "forced": "forced to diagnose", "no_diagnosis": "no valid diagnosis"}
    out = df.groupby("doctor", sort=False)[list(cols)].mean().rename(columns=cols)
    out.insert(0, "cases", df.groupby("doctor", sort=False).size())
    return out.round(3)
