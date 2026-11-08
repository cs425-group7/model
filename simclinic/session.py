"""Human evaluation sessions in the terminal (the proposal's "human sessions"), independent of the app.

    python -m simclinic.session doctor  --name taeho     # you play the patient, the GPT-2 doctor interviews you
    python -m simclinic.session patient --name shiv      # you play the doctor, the GPT-2 patient has a hidden card

Doctor mode: you are shown a hidden test card to play (or pass --free to describe your own symptoms). Type your
answers; the doctor asks until it diagnoses, then shows its top 3 with cited reasons. The refusal gate is on, so
you can also test out-of-scope messages.
Patient mode: a random test card is drawn and hidden. Ask questions; type `/dx <condition>; <condition>; <condition>`
to give your differential. The card is then revealed and your answer scored.

Every session is appended to data/human_sessions/sessions.jsonl with the card, the turns and the scores.
Notebook 13 compares these sessions with auto mode.
"""
import argparse
import json
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "doctor"
LOG = ROOT / "data" / "human_sessions" / "sessions.jsonl"


def _load_common():
    from .lexicon import Lexicon
    from .prompts import DiseaseMatcher
    from .symcat import Condition
    from . import cards as C

    conditions = json.loads((DATA / "conditions.json").read_text())
    table = {n: Condition(**c) for n, c in json.loads((DATA / "symcat_conditions.json").read_text()).items()}
    lexicon = Lexicon.load(symptoms=sorted({s for n in conditions for s in table[n].symptoms}))
    matcher = DiseaseMatcher({n: C.disease_names(table[n]) for n in conditions})
    cards = [json.loads(l) for l in (DATA / "cards" / "test.jsonl").read_text().splitlines() if l.strip()]
    return conditions, table, lexicon, matcher, cards


def _describe(card, lexicon):
    p = card["persona"]
    return (f"{p['age']}, {p['sex']}, {p['ethnicity']}, {p['occupation']}, {p['temperament']}, onset {p['onset']}, "
            f"{p['severity']}\n  symptoms: {'; '.join(lexicon.lay(s) for s in card['symptoms'])}")


def doctor_mode(args):
    from .doctors import GPT2Doctor
    from .explain import explain
    from .refusal import GuardedDoctor

    conditions, table, lexicon, matcher, cards = _load_common()
    doctor = GPT2Doctor.from_pretrained(str(ROOT / "checkpoints" / "doctor" / "best"), conditions)
    gate_path = ROOT / "checkpoints" / "doctor" / "refusal_gate.pkl"
    if gate_path.exists():
        with open(gate_path, "rb") as f:
            doctor = GuardedDoctor(doctor, pickle.load(f))
    card = None if args.free else random.Random(args.seed).choice(cards)
    # The refusal gate misses some emergencies (notebook 11), so the contacts are always shown.
    print("\nPractice tool only, not medical advice. In an emergency call 995; "
          "Samaritans of Singapore (24h): 1767.")
    if card:
        print(f"\nPlay this patient (the doctor cannot see it):\n  {_describe(card, lexicon)}\n"
              "Answer only from the card; say no to anything else.\n")
    history, result = [], {}
    while True:
        forced = sum(t["role"] == "doctor" for t in history) >= args.max_questions
        turn = doctor.next_turn(history, force_diagnosis=forced)
        if getattr(turn, "refused", None):
            print(f"Doctor: {turn.text}")
            history.append({"role": "doctor", "text": turn.text, "refused": turn.refused})
            if turn.ends:
                break
        elif turn.diagnosis is not None:
            print(f"Doctor: {turn.text}\n\nTop 3: {'; '.join(turn.diagnosis)}\n")
            for c in turn.diagnosis:
                print(explain(c, history, table, lexicon).render(lexicon), "\n")
            history.append({"role": "doctor", "text": turn.text})
            result["diagnosis"] = turn.diagnosis
            break
        else:
            print(f"Doctor: {turn.text}")
            history.append({"role": "doctor", "text": turn.text})
        history.append({"role": "patient", "text": input("You: ").strip()})
    if card and "diagnosis" in result:
        result |= {"top1": result["diagnosis"][:1] == [card["disease"]], "top3": card["disease"] in result["diagnosis"]}
        print(f"Hidden disease: {card['disease']} -> {'correct' if result['top1'] else 'in top 3' if result['top3'] else 'wrong'}")
    return card, history, result


def patient_mode(args):
    from .patient import PatientSimulator

    conditions, table, lexicon, matcher, cards = _load_common()
    run = ROOT / "models" / "patient" / args.patient_run
    if not (run / "config.json").exists():
        sys.exit(f"patient checkpoint not found: {run} (copy notebook 08's run folder there)")
    sim = PatientSimulator(str(run))
    card = random.Random(args.seed).choice(cards)
    print("\nA patient is waiting. Ask questions; `/dx A; B; C` to diagnose (names from the list: `/list`).\n")
    history, result = [], {}
    while True:
        q = input("You (doctor): ").strip()
        if q == "/list":
            print("\n".join(conditions))
            continue
        if q.startswith("/dx"):
            names = [matcher(x.strip()) for x in q[3:].split(";") if x.strip()]
            result["diagnosis"] = [n for n in names if n][:3]
            break
        history.append({"role": "doctor", "text": q})
        reply = sim.reply(card, history)
        history.append({"role": "patient", "text": reply})
        print(f"Patient: {reply}")
    dx = result["diagnosis"]
    result |= {"top1": dx[:1] == [card["disease"]], "top3": card["disease"] in dx}
    print(f"\nHidden card: {card['disease']}\n  {_describe(card, lexicon)}")
    print(f"Your answer: {'; '.join(dx)} -> {'correct' if result['top1'] else 'in top 3' if result['top3'] else 'wrong'}")
    try:
        result["in_character"] = int(input("Rate the patient 1-5 (in character, consistent with the card): "))
    except ValueError:
        pass
    return card, history, result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["doctor", "patient"])
    ap.add_argument("--name", required=True, help="who is playing (kept in the log)")
    ap.add_argument("--seed", type=int, default=None, help="card seed (default: random)")
    ap.add_argument("--free", action="store_true", help="doctor mode: no card, describe your own symptoms")
    ap.add_argument("--max-questions", type=int, default=10)
    ap.add_argument("--patient-run", default="stage2-gpt2-medium-lr5e-05")
    args = ap.parse_args(argv)
    args.seed = args.seed if args.seed is not None else int(time.time())
    start = time.time()
    card, history, result = (doctor_mode if args.mode == "doctor" else patient_mode)(args)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "mode": args.mode, "player": args.name,
                            "seed": args.seed, "card": card, "turns": history, **result,
                            "minutes": round((time.time() - start) / 60, 1)}, ensure_ascii=False) + "\n")
    print(f"\nsaved to {LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
