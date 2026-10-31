"""Generate multi-turn consultations from case cards with a local instruct model (Ollama; no external API).

No source has multi-turn consultations tied to a known disease, so we write them with a larger open model.
Each prompt holds:
  * the case card (disease, symptoms in lay words, persona);
  * a few symptoms the patient does NOT have, taken from the conditions most easily confused with this one,
    so that the doctor also asks questions answered with "no";
  * two or three real MedDialog patient messages as style examples;
  * the history-taking checklist.
The reply is constrained to JSON. Every dialogue is then checked: it must have the right structure, the
patient must not name its disease or claim symptoms off its card, the doctor must not name the disease, and
the patient must reveal at least two card symptoms. One model writes both sides from the card, so the doctor
is also checked for knowing what it has not been told (asking where the pain is before any pain was
mentioned, "you mentioned" about something the patient never said, "your pregnancy" before the patient said
she is pregnant), for asking where a symptom without a place is ("where do you feel the fever?") and for
asking about a symptom that cannot apply to the patient (testicle pain in a woman). An anxious patient must
sound worried at least once without the doctor raising it first. A rejected
dialogue is regenerated with a new seed, up to three attempts per dialogue.

Run the whole set from the model folder (resumable: finished dialogues are skipped on a restart):
    python -m simclinic.generate --cards data/processed/cards/cards.jsonl --out-dir data/processed/dialogues
"""
import itertools
import json
import random
import re
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cards import fits_persona, named_diseases
from .lexicon import answer_polarity, revealed_symptoms

SCHEMA = {
    "type": "object",
    "properties": {"turns": {"type": "array", "items": {
        "type": "object",
        "properties": {"role": {"type": "string", "enum": ["doctor", "patient"]}, "text": {"type": "string"}},
        "required": ["role", "text"]}}},
    "required": ["turns"],
}
CHECKLIST = ("main complaint, onset and duration, site (of a pain, rash or swelling the patient has described), "
             "character, radiation, associated symptoms, timing and pattern, what makes it better or worse, "
             "severity, and relevant history (medicines, sick contacts, travel)")
# A line of its own in the patient's style: inside the persona line the model mostly ignored "anxious".
TEMPERAMENT = {
    "anxious": "Is anxious: without being asked, says at least once that they are worried or scared about what "
               "it could be, and asks whether it is serious.",
    "terse": "Is terse: gives very short answers, often just a few words.",
    "chatty": "Is chatty: adds extra detail about daily life and work.",
    "calm": "Is calm: answers plainly and politely.",
}
LITERACY = {
    "low": "uses simple everyday words and may describe symptoms vaguely",
    "medium": "uses everyday words",
    "high": "describes symptoms precisely, but still in lay terms",
}
CLOSING = "Thank you, I have a clearer picture now."

PROMPT = """Write a realistic consultation between a general practitioner (GP) in Singapore and a patient.

THE PATIENT (the doctor does not know any of this)
- Condition: {disease}. The patient does not know this and never names it or any other diagnosis.
- Main complaint: {main}
- Other symptoms the patient has: {others}
- Symptoms the patient does NOT have: {negatives}
- Persona: {age}-year-old {sex} ({ethnicity}), {occupation}. Temperament: {temperament}. \
Health literacy: {literacy} ({literacy_note}).
- The symptoms started {onset}. Severity: {severity}.

THE DOCTOR
- Does not know the condition or the patient's symptoms, only what the patient has said so far. Opens by \
asking what brings the patient in.
- Asks ONE short question per turn, {min_q} to {max_q} questions in total, following this history-taking \
checklist for the complaints the patient describes: {checklist}.
- Asks where a pain is, what it feels like or whether it spreads ONLY if the patient has already said they \
have pain. If they have not, the doctor first asks whether they have any pain.
- Asks about symptoms that tell several possible conditions apart, including at least two of the symptoms \
the patient does NOT have, and that fit what the patient has said, never an unusual one out of nowhere. \
Never says "you mentioned" about something the patient has not said.
- Ends with one short closing remark that does not name any condition.
- Nobody uses names: the doctor does not call the patient Mr, Mrs or Ms anything.

THE PATIENT'S STYLE
- {temperament_note}
- Opens with the main complaint, maybe one more symptom, and reveals the others when asked. Over the \
consultation, reveals most of their symptoms.
- Mentions no symptom outside their list, not even small ones such as feeling hot, sweating, itching, \
tiredness or feeling unwell. If asked about anything else, says they don't have it.
- Names each symptom in words when revealing it, even when terse ("{example} too", not just a bare word).
- Talks in everyday spoken words, in short turns. Real patients write like the examples below (informal, \
hedging, sometimes with typos); copy the tone, not the content:
{examples}

Return JSON: {{"turns": [{{"role": "doctor", "text": "..."}}, {{"role": "patient", "text": "..."}}, ...]}}. \
Start with the doctor and end with the doctor's closing remark."""


def build_style_index(messages: list, lexicon, min_words: int = 15, max_words: int = 120) -> dict:
    """{symptom: [message index]} over patient messages of a usable length."""
    index = {}
    for i, text in enumerate(messages):
        if min_words <= len(text.split()) <= max_words:
            for s in {s for _, _, s in lexicon.find(text)}:
                index.setdefault(s, []).append(i)
    return index


def style_examples(card: dict, messages: list, index: dict, banned: list, rng: random.Random, k: int = 2) -> list:
    """k real patient messages that mention one of the card's symptoms and name none of the banned words."""
    pool = sorted({i for s in card["symptoms"] for i in index.get(s, [])})
    pattern = re.compile("|".join(rf"\b{re.escape(b.lower())}\b" for b in banned)) if banned else None
    picked = []
    for i in rng.sample(pool, min(len(pool), 50)):
        if not (pattern and pattern.search(messages[i].lower())):
            picked.append(messages[i])
        if len(picked) == k:
            break
    while len(picked) < k:                                  # rare: no symptom match, fall back to any message
        picked.append(messages[rng.randrange(len(messages))])
    return picked


def negative_symptoms(card: dict, conditions: dict, nb, lexicon, rng: random.Random, k: int = 4) -> list:
    """Symptoms of the conditions most easily confused with this card that the patient does not have.
    Symptoms that everyday speech does not tell apart from a card symptom are left out, so a patient with
    knee pain is never told to deny joint pain, and so are symptoms that cannot apply to the persona."""
    rivals = [d for d, _ in nb.rank({s: True for s in card["symptoms"]}, k=6) if d != card["disease"]][:4]
    pool = []
    for d in rivals:
        pool += [s for s in conditions[d].symptoms if s not in pool and not lexicon.on_card(s, card["symptoms"])
                 and not any(lexicon.on_card(c, [s]) for c in card["symptoms"])
                 and fits_persona(s, card["persona"])][:4]
    rng.shuffle(pool)
    return pool[:k]


def build_prompt(card: dict, lexicon, examples: list, negatives: list, min_q: int = 5, max_q: int = 9) -> str:
    p = card["persona"]
    return PROMPT.format(
        disease=card["disease"],
        main=lexicon.lay(card["symptoms"][0]),
        others="; ".join(lexicon.lay(s) for s in card["symptoms"][1:]),
        negatives="; ".join(lexicon.lay(s) for s in negatives),
        age=p["age"], sex={"F": "woman", "M": "man"}[p["sex"]], ethnicity=p["ethnicity"], occupation=p["occupation"],
        temperament=p["temperament"], temperament_note=TEMPERAMENT[p["temperament"]],
        literacy=p["health_literacy"], literacy_note=LITERACY[p["health_literacy"]],
        onset=p["onset"], severity=p["severity"], min_q=min_q, max_q=max_q, checklist=CHECKLIST,
        # the model copies this example into the dialogue, so it must be one of the card's own symptoms
        example="I've got " + lexicon.lay(card["symptoms"][-1]),
        examples="\n".join(f'{i + 1}. "{e}"' for i, e in enumerate(examples)))


NAME = re.compile(r",?\s*\b(?:Mr|Mrs|Ms|Miss|Mdm|Madam|Dr)\.?\s+[A-Z][a-z]+\b")


def parse_turns(raw: str) -> tuple:
    """Return (turns, fixes). Light repairs only: strip role prefixes and invented names ("Hello, Mrs. Tan,
    what brings you in?"), merge same-role neighbours, add the opener or the closing remark if the model left
    one out, and replace a last doctor question that nobody answers with a closing remark."""
    turns = json.loads(raw)["turns"]
    fixes, out = [], []
    for t in turns:
        text = re.sub(r"^\s*(?:doctor|patient|dr\.?)\s*:\s*", "", str(t.get("text", "")), flags=re.I).strip()
        if NAME.search(text):
            text = NAME.sub("", text).strip(" ,")
            text = text[:1].upper() + text[1:]
            fixes.append("removed name")
        if not text:
            continue
        if out and out[-1]["role"] == t["role"]:
            out[-1]["text"] += " " + text
            fixes.append("merged same-role turns")
        else:
            out.append({"role": t["role"], "text": text})
    if out and out[0]["role"] != "doctor":
        out.insert(0, {"role": "doctor", "text": "What brings you in today?"})
        fixes.append("added opener")
    if out and out[-1]["role"] != "doctor":
        out.append({"role": "doctor", "text": CLOSING})
        fixes.append("added closing")
    elif len(out) > 1 and out[-1]["text"].endswith("?"):
        out[-1] = {"role": "doctor", "text": CLOSING}
        fixes.append("replaced unanswered last question")
    return out, fixes


# Symptoms with no place on the body: "where do you feel the cough?" is not a question a GP asks.
NOT_LOCALISED = {
    "Abusing alcohol", "Anxiety and nervousness", "Apnea", "Breathing fast", "Chills", "Constipation", "Cough",
    "Coughing up sputum", "Decreased appetite", "Delusions or hallucinations", "Depression",
    "Depressive or psychotic symptoms", "Diarrhea", "Difficulty breathing", "Difficulty speaking",
    "Disturbance of memory", "Dizziness", "Drug abuse", "Excessive anger", "Excessive urination at night",
    "Fainting", "Fatigue", "Fears and phobias", "Feeling hot", "Feeling ill", "Fever", "Flatulence",
    "Flu-like syndrome", "Frequent urination", "Hesitancy", "Hoarse voice", "Hostile behavior", "Impotence",
    "Increased heart rate", "Insomnia", "Involuntary urination", "Irregular heartbeat", "Low self-esteem",
    "Low urine output", "Melena", "Nausea", "Palpitations", "Regurgitation", "Retention of urine", "Seizures",
    "Shortness of breath", "Sleepiness", "Sleepwalking", "Sneezing", "Sweating", "Temper problems", "Vomiting",
    "Vomiting blood", "Weight gain", "Wheezing"}
PAIN_WORD = re.compile(r"\b(?:pain|hurt|sore|tender|cramp|burn|sting|discomfort|uncomfortable|throb|stab|sharp"
                       r"|agony)|ache|aching", re.I)
# "the pain", "your stomach pain", "where does it hurt": the doctor takes a pain for granted
PRESUMES_PAIN = re.compile(r"\b(?:the|your|this|that)\s+(?:[a-z]+\s+){0,2}?(?:pain|ache|discomfort|soreness)\b"
                           r"|\bwhere\b[^?.]*\bhurts?\b", re.I)
# "how is your pregnancy going?" to a patient who never said she is pregnant; "are you pregnant?" is fine
PREGNANCY = re.compile(r"pregnan|\bbaby\b", re.I)
PRESUMES_PREGNANCY = re.compile(r"\b(?:your|the|this)\s+(?:pregnancy|baby)\b", re.I)
ASKS_PREGNANT = re.compile(r"\b(?:are|could|might|if)\s+you(?:'re| are)?\s+(?:currently\s+|possibly\s+)?(?:be\s+)?"
                           r"pregnant\b", re.I)
WHERE = re.compile(r"(?<!\btimes )(?<!\bplaces )\bwhere\b[^?.]*", re.I)
YOU_SAID = re.compile(r"\byou(?:'ve| have)?\s+(?:mentioned|said|told me|described|reported)\b[^,;?.!]*", re.I)
WORRY = re.compile(r"\b(?:worr\w*|scared|afraid|frighten\w*|nervous|concern\w*|anxious|serious|fear\w*"
                   r"|panick\w*|dangerous|something bad|is it bad)\b", re.I)


def doctor_problems(turns: list, lexicon) -> list:
    """Doctor questions that only someone who has read the card would ask, or nobody would: where a pain is
    before any pain was mentioned, "you mentioned" a symptom the patient never mentioned, "your pregnancy"
    before the patient said she is pregnant, and where a symptom without a place is ("where do you feel the
    dizziness?")."""
    problems, pain, pregnant, heard = [], False, False, set()
    for i, t in enumerate(turns[:-1]):                     # the closing remark asks nothing
        if t["role"] == "patient":
            heard |= {s for _, _, s in lexicon.find(t["text"])}
            pain = pain or bool(PAIN_WORD.search(t["text"])
                                or (PAIN_WORD.search(turns[i - 1]["text"]) and answer_polarity(t["text"])))
            pregnant = pregnant or bool(PREGNANCY.search(t["text"])
                                        or (ASKS_PREGNANT.search(turns[i - 1]["text"]) and answer_polarity(t["text"])))
            continue
        if not pain and PRESUMES_PAIN.search(t["text"]):
            problems.append("doctor presumes a pain nobody mentioned")
        if not pregnant and PRESUMES_PREGNANCY.search(t["text"]) and not ASKS_PREGNANT.search(t["text"]):
            problems.append("doctor presumes a pregnancy nobody mentioned")
        for m in WHERE.finditer(t["text"]):
            asked = {s for _, _, s in lexicon.find(m.group())}
            if asked and asked <= NOT_LOCALISED and not PAIN_WORD.search(m.group()):
                problems.append(f"doctor asks where: {sorted(asked)}")
        known = heard | set(revealed_symptoms(turns[:i], lexicon))
        for m in YOU_SAID.finditer(t["text"]):
            unheard = [s for _, _, s in lexicon.find(m.group()) if not (
                lexicon.on_card(s, known) or any(lexicon.on_card(k, [s]) for k in known))]
            if unheard:
                problems.append(f"doctor says the patient mentioned {unheard}")
    return list(dict.fromkeys(problems))


def check_dialogue(turns: list, card: dict, lexicon, disease_names: list, all_names: list,
                   min_q: int = 4, max_q: int = 12) -> tuple:
    """Return (turns, problems). A closing remark that names a condition is replaced with a neutral one."""
    problems = []
    n_q = sum(t["role"] == "doctor" for t in turns) - 1
    if not min_q <= n_q <= max_q:
        problems.append(f"{n_q} questions")

    leaks = [n for n in disease_names
             if not [(a, b) for a, b, _ in lexicon.find(n)] == [(0, len(n))]]   # names that are also symptoms are fine
    if any(named_diseases(t["text"], leaks) for t in turns[:-1] if t["role"] == "doctor"):
        problems.append("doctor names the disease")
    if named_diseases(" ".join(t["text"] for t in turns if t["role"] == "patient"), leaks):
        problems.append("patient names the disease")
    if named_diseases(turns[-1]["text"], all_names):
        turns = turns[:-1] + [{"role": "doctor", "text": CLOSING}]
    revealed = revealed_symptoms(turns, lexicon)
    off_card = sorted(s for s, present in revealed.items() if present and not lexicon.on_card(s, card["symptoms"]))
    if off_card:
        problems.append(f"off-card symptoms: {off_card}")
    found = [s for s in card["symptoms"] if revealed.get(s)]
    if len(found) < min(2, len(card["symptoms"])):
        problems.append(f"only {len(found)} card symptoms revealed")
    problems += doctor_problems(turns, lexicon)
    persona = card.get("persona")
    if persona:
        asked = {s for t in turns if t["role"] == "doctor" for _, _, s in lexicon.find(t["text"])}
        asked |= {"Problems during pregnancy"} if any(
            t["role"] == "doctor" and PREGNANCY.search(t["text"]) for t in turns) else set()
        unfit = sorted(s for s in asked if not fits_persona(s, persona))
        if unfit:
            problems.append(f"doctor asks about symptoms that cannot apply: {unfit}")
    # the worry has to come from the patient: "Are you worried?" "Yes" is the doctor reading the card
    if persona and persona.get("temperament") == "anxious" and not any(
            t["role"] == "patient" and WORRY.search(t["text"]) and not WORRY.search(turns[i - 1]["text"])
            for i, t in enumerate(turns)):
        problems.append("anxious patient never sounds worried unprompted")
    return turns, problems


def diagnosis_label(turns: list, card: dict, nb, lexicon, conditions: dict) -> list:
    """True disease first, then the two conditions naive Bayes ranks highest on what the dialogue revealed,
    among those that explain at least a third of the symptoms the patient has. With few symptoms revealed,
    the denials alone can rank a condition that explains none of them (depression -> tonsillitis), so a
    label can have fewer than three names."""
    revealed = revealed_symptoms(turns, lexicon)
    present = [s for s, has in revealed.items() if has]

    def explains(d):
        return sum(lexicon.on_card(s, conditions[d].symptoms) for s in present) >= max(1, len(present) / 3)

    others = (d for d, _ in nb.rank(revealed) if d != card["disease"] and explains(d))
    return [card["disease"], *itertools.islice(others, 2)]


DIALOGUES_PER_CARD = {"train": 3, "dev": 1, "test": 1}     # 1,610 x 3 + 2 x 202 ~ 5,200 dialogues


def plan(cards: list, per_card: dict = DIALOGUES_PER_CARD) -> list:
    """[(dialogue_id, card)] in a fixed pseudo-random order, so any prefix (a pilot run) covers all diseases."""
    items = [(f"{c['card_id']}-d{k}", c) for c in cards for k in range(per_card[c["split"]])]
    return sorted(items, key=lambda item: zlib.crc32(item[0].encode()))


class DialogueGenerator:
    """Holds what one generation call needs; `self(dialogue_id, card, attempt)` returns one checked record.

    The seed of each call depends only on (dialogue_id, attempt), so a run is reproducible and resumable."""

    def __init__(self, chat, conditions: dict, names: dict, lexicon, nb, messages: list, model: str = ""):
        self.chat, self.conditions, self.names, self.lexicon, self.nb = chat, conditions, names, lexicon, nb
        self.all_names = sorted({n for ns in names.values() for n in ns}, key=len, reverse=True)
        self.messages, self.index, self.model = messages, build_style_index(messages, lexicon), model

    def __call__(self, dialogue_id: str, card: dict, attempt: int) -> dict:
        seed = zlib.crc32(f"{dialogue_id}/{attempt}".encode())
        rng = random.Random(seed)
        names = self.names[card["disease"]]
        examples = style_examples(card, self.messages, self.index, names, rng, k=rng.choice([2, 3]))
        negatives = negative_symptoms(card, self.conditions, self.nb, self.lexicon, rng)
        prompt = build_prompt(card, self.lexicon, examples, negatives)
        raw = self.chat([{"role": "user", "content": prompt}], schema=SCHEMA, seed=seed)
        try:
            turns, fixes = parse_turns(raw)
            turns, problems = check_dialogue(turns, card, self.lexicon, names, self.all_names)
        except (ValueError, KeyError, TypeError, IndexError) as e:      # unreadable or empty JSON
            turns, fixes, problems = [], [], [f"unreadable reply: {e!r}"[:200]]
        record = {"dialogue_id": dialogue_id, "card_id": card["card_id"], "split": card["split"],
                  "disease": card["disease"], "turns": turns, "attempt": attempt, "seed": seed,
                  "model": self.model, "negatives": negatives, "fixes": fixes, "problems": problems}
        if problems:
            record["raw"] = raw
        else:
            record["diagnosis"] = diagnosis_label(turns, card, self.nb, self.lexicon, self.conditions)
        return record


def read_jsonl(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(generate, planned: list, out_dir, workers: int = 4, max_attempts: int = 3, log_every: int = 100) -> dict:
    """Generate every planned dialogue. Accepted dialogues are appended to generated.jsonl and failed attempts
    to rejected_attempts.jsonl as they finish, so a restart skips dialogues that are accepted or out of tries.
    A connection error leaves the dialogue for the next run."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_path, bad_path = out_dir / "generated.jsonl", out_dir / "rejected_attempts.jsonl"
    done = {r["dialogue_id"] for r in read_jsonl(ok_path)}
    tries = {}
    for r in read_jsonl(bad_path):
        tries[r["dialogue_id"]] = tries.get(r["dialogue_id"], 0) + 1
    todo = [(d, c) for d, c in planned if d not in done and tries.get(d, 0) < max_attempts]
    stats = {"planned": len(planned), "already done": len(planned) - len(todo), "accepted": 0, "gave up": 0,
             "calls": 0, "errors": 0}
    lock, start = threading.Lock(), time.time()
    print(f"{len(todo)} of {len(planned)} dialogues to generate with {workers} workers", flush=True)

    def work(item):
        dialogue_id, card = item
        for attempt in range(tries.get(dialogue_id, 0), max_attempts):
            try:
                record = generate(dialogue_id, card, attempt)
            except OSError as e:                                   # server down or timed out
                with lock:
                    stats["errors"] += 1
                    print(f"{dialogue_id}: {e!r}", flush=True)
                return
            with lock:
                with open(bad_path if record["problems"] else ok_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["calls"] += 1
            if not record["problems"]:
                break
        with lock:
            stats["accepted" if not record["problems"] else "gave up"] += 1
            n = stats["accepted"] + stats["gave up"]
            if n % log_every == 0 or n == len(todo):
                rate = (time.time() - start) / n
                print(f"{n}/{len(todo)} done, {stats['accepted']} accepted, {stats['calls']} calls, "
                      f"{rate:.1f} s per dialogue, ~{rate * (len(todo) - n) / 3600:.1f} h left", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, todo))
    return stats


def recheck(generate, cards: dict, out_dir) -> int:
    """Re-apply the current filter to accepted dialogues, after the filter or the lexicon changed. Dialogues
    that now fail move to rejected_attempts.jsonl, so the next run regenerates them; the others get their
    diagnosis label recomputed. Never run this while a generation run is writing to the same folder."""
    out_dir = Path(out_dir)
    ok_path, bad_path = out_dir / "generated.jsonl", out_dir / "rejected_attempts.jsonl"
    keep, moved = [], []
    for r in read_jsonl(ok_path):
        card = cards[r["card_id"]]
        turns, problems = check_dialogue(r["turns"], card, generate.lexicon, generate.names[card["disease"]],
                                         generate.all_names)
        if problems:
            moved.append({**r, "problems": problems, "rechecked": True})
        else:
            keep.append({**r, "turns": turns,
                         "diagnosis": diagnosis_label(turns, card, generate.nb, generate.lexicon, generate.conditions)})
    ok_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
    with open(bad_path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in moved)
    return len(moved)


def main(argv=None):
    import argparse

    from .cards import CONDITIONS, disease_names, load_conditions
    from .lexicon import Lexicon
    from .llm import OllamaChat
    from .naive_bayes import SymcatNaiveBayes

    p = argparse.ArgumentParser(description="Generate SimClinic consultations from case cards with a local "
                                            "Ollama model (start `ollama serve` first).")
    p.add_argument("--cards", required=True, help="cards JSONL with a split field (notebook 07)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--symcat", default="data/raw/symcat/symcat-801-diseases.csv")
    p.add_argument("--style", default="data/processed/diagnose-me/clean/train.jsonl",
                   help="cleaned MedDialog dialogues (notebook 02); patient messages become style examples")
    p.add_argument("--model", default="qwen2.5:14b-instruct-q4_K_M")
    p.add_argument("--workers", type=int, default=4, help="parallel requests; set OLLAMA_NUM_PARALLEL to match")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--limit", type=int, help="only the first N planned dialogues (a pilot run)")
    p.add_argument("--recheck", action="store_true",
                   help="first re-filter accepted dialogues with the current code and regenerate the ones that fail")
    args = p.parse_args(argv)

    conditions = load_conditions(Path(args.symcat))
    vocab = sorted({s for n in CONDITIONS for s in conditions[n].symptoms})
    lexicon = Lexicon.load(symptoms=vocab)
    nb = SymcatNaiveBayes({n: conditions[n].symptoms for n in CONDITIONS})
    messages = [d["turns"][0]["text"] for d in read_jsonl(args.style) if d["turns"][0]["role"] == "patient"]
    chat = OllamaChat(args.model, temperature=args.temperature, max_tokens=1500, num_ctx=3072)
    generate = DialogueGenerator(chat, conditions, {n: disease_names(conditions[n]) for n in CONDITIONS},
                                 lexicon, nb, messages, model=args.model)
    cards = read_jsonl(args.cards)
    if args.recheck:
        print(f"recheck: {recheck(generate, {c['card_id']: c for c in cards}, args.out_dir)} accepted dialogues "
              "now fail the filter and will be regenerated", flush=True)
    planned = plan(cards)[: args.limit]
    stats = run(generate, planned, args.out_dir, workers=args.workers, max_attempts=args.max_attempts)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
