"""Prompts for the instruction-tuned doctor baselines (zero-shot, few-shot, chain-of-thought) and the
parsing of their replies back into a question or a top-3 differential."""
import difflib
import re
from dataclasses import dataclass

STRATEGIES = ("zero_shot", "few_shot", "cot")

SYSTEM = """You are a general practitioner in Singapore taking a patient's history in a short consultation. \
Find out which condition the patient has by asking one short, clear question at a time in plain language. \
Use a history-taking checklist (site, onset, character, radiation, associated symptoms, timing, \
exacerbating and relieving factors, severity), and prefer questions that best tell the likely conditions apart. \
Gather enough information before you diagnose: a typical consultation needs 5 to {max_questions} questions, \
and you may ask at most {max_questions}.

The patient's condition is one of these:
{candidates}

Reply in exactly one of these formats and write nothing else:
QUESTION: <one question for the patient>
{diagnosis_format}
Use condition names exactly as they are written in the list."""

DIAGNOSIS_FORMAT = "DIAGNOSIS: <most likely condition>; <second most likely>; <third most likely>"
COT_FORMAT = """When you are ready to diagnose, think step by step in this format:
FACTS: <the symptoms the patient has and the ones they denied>
REASONING: <compare the conditions that fit, and rule out the others with reasons>
DIAGNOSIS: <most likely condition>; <second most likely>; <third most likely>"""

FEW_SHOT_HEADER = "\n\nHere are example consultations that show the expected style:\n"
START = "[A new patient sits down. Start the consultation.]"
FORCE = "\n\n[The consultation time is up. Reply now with your final answer in the DIAGNOSIS format.]"
FORCE_COT = "\n\n[The consultation time is up. Reply now with FACTS, REASONING and DIAGNOSIS.]"
RETRY = "[Give your final answer now. Reply with the DIAGNOSIS line only.]"


def system_prompt(candidates: list, strategy: str, max_questions: int, examples: list | None = None) -> str:
    text = SYSTEM.format(max_questions=max_questions,
                         candidates="\n".join(f"- {c}" for c in candidates),
                         diagnosis_format=COT_FORMAT if strategy == "cot" else DIAGNOSIS_FORMAT)
    if strategy == "few_shot" and examples:          # CoT stays zero-shot so the two effects stay separate
        text += FEW_SHOT_HEADER + "\n\n".join(render_example(i + 1, ex) for i, ex in enumerate(examples))
    return text


def render_example(n: int, example: dict) -> str:
    lines = [f"Example {n}:"]
    for t in example["turns"][:-1]:                    # the last doctor turn is the closing remark
        lines.append(f"Doctor: QUESTION: {t['text']}" if t["role"] == "doctor" else f"Patient: {t['text']}")
    lines.append("Doctor: DIAGNOSIS: " + "; ".join(example["diagnosis"]))
    return "\n".join(lines)


def chat_messages(history: list, system: str, force: bool, cot: bool) -> list:
    """System prompt + dialogue as chat turns (doctor = assistant). The start note keeps user/assistant
    alternation for chat templates that require the first message to come from the user."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": START}]
    for t in history:
        if t["role"] == "doctor":
            messages.append({"role": "assistant", "content": f"QUESTION: {t['text']}"})
        else:
            messages.append({"role": "user", "content": t["text"]})
    if force:
        messages[-1] = {**messages[-1], "content": messages[-1]["content"] + (FORCE_COT if cot else FORCE)}
    return messages


@dataclass
class ParsedReply:
    question: str | None = None
    diagnosis: list | None = None       # canonical names, best first
    unmatched: list | None = None       # diagnosis items that matched no candidate
    reasoning: str | None = None


def _clean(text: str) -> str:
    return re.sub(r"[*_`#]", "", text).strip()


DIRECTIVE = re.compile(r"\b(QUESTION|FACTS|REASONING|DIAGNOSIS)\s*:", re.I)


def parse_reply(raw: str, matcher, want_diagnosis: bool = False) -> ParsedReply:
    """Act on the reply's first directive only: some models write a question and then role-play the
    patient's answer and their own diagnosis in the same reply. A reply that starts reasoning (FACTS /
    REASONING) but never reaches a valid DIAGNOSIS returns diagnosis=[]. With want_diagnosis (forced
    diagnosis) a DIAGNOSIS anywhere in the reply counts."""
    text = _clean(raw)
    first = DIRECTIVE.search(text)
    diagnosing = want_diagnosis or (first is not None and first.group(1).upper() != "QUESTION")
    if diagnosing:
        # The first DIAGNOSIS line with valid names: "Here's my diagnosis:" before the FACTS block has none.
        fallback = None
        for m in re.finditer(r"DIAGNOSIS\s*:", text, re.I):
            names, unmatched = _diagnosis_items(text[m.end():], matcher)
            parsed = ParsedReply(diagnosis=names, unmatched=unmatched, reasoning=text[: m.start()].strip() or None)
            if names:
                return parsed
            fallback = fallback or parsed
        if fallback or (first is not None and first.group(1).upper() != "QUESTION"):
            return fallback or ParsedReply(diagnosis=[], unmatched=[], reasoning=text)
    m = re.search(r"QUESTION\s*:\s*(.+)", text, re.I | re.S)
    if m:
        question = re.sub(r"^(?:QUESTION\s*:\s*)+", "", m.group(1).strip(), flags=re.I).split("\n")[0]
    else:                                   # no directive: the first sentence that asks something
        asks = re.findall(r"[^.!?\n]*\?", text)
        question = asks[0] if asks else text.split("\n")[0]
    question = re.split(r"\s*\[", question)[0]            # Mistral appends "[If the answer is yes ..."
    if "?" in question:
        question = question[: question.rfind("?") + 1]
    return ParsedReply(question=question.strip() or "Could you tell me more about how you are feeling?")


def _diagnosis_items(body: str, matcher):
    body = re.split(r"\n\s*\n|\b(?:QUESTION|FACTS|REASONING)\s*:", body.strip(), flags=re.I)[0]
    items = [x for x in re.split(r"\s*(?:;|\n|\b\d+[.)]\s)\s*", body) if x.strip(" .,-")]
    if len(items) == 1 and "," in items[0]:
        items = items[0].split(",")
    names, unmatched = [], []
    for item in items[:5]:
        name = matcher(item.strip(" .,-[]"))
        if name and name not in names:
            names.append(name)
        elif not name:
            unmatched.append(item.strip())
    return names[:3], unmatched


class DiseaseMatcher:
    """Map a free-text condition name to a canonical candidate via aliases, then fuzzy matching."""

    def __init__(self, aliases: dict):
        """aliases: {canonical name: [names it may be written as]}"""
        self.lookup = {}
        for canonical, names in aliases.items():
            for n in [canonical, *names]:
                self.lookup.setdefault(self._key(n), canonical)

    @staticmethod
    def _key(text: str) -> str:
        text = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("'s", "s"))
        return re.sub(r"^(?:an?|the)\s+", "", " ".join(text.split()))

    def __call__(self, text: str):
        variants = [text, re.sub(r"\s*\([^)]*\)", "", text), *re.findall(r"\(([^)]+)\)", text)]
        for v in variants:
            if (k := self._key(v)) in self.lookup:
                return self.lookup[k]
        for v in variants:
            close = difflib.get_close_matches(self._key(v), list(self.lookup), n=1, cutoff=0.9)
            if close:
                return self.lookup[close[0]]
        key = f" {self._key(text)} "
        inside = [k for k in self.lookup if len(k) >= 5 and f" {k} " in key]
        return self.lookup[max(inside, key=len)] if inside else None
