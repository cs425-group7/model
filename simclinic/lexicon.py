"""Symptom lexicon: Symcat names <-> lay phrasings, and extraction of revealed symptoms from a dialogue.

`revealed_symptoms()` is what the naive Bayes baseline reads: for every symptom the dialogue mentions it
decides whether the patient has it (True) or denied it (False). It combines two cues:
  * explicit mentions in patient turns, with a simple clause-level negation check
    ("I've had a cough but no fever" -> cough True, fever False);
  * yes/no answers to the doctor's question ("Any fever?" -> "Nope." -> fever False).

Matching is more forgiving than the phrase list alone:
  * "the" and "your" read as "my", so "pain around the bottom" also matches "pain around my bottom";
  * degree words are dropped before matching ("my throat is really sore" -> "my throat is sore");
  * every "<state> <part>" phrase also matches as "<part> is / feels / looks <state>" ("sore throat" ->
    "throat feels sore"), and every "<part> pain" phrase as "<part> hurts", "pain in my <part>", ... .
    Listed phrases always win over these generated variants.

`on_card()` is the symptom-fidelity test: a mention is on-card when the card has the symptom, or one that
everyday speech does not tell apart from it ("I feel hot" on a card with fever).
"""
import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("symptom_lexicon.json")
SUFFIXES = ("ing", "es", "ed", "s")          # a phrase also matches with these endings ("rash" -> "rashes")

NEGATION = re.compile(r"\b(?:no|not|never|without|nor|neither|none|deny|denies)\b|n't\b")
UNCERTAIN = re.compile(r"\b(?:not sure|unsure|don't know|perhaps|might)\b")
# Clause boundaries for negation scope. A comma only ends a clause when a new subject follows
# ("I don't have a fever, I have a cough"), so "no fever, cough or headache" stays one clause.
# "just" / "only" start the part a patient does have ("no chest pain, just the stomach").
CLAUSE_BREAK = re.compile(r"[.;!?]|\b(?:but|though|although|however|except|apart from|besides|other than"
                          r"|just|only)\b"
                          r"|,\s*(?=(?:i|i've|i'm|it|it's|my|she|he|they|we)\b)"
                          r"|\b(?:and|so)\s+(?=(?:i|i've|i'm|it|it's|my)\b)")
DETERMINER = re.compile(r"\b(?:the|your)\b")
# Degree words carry no symptom information. "not really" is an answer, and "so I ..." starts a clause.
FILLER = re.compile(r"\b(?:a little bit|a little|a bit of|a bit|kind of|sort of|kinda|(?<!not )really|very|quite"
                    r"|pretty|extremely|terribly|super|slightly|somewhat|also|still|always|constantly|sometimes"
                    r"|started|starting|began|keeps|kept|so(?!\s+(?:i|i've|i'm|it|it's|my)\b))\b\s*")

# Spoken variants of "<state> <part>" and "<part> pain" phrases.
STATES = set("sore swollen red itchy stiff weak blocked stuffy stuffed dry painful numb puffy tight runny bloated "
             "flaky irritated watery sticky crusty bloodshot scaly cracked rough raw congested clogged plugged blurry "
             "blurred cloudy loose hard black dark pale bloody smelly pink discoloured discolored hoarse raspy croaky "
             "noisy irregular poor low reduced muffled yellow thick brittle scratchy achy aching tender inflamed "
             "enlarged fast rapid racing pounding thinning".split())
COPULAS = ["is", "are", "feels", "feel", "looks", "look", "has been", "have been", "is feeling", "are feeling",
           "gets", "get", "got", "was", "were", "seems", "seem"]
PAIN_PARTS = set("ankle arm back ear elbow eye face foot toe hand finger head hip joint knee leg mouth muscle neck "
                 "shoulder side sinus stomach tummy belly testicle tooth wrist chest throat body flank bladder pelvis "
                 "forehead".split()) | {"big toe", "lower back", "lower stomach", "upper stomach", "lower belly",
                                        "upper belly", "lower tummy"}
NOT_A_PART = set("a an the my in on of at to from around behind when while during or with".split())
PLURALS = {"foot": "feet", "tooth": "teeth"}
# "my eyes are red and swollen" -> "my eyes are red and eyes are swollen", so both states match.
COORDINATED = re.compile(rf"\b([a-z']+) ({'|'.join(sorted(COPULAS, key=len, reverse=True))}) ([a-z]+),? and ([a-z]+)\b")


def _distribute(m) -> str:
    part, copula, a, b = m.groups()
    return f"{part} {copula} {a} and {part} {copula} {b}" if a in STATES and b in STATES else m.group()
# A leading yes/no only counts as a separate answer when punctuation follows it: "No, I've had a cough"
# (cough present) but "No fever" (fever denied).
LEADING_ANSWER = re.compile(r"^\s*(?:no|nope|nah|not really|yes|yeah|yep|yup)\b\s*[,.!-]+\s*")
NO_ANSWER = re.compile(r"^\s*(?:no|nope|nah|not really|not at all|i don't|i haven't|i didn't|i'm not|never|none|nothing)\b")
YES_ANSWER = re.compile(r"^\s*(?:yes|yeah|yep|yup|yea|i do|i have|i've|i did|i am|definitely|a bit|a little|"
                        r"sometimes|quite|correct|sure|actually,? yes)\b")


def _normalise(text: str) -> str:
    # Questions say "pain in your ear", the lexicon says "pain in the ear", patients say "pain in my ear".
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\byou\b", "i", DETERMINER.sub("my", text))


def _simplify(text: str) -> str:
    """The text symptoms are matched in: normalised, without degree words, with coordinated states
    distributed. Idempotent, so offsets into it agree wherever it is recomputed."""
    return COORDINATED.sub(_distribute, re.sub(r"  +", " ", FILLER.sub("", _normalise(text))))


def _key(phrase: str) -> str:
    return " ".join(re.split(r"[\s-]+", _simplify(phrase).strip()))


def _plural(part: str) -> str:
    *head, last = part.split()
    return " ".join([*head, PLURALS.get(last, last if last.endswith("s") else last + "s")])


def spoken_variants(phrase: str) -> list:
    """How patients say a listed phrase in a sentence: "swollen lip" -> "my lip is swollen", "lips feel swollen";
    "ear pain" -> "my ear hurts", "ears are hurting", "pain in my ears"."""
    words = phrase.lower().split()
    if words and words[0] in ("a", "an"):
        words = words[1:]
    out = []
    if 2 <= len(words) <= 4 and words[0] in STATES and words[-1] != "pain" and not NOT_A_PART & set(words[1:]):
        state = words[0]
        for part in dict.fromkeys([" ".join(words[1:]), _plural(" ".join(words[1:]))]):
            out += [f"{part} {c} {state}" for c in COPULAS] + [f"{part} {state}"]    # "is my lip swollen?"
    if words[-1:] == ["pain"] and " ".join(words[:-1]) in PAIN_PARTS:
        for part in dict.fromkeys([" ".join(words[:-1]), _plural(" ".join(words[:-1]))]):
            out += [f"{part} hurt", f"{part} is hurt", f"{part} are hurt", f"{part} ache", f"{part} aching",
                    f"pain in my {part}", f"pains in my {part}", f"{part} is killing me"]
            out += [f"{part} {c} {s}" for c in ("has been", "have been", "was", "were") for s in ("hurt", "aching")]
            out += [f"{part} {c} {s}" for c in COPULAS for s in ("sore", "painful")]
    if words[-1:] == ["swelling"] and len(words) <= 3:
        part = " ".join(words[:-1])
        out += [f"swelling {p} my {part}" for p in ("in", "of", "around", "on")] + [f"{part} has swollen", f"{part} swelled"]
    return out


def _trie_regex(keys) -> str:
    """One regex for all phrases, built as a character trie so most positions fail on the first character.
    At every node the longer continuation is tried before ending, so the longest phrase wins
    ("lower back pain" over "back pain", "throwing up blood" over "throwing up")."""
    trie = {}
    for k in keys:
        node = trie
        for ch in k:
            node = node.setdefault(ch, {})
        node[""] = True

    def build(node, last):
        end = r"(?:ing|es|ed|s)?" if last.isalpha() else ""
        alts = [(r"[\s-]+" if ch == " " else re.escape(ch)) + build(child, ch)
                for ch, child in sorted(node.items()) if ch]
        if not alts:
            return end
        body = alts[0] if len(alts) == 1 else "(?:" + "|".join(alts) + ")"
        return f"(?:{body}|{end})" if "" in node else body

    return r"(?<![a-z'])" + build(trie, "") + r"(?![a-z])"


class Lexicon:
    def __init__(self, phrases: dict, symptoms=None):
        self.phrases = {s: list(p) for s, p in phrases.items() if not s.startswith("_")}
        for s in symptoms or ():
            self.phrases.setdefault(s, [s.lower()])          # fallback: the Symcat name itself
        # Stop phrases match like symptoms but stand for none, so they block a shorter phrase inside them:
        # "anxious about the results" is the persona's temperament, not the symptom "anxious".
        self._lookup = dict.fromkeys(map(_key, phrases.get("_not_symptoms", [])), "")
        for s, ps in self.phrases.items():
            for p in [*ps, s]:
                self._lookup.setdefault(_key(p), s)           # first symptom to claim a phrase keeps it
        for s, ps in self.phrases.items():                    # generated variants never displace a listed phrase
            for p in ps:
                for v in spoken_variants(p):
                    self._lookup.setdefault(_key(v), s)
        self.covered_by = {}                                  # symptom -> card symptoms that make it on-card
        for group in phrases.get("_same_complaint", []):
            for s in group:
                self.covered_by.setdefault(s, set()).update(g for g in group if g != s)
        for s, covering in phrases.get("_covered_by", {}).items():
            if not s.startswith("_"):
                self.covered_by.setdefault(s, set()).update(covering)
        self._regex = re.compile(_trie_regex(self._lookup))
        self._find = lru_cache(maxsize=100_000)(self._find_uncached)

    def _symptom(self, matched: str):
        key = " ".join(re.split(r"[\s-]+", matched))
        if key in self._lookup:
            return self._lookup[key]
        for suffix in SUFFIXES:
            if key.endswith(suffix) and key[: -len(suffix)] in self._lookup:
                return self._lookup[key[: -len(suffix)]]
        return None

    def _find_uncached(self, low: str) -> tuple:
        return tuple((m.start(), m.end(), s) for m in self._regex.finditer(low) if (s := self._symptom(m.group())))

    @classmethod
    def load(cls, path=None, symptoms=None) -> "Lexicon":
        return cls(json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8")), symptoms)

    def lay(self, symptom: str) -> str:
        return self.phrases.get(symptom, [symptom.lower()])[0]

    def find(self, text: str) -> list:
        """Leftmost-longest symptom mentions as (start, end, symptom), offsets into the simplified text."""
        return list(self._find(_simplify(text)))

    def on_card(self, symptom: str, card_symptoms) -> bool:
        """True if the card has this symptom, or one that everyday speech does not tell apart from it."""
        return symptom in card_symptoms or not self.covered_by.get(symptom, set()).isdisjoint(card_symptoms)

    def mentions(self, text: str) -> list:
        """Mentions in a patient statement as (symptom, polarity), polarity True / False / None (unsure)."""
        low = _simplify(text)
        lead = LEADING_ANSWER.match(low)
        floor = lead.end() if lead else 0
        breaks = [m.end() for m in CLAUSE_BREAK.finditer(low)]
        out = []
        for start, _, symptom in self.find(low):
            clause_start = max([floor] + [b for b in breaks if b <= start])
            prefix = low[clause_start:start]
            if UNCERTAIN.search(prefix):
                out.append((symptom, None))
            else:
                out.append((symptom, not NEGATION.search(prefix)))
        return out


def answer_polarity(text: str):
    """True for a yes-type answer, False for a no-type answer, None otherwise."""
    low = _normalise(text)
    if NO_ANSWER.match(low):
        return False
    if YES_ANSWER.match(low):
        return True
    return None


def revealed_symptoms(turns: list, lexicon: Lexicon) -> dict:
    """{symptom: True (has it) / False (denied)} from a list of {"role", "text"} turns.

    A symptom the patient names keeps its latest stated polarity. A bare yes/no answer only fills in symptoms
    nothing has been said about yet: after "I've had a rash on my lip", "Any rash elsewhere?" -> "No, just
    the lip" does not take the rash back."""
    revealed, asked = {}, []
    for t in turns:
        if t["role"] == "doctor":
            asked = list(dict.fromkeys(s for _, _, s in lexicon.find(t["text"])))
            continue
        for symptom, polarity in lexicon.mentions(t["text"]):
            if polarity is not None:
                revealed[symptom] = polarity
        polarity = answer_polarity(t["text"])
        for symptom in asked:
            if symptom not in revealed and (polarity is False or (polarity is True and len(asked) == 1)):
                revealed[symptom] = polarity
        asked = []
    return revealed
