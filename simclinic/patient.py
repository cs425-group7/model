"""Patient-model format and inference for the SimClinic app.

Notebook 08 trains on exactly the strings this module builds (its last section checks this), and the
app loads the trained model through PatientSimulator:

    <|disease|>Flu<|symptoms|>a fever; a cough<|persona|>34, F, Chinese, teacher, anxious   <- hidden header, masked
    <|doctor|>What brings you in today?<|patient|>I've had a fever since Monday...<|doctor|>
                                                  └──────── loss computed here ──────────┘

Card symptoms are written in lay words, the first phrasing in symptom_lexicon.json ("Coryza" -> "a runny nose").
Every patient turn is trained together with the <|doctor|> marker after it, so the model learns
to end its turn. At inference, generation stops on that marker.
"""
import json
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROLE_TOKENS = {"doctor": "<|doctor|>", "patient": "<|patient|>"}
HEADER_TOKENS = {"disease": "<|disease|>", "symptoms": "<|symptoms|>", "persona": "<|persona|>"}
DIAGNOSIS_TOKEN = "<|diagnosis|>"
# Same order as the notebooks' tokenizer setup, so the token ids match.
SPECIAL_TOKENS = list(ROLE_TOKENS.values()) + list(HEADER_TOKENS.values()) + [DIAGNOSIS_TOKEN]
STOP_TOKEN = ROLE_TOKENS["doctor"]
DOCTOR_OPENER = "What brings you in today?"   # notebook 02's opener (notebook 01 used it lowercased)
PERSONA_FIELDS = ["age", "sex", "ethnicity", "occupation", "temperament", "health_literacy", "onset", "severity"]
LEXICON_JSON = Path(__file__).with_name("symptom_lexicon.json")


@lru_cache(maxsize=1)
def _lay_phrases() -> dict:
    data = json.loads(LEXICON_JSON.read_text(encoding="utf-8"))
    return {s: p[0] for s, p in data.items() if not s.startswith("_") and p}


def lay(symptom: str) -> str:
    """Everyday wording of a card symptom, as Lexicon.lay gives it ("Coryza" -> "a runny nose")."""
    return _lay_phrases().get(symptom, symptom.lower())


def format_header(card=None) -> str:
    """Hidden case-card header; "" when there is no card (the diagnose-me data)."""
    if not card:
        return ""
    persona = card.get("persona") or {}
    persona_text = ", ".join(str(persona[f]) for f in PERSONA_FIELDS if persona.get(f) not in (None, ""))
    return (HEADER_TOKENS["disease"] + card["disease"]
            + HEADER_TOKENS["symptoms"] + "; ".join(lay(s) for s in card["symptoms"])
            + HEADER_TOKENS["persona"] + persona_text)


def _with_opener(turns, opener):
    turns = list(turns)
    if opener and (not turns or turns[0]["role"] != "doctor"):
        turns = [{"role": "doctor", "text": opener}] + turns
    return turns


def _render(turns) -> str:
    return "".join(ROLE_TOKENS[t["role"]] + t["text"] for t in turns)


def format_dialogue(turns, card=None, opener=DOCTOR_OPENER):
    """Training string and the [start, end) character spans to compute loss on.

    Turns after the last patient turn are dropped (they would be masked anyway), and the string
    always ends with the <|doctor|> marker that closes the last patient turn.
    """
    turns = _with_opener(turns, opener)
    roles = [t["role"] for t in turns]
    if any(a == b for a, b in zip(roles, roles[1:])):
        raise ValueError(f"roles must alternate: {roles}")
    turns = turns[: max(i for i, r in enumerate(roles) if r == "patient") + 1]

    text, spans = format_header(card), []
    for t in turns:
        text += ROLE_TOKENS[t["role"]]
        start = len(text)
        text += t["text"]
        if t["role"] == "patient":
            spans.append((start, len(text) + len(STOP_TOKEN)))   # include the <|doctor|> that follows
    return text + STOP_TOKEN, spans


def mask_labels(input_ids, offsets, spans):
    """labels = input_ids, with -100 on every token that does not start inside a span."""
    return [tok if (e > s and any(a <= s < b for a, b in spans)) else -100
            for tok, (s, e) in zip(input_ids, offsets)]


def build_prompt(card, history, tokenizer, max_len, opener=DOCTOR_OPENER) -> str:
    """Inference prompt: the header plus the most recent turns that fit in max_len tokens, ending
    in <|patient|> so the model writes the patient's next turn. The header is never dropped."""
    turns = _with_opener(history, opener)
    if turns[-1]["role"] != "doctor":
        raise ValueError("history must end with a doctor turn")
    header = format_header(card)

    def render(ts):
        return header + _render(ts) + ROLE_TOKENS["patient"]

    while len(turns) > 1 and len(tokenizer(render(turns))["input_ids"]) > max_len:
        turns = turns[1:]
        while turns[0]["role"] != "doctor":       # windows start with a doctor turn, as in training
            turns = turns[1:]
    return render(turns)


class PatientSimulator:
    """Plays the patient for a case card with a fine-tuned checkpoint.

        sim = PatientSimulator("models/patient/stage2-gpt2-lr5e-05")   # a run folder from notebook 08
        sim.reply(card, [{"role": "doctor", "text": "What brings you in today?"}])
    """

    def __init__(self, model_dir, tokenizer_dir=None, device=None, max_len=1024):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir or model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(self.device).eval()
        if self.model.get_input_embeddings().num_embeddings < len(self.tokenizer):
            self.model.resize_token_embeddings(len(self.tokenizer))   # only for a base model, never fine-tuned
        cfg = self.model.config
        self.max_len = min(max_len, getattr(cfg, "n_positions", None) or cfg.max_position_embeddings)
        self.stop_ids = [self.tokenizer.convert_tokens_to_ids(STOP_TOKEN), self.tokenizer.eos_token_id]

    @torch.no_grad()
    def reply(self, card, history, temperature=0.8, top_p=0.9, max_new_tokens=120) -> str:
        prompt = build_prompt(card, history, self.tokenizer, self.max_len - max_new_tokens)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        if enc["input_ids"].shape[1] > self.max_len - max_new_tokens:
            raise ValueError("the header plus the last doctor turn do not fit in the context")
        out = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                                  max_new_tokens=max_new_tokens, eos_token_id=self.stop_ids,
                                  pad_token_id=self.tokenizer.pad_token_id)
        text = self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:])
        for tok in SPECIAL_TOKENS + [self.tokenizer.eos_token]:   # keep only the patient's own words
            text = text.split(tok)[0]
        return text.strip()
