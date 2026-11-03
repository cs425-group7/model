"""Doctor agents. Each one implements `next_turn(history, force_diagnosis=False) -> DoctorTurn`, where
history is a list of {"role": "doctor"|"patient", "text"} turns. A turn either asks a question or closes the
consultation with a top-3 differential.

* NBDoctor   - asks the symptom with the highest expected information gain under Symcat naive Bayes.
               Also produces the stand-in training dialogues.
* GPT2Doctor - the fine-tuned GPT-2 doctor (or pretrained GPT-2 with a plain-text prompt as a baseline).
* LLMDoctor  - an instruction-tuned 7-8B model prompted zero-shot, few-shot or with chain of thought.
"""
import random
import re
from dataclasses import dataclass

from .format import DIAGNOSIS_SEP, DIAGNOSIS_TOKEN, ROLE_TOKENS, doctor_prompt
from .lexicon import revealed_symptoms
from .prompts import RETRY, chat_messages, parse_reply, system_prompt

OPENER = "What brings you in today?"
CLOSINGS = ["Thank you, I have a clearer picture now.", "Thanks, that's very helpful.",
            "Okay, thank you for answering my questions.", "Alright, I think I know what's going on."]


@dataclass
class DoctorTurn:
    text: str
    diagnosis: list | None = None       # set when the doctor closes the consultation
    reasoning: str | None = None        # chain-of-thought scratchpad, never shown to the patient
    raw: str | None = None


class NBDoctor:
    TEMPLATES = ["Have you had {lay}?", "Have you noticed {lay}?", "Any {bare}?", "What about {bare}?",
                 "And have you had {lay}?"]
    PAIR_TEMPLATES = ["Any {a} or {b}?", "Have you had {la} or {lb}?"]
    ONSET = ["When did this start?", "How long has this been going on?", "When did you first notice it?"]
    SEVERITY = ["How bad is it, on a scale of 1 to 10?", "How severe would you say it is?"]
    ELSE = ["Is there anything else you've noticed?", "Any other symptoms?"]

    def __init__(self, nb, lexicon, max_questions: int = 8, min_questions: int = 3, confidence: float = 0.7,
                 seed: int = 0):
        self.nb, self.lexicon = nb, lexicon
        self.max_questions, self.min_questions, self.confidence = max_questions, min_questions, confidence
        self.rng = random.Random(seed)

    def _bare(self, symptom):
        return re.sub(r"^(?:an?)\s+", "", self.lexicon.lay(symptom))

    def next_turn(self, history, force_diagnosis=False):
        if not history:
            return DoctorTurn(self.rng.choice([OPENER, OPENER, "Hello, what can I do for you today?"]))
        evidence = revealed_symptoms(history, self.lexicon)
        asked_text = " ".join(t["text"] for t in history if t["role"] == "doctor")
        n_questions = sum(t["role"] == "doctor" for t in history)
        ranked = self.nb.rank(evidence, k=3)
        confident = n_questions >= self.min_questions and ranked[0][1] >= self.confidence
        if force_diagnosis or n_questions >= self.max_questions or confident:
            if not force_diagnosis and n_questions < self.max_questions and \
                    not any(q in asked_text for q in self.ELSE) and self.rng.random() < 0.5:
                return DoctorTurn(self.rng.choice(self.ELSE))
            return DoctorTurn(self.rng.choice(CLOSINGS), diagnosis=[d for d, _ in ranked])
        if n_questions == 1 and "started" not in history[-1]["text"].lower() and self.rng.random() < 0.6:
            return DoctorTurn(self.rng.choice(self.ONSET))
        if n_questions >= 3 and not any(q in asked_text for q in self.SEVERITY) and self.rng.random() < 0.15:
            return DoctorTurn(self.rng.choice(self.SEVERITY))
        already = set(evidence) | {s for _, _, s in self.lexicon.find(asked_text)}
        gains = sorted(((g, s) for s, g in self.nb.information_gain(evidence).items() if s not in already),
                       reverse=True)
        if len(gains) >= 2 and self.rng.random() < 0.25:
            s1, s2 = gains[0][1], gains[1][1]
            return DoctorTurn(self.rng.choice(self.PAIR_TEMPLATES).format(
                a=self._bare(s1), b=self._bare(s2), la=self.lexicon.lay(s1), lb=self.lexicon.lay(s2)))
        s = gains[0][1]
        return DoctorTurn(self.rng.choice(self.TEMPLATES).format(lay=self.lexicon.lay(s), bare=self._bare(s)))


class GPT2Doctor:
    PLAIN_HEADER = "The following is a conversation in which a doctor interviews a patient to find out what illness they have.\n\n"

    def __init__(self, model, tokenizer, candidates: list, max_new_tokens: int = 60, do_sample: bool = True,
                 temperature: float = 0.7, top_p: float = 0.9, batch_size: int = 16):
        self.model, self.tokenizer, self.candidates = model.eval(), tokenizer, list(candidates)
        self.special = ROLE_TOKENS["doctor"] in tokenizer.get_vocab()
        self.gen = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "top_p": top_p if do_sample else None,
                    "temperature": temperature if do_sample else None, "pad_token_id": tokenizer.eos_token_id}
        self.batch_size = batch_size
        n_positions = getattr(model.config, "n_positions", None) or model.config.max_position_embeddings
        self.max_ctx = min(tokenizer.model_max_length, n_positions)
        if self.special:
            ids = tokenizer.convert_tokens_to_ids
            self.patient_id, self.diagnosis_id = ids(ROLE_TOKENS["patient"]), ids(DIAGNOSIS_TOKEN)
            self.stop_ids = [self.patient_id, self.diagnosis_id, tokenizer.eos_token_id]

    @classmethod
    def from_pretrained(cls, path: str, candidates: list, device: str | None = None, **kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        return cls(model, tokenizer, candidates, **kwargs)

    def _plain(self, history):
        names = {"doctor": "Doctor", "patient": "Patient"}
        return self.PLAIN_HEADER + "".join(f"{names[t['role']]}: {t['text']}\n" for t in history)

    def _encode_tail(self, text: str, room: int) -> list:
        ids = self.tokenizer(text)["input_ids"]
        return ids[-(self.max_ctx - room):]             # keep the most recent context

    def next_turn(self, history, force_diagnosis=False):
        import torch

        if force_diagnosis:
            return DoctorTurn(CLOSINGS[0], diagnosis=self.rank(history))
        prompt = doctor_prompt(history) if self.special else self._plain(history) + "Doctor:"
        ids = self._encode_tail(prompt, self.gen["max_new_tokens"])
        input_ids = torch.tensor([ids], device=self.model.device)
        stop = self.stop_ids if self.special else self.tokenizer.convert_tokens_to_ids(["Ċ", "ĊĊ"])  # "\n", "\n\n"
        with torch.no_grad():
            out = self.model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                      eos_token_id=stop, **self.gen)[0, len(ids):].tolist()
        if not self.special:
            text = self.tokenizer.decode(out).split("\n")[0].strip()
            return DoctorTurn(text or "Can you tell me more?")
        ended_by = out[-1] if out and out[-1] in self.stop_ids else None
        text = self.tokenizer.decode([t for t in out if t not in self.stop_ids], skip_special_tokens=True).strip()
        if ended_by in (self.diagnosis_id, self.tokenizer.eos_token_id):
            return DoctorTurn(text or CLOSINGS[0], diagnosis=self.rank(history + [{"role": "doctor", "text": text}]))
        return DoctorTurn(text or "Can you tell me more?")

    def rank(self, history, k: int = 3) -> list:
        """Top-k candidates by log P(name | dialogue); the decision of *when* to diagnose is generated."""
        import torch

        if self.special:
            if history and history[-1]["role"] == "doctor":
                prefix = doctor_prompt(history[:-1]) + history[-1]["text"] + DIAGNOSIS_TOKEN
            else:
                prefix = doctor_prompt(history) + CLOSINGS[0] + DIAGNOSIS_TOKEN
            conts = [c + DIAGNOSIS_SEP.strip() for c in self.candidates]          # "Flu;"
        else:
            prefix = self._plain(history) + "Doctor: Based on what you have told me, the most likely diagnosis is"
            conts = [f" {c}." for c in self.candidates]
        cont_ids = [self.tokenizer(c)["input_ids"] for c in conts]
        prefix_ids = self._encode_tail(prefix, max(map(len, cont_ids)))
        scores = []
        for i in range(0, len(cont_ids), self.batch_size):
            chunk = cont_ids[i:i + self.batch_size]
            seqs = [prefix_ids + c for c in chunk]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), self.tokenizer.eos_token_id, device=self.model.device)
            mask = torch.zeros_like(ids)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s)
                mask[j, :len(s)] = 1
            # Logits only from the last prefix token on: over the whole prompt they take gigabytes.
            keep = width - len(prefix_ids) + 1
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
                logp = torch.log_softmax(logits.float(), -1)
            for j, c in enumerate(chunk):
                pos = torch.arange(len(c), device=ids.device)          # logp[:, 0] predicts the name's first token
                scores.append(logp[j, pos, torch.tensor(c, device=ids.device)].sum().item())
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.candidates[i] for i in order[:k]]


class LLMDoctor:
    def __init__(self, chat, candidates: list, matcher, strategy: str = "zero_shot", examples: list | None = None,
                 max_questions: int = 10, max_tokens: int | None = None):
        self.chat, self.matcher, self.strategy = chat, matcher, strategy
        self.system = system_prompt(candidates, strategy, max_questions, examples)
        # A question or a DIAGNOSIS line is short; only chain of thought needs room to reason.
        self.max_tokens = max_tokens or (600 if strategy == "cot" else 150)

    def next_turn(self, history, force_diagnosis=False):
        if not history:
            return DoctorTurn(OPENER)
        messages = chat_messages(history, self.system, force_diagnosis, cot=self.strategy == "cot")
        raw = self.chat(messages, max_tokens=self.max_tokens)
        parsed = parse_reply(raw, self.matcher, want_diagnosis=force_diagnosis)
        if parsed.diagnosis == [] or (force_diagnosis and parsed.diagnosis is None):
            retry = messages + [{"role": "assistant", "content": raw}, {"role": "user", "content": RETRY}]
            raw += "\n[retry]\n" + self.chat(retry, max_tokens=150)
            reasoning = parsed.reasoning
            parsed = parse_reply(raw.split("\n[retry]\n")[-1], self.matcher, want_diagnosis=True)
            parsed.reasoning = parsed.reasoning or reasoning
        if parsed.diagnosis is not None:
            return DoctorTurn(CLOSINGS[0], diagnosis=parsed.diagnosis, reasoning=parsed.reasoning, raw=raw)
        return DoctorTurn(parsed.question, raw=raw)
