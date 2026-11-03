"""Training-string format for the doctor role, shared by data preparation and inference.

    <|doctor|>What brings you in today?<|patient|>I've had a fever ...<|doctor|>Any cough?<|patient|>No.
    <|doctor|>Thanks, I have a clearer picture now.<|diagnosis|>Flu; Common cold; Strep throat<|endoftext|>

Loss is computed on every doctor turn plus the marker that ends it (<|patient|> hands the turn over,
<|diagnosis|> closes the consultation), and on the diagnosis line. Patient text and the <|doctor|> markers
are context. Doctor examples leave out the hidden case header, which only the patient model sees.
"""
ROLE_TOKENS = {"doctor": "<|doctor|>", "patient": "<|patient|>"}
EXTRA_SPECIAL_TOKENS = ["<|disease|>", "<|symptoms|>", "<|persona|>", "<|diagnosis|>"]   # same order as notebook 01
DIAGNOSIS_TOKEN = "<|diagnosis|>"
DIAGNOSIS_SEP = "; "


def render(turns: list) -> str:
    return "".join(ROLE_TOKENS[t["role"]] + t["text"] for t in turns)


def doctor_prompt(history: list) -> str:
    """What the doctor model is given at inference: the dialogue so far plus its own role marker."""
    return render(history) + ROLE_TOKENS["doctor"]


def format_doctor_example(turns: list, diagnosis: list, eos_token: str):
    """Return (text, spans), where spans are [start, end) character ranges to compute loss on.

    `turns` alternate doctor/patient, start with the doctor and end with the doctor's closing remark.
    """
    assert turns[0]["role"] == "doctor" and turns[-1]["role"] == "doctor", "doctor opens and closes"
    text, spans = "", []
    for t in turns:
        marker = ROLE_TOKENS[t["role"]]
        if t["role"] == "doctor":
            text += marker
            spans.append([len(text), len(text) + len(t["text"])])
            text += t["text"]
        else:
            spans[-1][1] = len(text) + len(marker)       # <|patient|> ends the doctor's turn
            text += marker + t["text"]
    text += DIAGNOSIS_TOKEN + DIAGNOSIS_SEP.join(diagnosis) + eos_token
    spans[-1][1] = len(text)                            # closing remark through the diagnosis and eos
    return text, spans
