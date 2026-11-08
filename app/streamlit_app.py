"""SimClinic app. Run from the repo root:  streamlit run app/streamlit_app.py"""
import random
import sys
import time
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make the repo's simclinic importable

import streamlit as st

from app import engine

ASSETS = Path(__file__).with_name("assets")
st.set_page_config(page_title="SimClinic", page_icon=str(ASSETS / "favicon.png"), layout="wide")
ss = st.session_state

MODES = {
    "patient": ("Interview a patient", "You're the doctor. Take a history, then diagnose."),
    "doctor": ("Be the patient", "The AI doctor interviews you and gives a diagnosis."),
    "auto": ("Watch the AI doctor", "The AI doctor interviews the AI patient."),
}
SEX = {"M": "Male", "F": "Female"}
PACE = {"Fast": 0.0, "Normal": 0.6, "Slow": 1.4}
CHAT_BOX = {"key": "chatbox", "border": False}   # sized by the .st-key-chatbox CSS: fills the window, scrolls inside
LOGO = (ASSETS / "logo.svg").read_text(encoding="utf-8")

# Every colour is a variable, so the dark theme only redefines the variables (plus Streamlit's own widgets).
LIGHT = """
:root {
  --paper: #F4F6F7; --surface: #FFFFFF; --surface-2: #FBFCFD; --ink: #1B2A3A; --ink-soft: #5B6875;
  --line: #D5DCE1; --chart: #2B4C7E; --chart-ink: #2B4C7E; --chart-tint: #E6ECF4; --mark: #F7DE6B;
  --mark-ink: #1B2A3A; --ok: #2E7D5B; --miss: #B3412F; --near: #B7791F; --near-ink: #8A5A12;
  --warn-bg: #FFF6E5; --warn-line: #E8C77A; --btn: #2B4C7E; --btn-hover: #213D66; --bubble-me: #FFFFFF;
}"""
DARK = """
:root {
  --paper: #141B24; --surface: #1C2531; --surface-2: #212C3A; --ink: #E6EBF0; --ink-soft: #9AA7B4;
  --line: #33414F; --chart: #7FA3D9; --chart-ink: #B9CDEB; --chart-tint: #22324A; --mark: #E3C64F;
  --mark-ink: #1B2A3A; --ok: #5CC197; --miss: #EC8A72; --near: #D9A441; --near-ink: #E8BE6A;
  --warn-bg: #33291A; --warn-line: #7A5F2A; --btn: #3D69A8; --btn-hover: #4A78BA; --bubble-me: #1C2531;
}
/* Streamlit's own widgets take their colours from the fixed config theme, so they are restyled here. */
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stBottom"] > div,
[data-testid="stHeader"] { background: var(--paper) !important; }
[data-testid="stSidebar"], [data-testid="stSidebarContent"] { background: var(--surface) !important; }
[data-testid="stMain"] p, [data-testid="stMain"] li, [data-testid="stSidebar"] p, [data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] > h1, [data-testid="stMarkdownContainer"] > h2, [data-testid="stMarkdownContainer"] > h3,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stSpinner"], [data-testid="stSpinner"] p, [data-testid="stHeader"] button { color: var(--ink) !important; }
[data-testid="stCaptionContainer"] p, [data-testid="stRadioCaption"] p { color: var(--ink-soft) !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] { color: var(--ink) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="base-input"],
[data-testid="stChatInput"], [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea,
[data-baseweb="select"] input, [data-testid="stTextInput"] input {
  background: var(--surface-2) !important; color: var(--ink) !important; border-color: var(--line) !important; }
[data-baseweb="select"] svg, [data-testid="stChatInput"] svg { fill: var(--ink-soft) !important; color: var(--ink-soft) !important; }
input::placeholder, textarea::placeholder { color: var(--ink-soft) !important; opacity: 1; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
  background: var(--surface) !important; }
[data-baseweb="popover"] li { color: var(--ink) !important; background: var(--surface) !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] li[aria-selected="true"] { background: var(--chart-tint) !important; }
[data-baseweb="tag"] { background: var(--chart-tint) !important; color: var(--chart-ink) !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] svg { color: var(--chart-ink) !important; fill: var(--chart-ink) !important; }
[data-testid="stBaseButton-secondary"] { background: var(--surface) !important; color: var(--ink) !important;
  border-color: var(--line) !important; }
[data-testid="stBaseButton-secondary"]:disabled { color: var(--ink-soft) !important; opacity: .6; }
[data-testid="stExpander"] details { background: var(--surface) !important; border-color: var(--line) !important; }
[data-testid="stExpander"] summary { background: var(--surface) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--ink) !important; }
[data-testid="stExpander"] summary:hover { background: var(--surface-2) !important; }
[data-testid="stDialog"] { background: rgba(5, 10, 16, 0.6) !important; }
[data-testid="stDialog"] > div, [data-testid="stDialog"] section[role="dialog"] {
  background: var(--surface) !important; color: var(--ink) !important; border: 1px solid var(--line); }
[data-testid="stDialog"] h2, [data-testid="stDialog"] button[aria-label="Close"],
[data-testid="stDialog"] button[aria-label="Close"] svg { color: var(--ink) !important; fill: var(--ink) !important; }
[data-testid="stRadio"] label p { color: var(--ink) !important; }
/* Streamlit 1.5x+ (react-aria) controls and their dropdown panels */
[data-testid="stSelectbox"] div[role="group"], [data-testid="stMultiSelect"] div[role="group"],
[data-testid="stTextInput"] div:has(> input) {
  background: var(--surface-2) !important; border-color: var(--line) !important; color: var(--ink) !important; }
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input, [data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [role="group"] span, [data-testid="stSelectbox"] [role="group"] div {
  color: var(--ink) !important; background: transparent !important; }
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg { color: var(--ink-soft) !important; fill: var(--ink-soft) !important; }
[data-testid="stMultiSelectDropdown"], [data-testid="stSelectboxDropdown"], [role="listbox"] {
  background: var(--surface) !important; border: 1px solid var(--line) !important; }
[role="option"] { color: var(--ink) !important; background: transparent !important; }
[role="option"]:hover, [role="option"][data-focused="true"], [role="option"][data-selected="true"] {
  background: var(--chart-tint) !important; }
[data-testid="stMultiSelect"] [role="group"] [role="row"], [data-testid="stMultiSelect"] [role="group"] [data-rac] span {
  color: var(--chart-ink) !important; }
[data-testid="stRadioOption"] > div > div:first-child { background: var(--surface-2) !important; border-color: var(--ink-soft) !important; }
"""

BASE = """
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
/* Text only: icon spans keep their own (Material Symbols) font. */
html, body, p, li, label, h1, h2, h3, input, textarea, button, .turn, .chart, .verdict {
  font-family: 'Atkinson Hyperlegible', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stAppViewContainer"] { background: var(--paper); transition: background .25s; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1240px; }
[data-testid="stMarkdownContainer"] h2 { color: var(--ink); letter-spacing: -0.01em; }
[data-testid="stBaseButton-primary"] { background: var(--btn) !important; border-color: var(--btn) !important; color: #FFFFFF !important; }
[data-testid="stBaseButton-primary"]:hover { background: var(--btn-hover) !important; border-color: var(--btn-hover) !important; }
[data-testid="stBaseButton-primary"] p { color: #FFFFFF !important; }
[data-testid="stBaseButton-primary"]:disabled { opacity: .4; cursor: not-allowed; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
.brand { display: flex; gap: 0.65rem; align-items: center; margin: 0 0 1.4rem; }
.brand svg { width: 44px; height: 44px; flex: none; }
.brand b { display: block; font-size: 1.5rem; color: var(--ink); line-height: 1.1; }
.brand span { display: block; font-size: 0.9rem; color: var(--ink-soft); margin-top: 0.15rem; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] { gap: 0.55rem; }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {
  border: 1px solid var(--line); border-radius: 10px; padding: 0.65rem 0.8rem 0.7rem; width: 100%;
  background: var(--surface); transition: border-color .15s, background .15s; cursor: pointer;
}
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:hover { border-color: var(--chart); }
[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div[data-selected="true"] {
  border-color: var(--chart); background: var(--chart-tint); box-shadow: inset 4px 0 0 var(--chart);
}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p { font-weight: 700; color: var(--ink); }
[data-testid="stSidebar"] [data-testid="stRadioCaption"] p { color: var(--ink-soft); }

/* Page heading */
.lede { color: var(--ink-soft); font-size: 1.05rem; margin: -0.4rem 0 1.4rem; max-width: 62ch; }
.counter { display: inline-block; font-size: 0.9rem; color: var(--chart-ink); background: var(--chart-tint);
  border-radius: 999px; padding: 0.2rem 0.75rem; margin-bottom: 0.8rem; }
.notice { font-size: 0.9rem; color: var(--ink-soft); background: var(--surface); border: 1px solid var(--line);
  border-radius: 10px; padding: 0.55rem 0.8rem; margin-bottom: 1rem; }

/* One-screen layout: the page never scrolls. The chat box takes the rest of the window height and scrolls
   inside; the right-hand column scrolls on its own only if it is taller than the window. Tune the rem
   offsets (heading + counter + chat input) if the box is too tall or too short. */
[data-testid="stAppViewContainer"], [data-testid="stMain"] { overflow: hidden !important; }
.st-key-chatbox { height: calc(100vh - 19rem) !important; min-height: 12rem; overflow-y: auto !important;
  overflow-x: hidden; padding-right: 0.4rem; flex: none !important; }
[data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2) {
  max-height: calc(100vh - 10rem); overflow-y: auto; padding-right: 0.4rem; }

/* Transcript */
.turn { display: flex; flex-direction: column; max-width: 82%; margin: 0 0 0.85rem; }
.turn .who { font-size: 0.8rem; color: var(--ink-soft); margin: 0 0.2rem 0.2rem; }
.turn .say { padding: 0.65rem 0.95rem; border-radius: 14px; line-height: 1.5; color: var(--ink); font-size: 1.02rem; }
.turn-doctor { align-items: flex-start; }
.turn-doctor .say { background: var(--chart-tint); border-top-left-radius: 4px; }
.turn-patient { align-items: flex-end; margin-left: auto; }
.turn-patient .say { background: var(--bubble-me); border: 1px solid var(--line); border-top-right-radius: 4px; }
.turn.you .who { color: var(--chart-ink); font-weight: 700; }
.turn.refused .say { background: var(--warn-bg); border: 1px solid var(--warn-line); }
.turn.refused .who { color: var(--near-ink); }
.empty { border: 1px dashed var(--line); border-radius: 12px; padding: 1.2rem 1.3rem; color: var(--ink-soft); background: var(--surface); }
.empty b { color: var(--ink); }

/* Typing indicator: shown while the model writes its reply */
.say.typing { display: inline-flex; gap: 5px; align-items: center; padding: 0.85rem 1rem; }
.say.typing i { width: 7px; height: 7px; border-radius: 50%; background: var(--ink-soft); display: block;
  animation: blink 1.2s infinite ease-in-out; }
.say.typing i:nth-child(2) { animation-delay: .2s; } .say.typing i:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 80%, 100% { opacity: .25; transform: translateY(0); } 40% { opacity: 1; transform: translateY(-3px); } }

/* The chart: wristband + case notes */
.chart { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin-bottom: 1rem; }
.band { display: grid; grid-template-columns: repeat(4, auto); gap: 0.2rem 1.2rem; padding: 0.9rem 1.1rem 0.9rem 1.4rem;
  border-bottom: 1px solid var(--line); position: relative; background: var(--surface-2); }
.band::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 6px; background: var(--chart); }
.band .f small { display: block; font-size: 0.75rem; color: var(--ink-soft); }
.band .f b { color: var(--ink); font-size: 1rem; }
.notes { padding: 0.9rem 1.1rem 1rem 1.4rem; }
.row { margin: 0 0 0.75rem; }
.row small { display: block; font-size: 0.78rem; color: var(--ink-soft); margin-bottom: 0.15rem; }
.row .v { color: var(--ink); font-size: 1.02rem; }
.sealed { color: var(--ink-soft); letter-spacing: 0.02em; }
.sealed::before { content: "🔒 "; }
.dx-name { font-size: 1.25rem; font-weight: 700; color: var(--ink); }

/* Highlighter symptoms */
.sx { display: inline-block; margin: 0 0.45rem 0.4rem 0; padding: 0.05rem 0.35rem; font-size: 1rem; color: var(--ink); }
.sx.found { color: var(--mark-ink); background-image: linear-gradient(var(--mark), var(--mark));
  background-repeat: no-repeat; background-position: 0 50%; background-size: 100% 78%; }
.sx.missed { color: var(--ink-soft); text-decoration: underline dotted var(--ink-soft); text-underline-offset: 4px; }
.sx.extra { color: var(--miss); text-decoration: line-through; }
.legend { font-size: 0.82rem; color: var(--ink-soft); margin-top: 0.2rem; }

/* The reveal: the one orchestrated motion. The sealed diagnosis resolves, then the highlighter sweeps. */
.anim .dx-name { animation: unseal .7s cubic-bezier(.2,.7,.2,1) both; }
.anim .sx.found { animation: sweep .45s ease-out both; }
.anim.verdict { animation: rise .35s ease-out both; }
@keyframes unseal { from { opacity: 0; filter: blur(8px); letter-spacing: .12em; } to { opacity: 1; filter: none; letter-spacing: normal; } }
@keyframes sweep { from { background-size: 0% 78%; color: var(--ink); } to { background-size: 100% 78%; color: var(--mark-ink); } }
@keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

/* Verdict */
.verdict { border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; border: 1px solid var(--line); background: var(--surface); }
.verdict h3 { margin: 0 0 0.25rem; font-size: 1.3rem; }
.verdict .vbody, .verdict .vbody p { margin: 0; color: var(--ink-soft); }
.verdict .vbody b { color: var(--ink); }
.verdict.ok { border-left: 6px solid var(--ok); } .verdict.ok h3 { color: var(--ok); }
.verdict.near { border-left: 6px solid var(--near); } .verdict.near h3 { color: var(--near-ink); }
.verdict.miss { border-left: 6px solid var(--miss); } .verdict.miss h3 { color: var(--miss); }
.ddx { list-style: none; padding: 0; margin: 0.3rem 0 0; counter-reset: r; }
.ddx li { counter-increment: r; padding: 0.4rem 0; border-bottom: 1px solid var(--line); color: var(--ink); }
.ddx li:last-child { border-bottom: none; }
.ddx li::before { content: counter(r); display: inline-block; width: 1.6rem; color: var(--ink-soft); }
.ddx li.hit { font-weight: 700; }
.ddx li.hit::after { content: "\u00a0\u00a0✓\u00a0correct"; color: var(--ok); font-weight: 400; font-size: 0.9rem; white-space: nowrap; }
.stats { display: flex; gap: 1.6rem; margin: 0.4rem 0 0.2rem; }
.stats div b { display: block; font-size: 1.35rem; color: var(--ink); }
.stats div small { color: var(--ink-soft); font-size: 0.8rem; }

/* Grounded explanations */
.why { margin: 0 0 0.9rem; }
.why h4 { margin: 0 0 0.25rem; font-size: 1rem; color: var(--ink); }
.why ul { list-style: none; padding: 0; margin: 0; }
.why li { padding: 0.15rem 0 0.15rem 1.3rem; position: relative; color: var(--ink); font-size: 0.95rem; line-height: 1.45; }
.why li::before { position: absolute; left: 0; font-weight: 700; }
.why li.for::before { content: "+"; color: var(--ok); }
.why li.against::before { content: "−"; color: var(--miss); }
.why li.odd::before { content: "?"; color: var(--ink-soft); }
.why li q { color: var(--ink-soft); }
.why .none { color: var(--ink-soft); font-size: 0.92rem; }

/* Welcome dialog */
.welcome p { color: var(--ink); line-height: 1.55; margin: 0 0 0.7rem; }
.welcome dl { margin: 0 0 0.9rem; }
.welcome dt { font-weight: 700; color: var(--ink); }
.welcome dd { margin: 0 0 0.55rem; color: var(--ink-soft); }
.welcome small { color: var(--ink-soft); }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""


@st.cache_resource(show_spinner="Loading cases and models…")
def resources():
    return engine.load_resources()


R = resources()

if "dark" not in ss:   # first visit: ?theme=dark|light in the URL, else follow the browser's theme
    requested = st.query_params.get("theme")
    if requested in ("dark", "light"):
        ss.dark = requested == "dark"
    else:
        try:
            ss.dark = st.context.theme.type == "dark"
        except Exception:
            ss.dark = False
st.markdown("<style>" + LIGHT + BASE + (DARK if ss.dark else "") + "</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------- state
def finish_session():
    """Log the session that is ending (if it finished and logging is on), with the optional rating."""
    if not engine.LOG_DIR or not ss.get("done") or ss.get("logged") or "card" not in ss:
        return
    mode = ss.get("session_mode")
    if mode not in ("patient", "doctor"):
        return
    rec = {"mode": mode, "player": ss.get("player") or "anonymous", "seed": ss.get("seed"), "card": ss.card,
           "turns": ss.turns, "diagnosis": ss.guesses if mode == "patient" else ss.result,
           "minutes": round((time.time() - ss.get("started", time.time())) / 60, 1)}
    dx = rec["diagnosis"] or []
    rec |= {"top1": dx[:1] == [ss.card["disease"]], "top3": ss.card["disease"] in dx[:3]}
    if mode == "patient" and ss.get("rating"):
        rec["in_character"] = ss.rating
    if mode == "doctor":
        rec["doctor"] = ss.get("doctor_name", R.doctor_names[0])
    else:
        rec["patient"] = ss.get("patient_name", R.patient_names[0])
    engine.log_session(rec)
    ss.logged = True


def new_case():
    finish_session()
    seed = random.randrange(10**6)
    ss.card = random.Random(seed).choice(R.case_pool)
    ss.turns, ss.done, ss.guesses, ss.result, ss.scores, ss.logged = [], False, [], None, None, False
    ss.pending, ss.anim = False, False
    ss.patient = engine.make_patient(R, ss.card, seed, ss.get("patient_name", R.patient_names[0]))
    mode = ss.get("mode", "patient")    # get: callbacks can run before the script initialises state
    ss.doctor = engine.make_doctor(R, ss.get("doctor_name", R.doctor_names[0]), seed, guarded=mode == "doctor")
    ss.seed, ss.started, ss.session_mode, ss.rating = seed, time.time(), mode, None
    if mode == "doctor":  # the doctor opens
        ss.turns.append({"role": "doctor", "text": ss.doctor.next_turn([]).text})


def finish(result=None):
    """Close the consultation and play the reveal once."""
    ss.done, ss.anim = True, True
    if result is not None:
        ss.result = result


if "mode" not in ss:
    ss.mode = "patient"
    new_case()


# --------------------------------------------------------------- display
def turn_html(t, you=None):
    who = "You" if t["role"] == you else t["role"].title()
    if t.get("refused"):
        who += " (declined)"
    cls = f"turn turn-{t['role']}" + (" you" if t["role"] == you else "") + (" refused" if t.get("refused") else "")
    return f'<div class="{cls}"><div class="who">{who}</div><div class="say">{escape(t["text"])}</div></div>'


def typing_html(role, you=None):
    who = "You" if role == you else role.title()
    return (f'<div class="turn turn-{role}" role="status" aria-live="polite"><div class="who">{who} is typing…</div>'
            f'<div class="say typing" aria-label="{who} is typing"><i></i><i></i><i></i></div></div>')


def show_turns(turns, you=None):
    if turns:
        st.markdown("".join(turn_html(t, you) for t in turns), unsafe_allow_html=True)


def symptom_marks(turns, card, animate=False):
    fb = engine.symptom_feedback(R, turns, card)
    found = set(fb["found"])
    chips, k = [], 0
    for s in card["symptoms"]:
        delay = ""
        if s in found:
            delay = f' style="animation-delay:{0.45 + 0.12 * k:.2f}s"' if animate else ""
            k += 1
        chips.append(f'<span class="sx {"found" if s in found else "missed"}"{delay}>{escape(R.lexicon.lay(s))}</span>')
    chips += [f'<span class="sx extra" title="Mentioned but not on the card">{escape(R.lexicon.lay(s))}</span>'
              for s in fb["off_card"]]
    legend = "Highlighted: uncovered. Dotted: never came up."
    if fb["off_card"]:
        legend += " Struck through: mentioned, but not on the card."
    if leaks := engine.disease_leaked(R, turns, card):
        legend += f" The patient named the condition ({escape(', '.join(leaks))})."
    return "".join(chips), legend, len(found)


def chart(card, reveal=False, turns=None, title=None, animate=False):
    p = card["persona"]
    band = "".join(f'<div class="f"><small>{k}</small><b>{escape(str(v))}</b></div>' for k, v in
                   [("Age", p["age"]), ("Sex", SEX.get(p["sex"], p["sex"])),
                    ("Ethnicity", p["ethnicity"]), ("Work", p["occupation"].capitalize())])
    rows = []
    if title:
        rows.append(f'<div class="row"><small>{escape(title)}</small></div>')
    if reveal:
        rows.append(f'<div class="row"><small>Diagnosis</small><div class="dx-name">{escape(card["disease"])}</div></div>')
        if turns is not None:
            chips, legend, _ = symptom_marks(turns, card, animate)
            rows.append(f'<div class="row"><small>Symptoms on the card</small><div>{chips}</div>'
                        f'<div class="legend">{legend}</div></div>')
        else:
            chips = "".join(f'<span class="sx">{escape(R.lexicon.lay(s))}</span>' for s in card["symptoms"])
            rows.append(f'<div class="row"><small>Symptoms on the card</small><div>{chips}</div></div>')
        rows.append(f'<div class="row"><small>Presentation</small><div class="v">{escape(p["temperament"].capitalize())}, '
                    f'started {escape(p["onset"])}, {escape(p["severity"])}.</div></div>')
    else:
        rows.append('<div class="row"><small>Diagnosis</small><div class="v sealed">Sealed until you reveal it</div></div>')
    st.markdown(f'<div class="chart{" anim" if animate else ""}"><div class="band">{band}</div>'
                f'<div class="notes">{"".join(rows)}</div></div>', unsafe_allow_html=True)


def differential(ranking, truth):
    items = "".join(f'<li class="{"hit" if d == truth else ""}">{escape(d)}</li>' for d in ranking[:3])
    return f'<ol class="ddx">{items or "<li>No valid diagnosis</li>"}</ol>'


def why(conditions, turns):
    """The team's grounded explanations (simclinic.explain), one block per condition."""
    blocks = []
    for e in engine.explanations(R, conditions, turns):
        name = escape(e.condition)
        items = [f'<li class="for">{escape(R.lexicon.lay(c.symptom)).capitalize()}, which {100 * c.percent:.0f}% of '
                 f'{name} patients have. <q>{escape(c.quote)}</q></li>' for c in e.support]
        items += [f'<li class="against">No {escape(R.lexicon.lay(c.symptom))}, although {100 * c.percent:.0f}% of '
                  f'{name} patients have it. <q>{escape(c.quote)}</q></li>' for c in e.against]
        items += [f'<li class="odd">{escape(R.lexicon.lay(c.symptom)).capitalize()} is not typical of it.</li>'
                  for c in e.unexplained]
        body = f'<ul>{"".join(items)}</ul>' if items else '<p class="none">Nothing the patient said points to it.</p>'
        blocks.append(f'<div class="why"><h4>{name}</h4>{body}</div>')
    st.markdown("".join(blocks), unsafe_allow_html=True)


def verdict(kind, title, body, extra="", animate=False):
    st.markdown(f'<div class="verdict {kind}{" anim" if animate else ""}"><h3>{escape(title)}</h3>'
                f'<div class="vbody">{body}</div>{extra}</div>', unsafe_allow_html=True)


def heading(mode):
    title, lede = MODES[mode]
    st.markdown(f"## {title}")
    st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


# --------------------------------------------------------------- welcome
@st.dialog("Welcome to SimClinic")
def welcome():
    st.markdown(
        '<div class="welcome"><p>Practise taking a medical history with an AI patient, or let an AI doctor '
        'interview you. Every case is a real condition a Singapore GP often sees.</p><dl>'
        f'<dt>{MODES["patient"][0]}</dt><dd>Ask questions, then make your diagnosis. The chart shows what you missed.</dd>'
        f'<dt>{MODES["doctor"][0]}</dt><dd>Act out a role card and see whether the doctor works it out.</dd>'
        f'<dt>{MODES["auto"][0]}</dt><dd>Two AIs run a consultation while you watch how the doctor reasons.</dd>'
        '</dl><small>A practice tool for history taking. Not medical advice.</small></div>', unsafe_allow_html=True)
    if st.button("Start practising", type="primary", use_container_width=True):
        st.rerun()


if not ss.get("welcomed"):
    ss.welcomed = True     # once per visit; closing the dialog with × also dismisses it
    welcome()


# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f'<div class="brand">{LOGO}<div><b>SimClinic</b><span>History-taking practice</span></div></div>',
                unsafe_allow_html=True)
    st.radio("Mode", list(MODES), format_func=lambda m: MODES[m][0], key="mode", on_change=new_case,
             captions=[MODES[m][1] for m in MODES], label_visibility="collapsed")
    st.write("")
    if ss.mode != "doctor":
        st.selectbox("Patient", R.patient_names, key="patient_name", on_change=new_case)
    if ss.mode != "patient":
        st.selectbox("Doctor", R.doctor_names, key="doctor_name", on_change=new_case)
    st.button("New case", on_click=new_case, use_container_width=True, type="primary")
    if engine.LOG_DIR:
        st.text_input("Your name, for the session log", key="player", placeholder="e.g. tanish")
    st.write("")
    st.toggle("Dark mode", key="dark")
    with st.expander("About the models"):
        st.caption(f"Cases: {R.cards_source}.  \nPatients: {', '.join(R.patient_names)}.  \n"
                   f"Doctors: {', '.join(R.doctor_names)}.")
        if R.patient_error:
            st.caption(f"The GPT-2 patient failed to load: {R.patient_error}")
        if R.doctor_error:
            st.caption(f"The GPT-2 doctor failed to load: {R.doctor_error}")
        st.caption("Refusal gate: " + ("on in Be the patient." if R.gate else
                   f"failed to load ({R.gate_error})." if R.gate_error else "not found, so the doctor answers everything."))
        if engine.LOG_DIR:
            st.caption(f"Logging finished sessions to {engine.LOG_DIR}/sessions.jsonl.")
    st.caption("A practice tool for history taking. Not medical advice.")

card = ss.card
animate = ss.get("anim", False)
ss.anim = False                      # the reveal plays once, not on every rerun (e.g. when rating)
heading(ss.mode)
chat, side = st.columns([3, 2], gap="large")

# ---------------------------------------------------------- patient mode
if ss.mode == "patient":
    with chat:
        n_q = sum(t["role"] == "doctor" for t in ss.turns)
        if ss.turns:
            st.markdown(f'<span class="counter">{n_q} question{"s" if n_q != 1 else ""} asked</span>',
                        unsafe_allow_html=True)
        with st.container(**CHAT_BOX):
            if not ss.turns:
                st.markdown('<div class="empty"><b>Your patient is in the room.</b><br>Start with an open question, '
                            'like "What brings you in today?", then narrow down.</div>', unsafe_allow_html=True)
            show_turns(ss.turns, you="doctor")
            if ss.get("pending"):
                st.markdown(typing_html("patient"), unsafe_allow_html=True)
        if ss.get("pending"):                     # the question is on screen; now the patient answers
            ss.turns.append({"role": "patient", "text": ss.patient.reply(ss.turns)})
            ss.pending = False
            st.rerun()
        elif not ss.done:
            if q := st.chat_input("Ask the patient a question"):
                ss.turns.append({"role": "doctor", "text": q})
                ss.pending = True
                st.rerun()
    with side:
        if not ss.done:
            chart(card)
            ss.guesses = st.multiselect("Your diagnosis, most likely first (up to 3)",
                                        sorted(engine.CONDITIONS), max_selections=3,
                                        placeholder="Search conditions")
            if st.button("Reveal diagnosis", type="primary", disabled=not ss.guesses or ss.get("pending"),
                         use_container_width=True):
                finish()
                st.rerun()
        else:
            s = engine.user_diagnosis_score(ss.guesses, card)
            n_q = sum(t["role"] == "doctor" for t in ss.turns)
            _, _, n_found = symptom_marks(ss.turns, card)
            stats = (f'<div class="stats"><div><b>{n_q}</b><small>questions asked</small></div>'
                     f'<div><b>{n_found} of {len(card["symptoms"])}</b><small>symptoms uncovered</small></div></div>')
            body = "Your differential:" + differential(ss.guesses, card["disease"])
            if s["top1"]:
                verdict("ok", "Correct diagnosis", body, stats, animate)
            elif s["top3"]:
                verdict("near", "In your differential, but not first", body, stats, animate)
            else:
                verdict("miss", "Not this time", body, stats, animate)
            chart(card, reveal=True, turns=ss.turns, animate=animate)
            with st.expander(f"What pointed to {card['disease']}"):
                why([card["disease"]], ss.turns)
            if engine.LOG_DIR:
                st.radio("How well did the patient stay in character? (1 = poorly, 5 = fully)", [1, 2, 3, 4, 5],
                         index=None, horizontal=True, key="rating")
            st.button("Next patient", on_click=new_case, type="primary", use_container_width=True)

# ----------------------------------------------------------- doctor mode
elif ss.mode == "doctor":
    with chat:
        st.markdown(f'<div class="notice">{escape(engine.EMERGENCY_NOTE)}</div>', unsafe_allow_html=True)
        n_q = sum(t["role"] == "doctor" for t in ss.turns)
        st.markdown(f'<span class="counter">Question {min(n_q, engine.MAX_QUESTIONS)} of {engine.MAX_QUESTIONS}</span>',
                    unsafe_allow_html=True)
        with st.container(**CHAT_BOX):
            show_turns(ss.turns, you="patient")
            if ss.get("pending"):
                st.markdown(typing_html("doctor"), unsafe_allow_html=True)
        if ss.get("pending"):                     # your answer is on screen; now the doctor replies
            turn = ss.doctor.next_turn(ss.turns, force_diagnosis=n_q >= engine.MAX_QUESTIONS)
            entry = {"role": "doctor", "text": turn.text}
            if getattr(turn, "refused", None):          # GuardedDoctor declined the last message
                entry["refused"] = turn.refused
            ss.turns.append(entry)
            ss.pending = False
            if getattr(turn, "ends", False):             # an emergency ends the consultation
                finish([])
            elif turn.diagnosis is not None:
                finish(turn.diagnosis)
            st.rerun()
        elif not ss.done:
            if a := st.chat_input("Answer the doctor"):
                ss.turns.append({"role": "patient", "text": a})
                ss.pending = True
                st.rerun()
    with side:
        if not ss.done:
            st.markdown('<p class="lede" style="margin:0 0 .6rem">Play this patient, or describe anything you like. '
                        'Answer in everyday words and don\'t name the condition.</p>', unsafe_allow_html=True)
            chart(card, reveal=True, turns=ss.turns, title="Your role card")   # highlights what the doctor has found
            if st.button("Ask for a diagnosis now", disabled=len(ss.turns) < 2 or ss.get("pending"),
                         use_container_width=True):
                turn = ss.doctor.next_turn(ss.turns, force_diagnosis=True)
                ss.turns.append({"role": "doctor", "text": turn.text})
                finish(turn.diagnosis or [])
                st.rerun()
        elif not ss.result:
            verdict("miss", "Consultation ended",
                    "The doctor stopped because your message sounded like an emergency. "
                    "In a real emergency, call 995.", animate=animate)
            chart(card, reveal=True, turns=ss.turns, title="Your role card", animate=animate)
            st.button("Next case", on_click=new_case, type="primary", use_container_width=True)
        else:
            hit = card["disease"] in ss.result[:3]
            verdict("ok" if ss.result[:1] == [card["disease"]] else "near" if hit else "miss",
                    "The doctor's differential",
                    f"If you played the role card, the answer was <b>{escape(card['disease'])}</b>."
                    + differential(ss.result, card["disease"]), animate=animate)
            with st.expander("Why these three?", expanded=True):
                why(ss.result[:3], ss.turns)
            chart(card, reveal=True, turns=ss.turns, title="Your role card", animate=animate)
            st.button("Next case", on_click=new_case, type="primary", use_container_width=True)

# ------------------------------------------------------------- auto mode
else:
    with chat:
        if not ss.done:
            pace = st.radio("Pace", list(PACE), index=1, horizontal=True, key="pace")
            if st.button("Start consultation", type="primary"):
                counter, live = st.empty(), st.container(**CHAT_BOX)
                steps = engine.auto_steps(R, ss.doctor, card, ss.seed, ss.get("patient_name"))
                speaker = "doctor"
                while True:
                    slot = live.empty()
                    slot.markdown(typing_html(speaker), unsafe_allow_html=True)
                    step = next(steps)
                    if isinstance(step, tuple):
                        slot.empty()
                        _, record, scores = step
                        ss.turns, ss.scores = record["turns"], scores
                        finish(record["diagnosis"])
                        break
                    ss.turns = ss.turns + [step]
                    n_q = sum(t["role"] == "doctor" for t in ss.turns)
                    counter.markdown(f'<span class="counter">Question {min(n_q, engine.MAX_QUESTIONS)} of '
                                     f'{engine.MAX_QUESTIONS}</span>', unsafe_allow_html=True)
                    slot.markdown(turn_html(step), unsafe_allow_html=True)
                    speaker = "patient" if step["role"] == "doctor" else "doctor"
                    time.sleep(PACE[pace])
                st.rerun()
            else:
                st.markdown('<div class="empty"><b>Ready when you are.</b><br>The doctor will ask up to '
                            f'{engine.MAX_QUESTIONS} questions, then give its top three diagnoses. '
                            'The chart on the right is sealed until it finishes.</div>', unsafe_allow_html=True)
        else:
            n_q = ss.scores["n_questions"]
            st.markdown(f'<span class="counter">{n_q} questions asked</span>', unsafe_allow_html=True)
            with st.container(**CHAT_BOX):
                show_turns(ss.turns)
    with side:
        if ss.done:
            s = ss.scores
            kind = "ok" if s["top1"] else "near" if s["top3"] else "miss"
            title = {"ok": "Correct diagnosis", "near": "In the differential, but not first",
                     "miss": "Missed the diagnosis"}[kind]
            nb = "right" if s["nb_top1"] else "wrong"
            body = (differential(ss.result, card["disease"])
                    + f'<p style="margin-top:.6rem">Naive Bayes, given the same answers, was {nb} at its first guess.'
                    + (" The doctor ran out of questions." if s["forced"] else "") + "</p>")
            verdict(kind, title, body, animate=animate)
            with st.expander("Why these three?"):
                why(ss.result[:3], ss.turns)
            chart(card, reveal=True, turns=ss.turns, animate=animate)
            st.button("Next case", on_click=new_case, type="primary", use_container_width=True)
        else:
            chart(card)
