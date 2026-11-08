# SimClinic app (Tanish)

Streamlit app on top of the shared `simclinic` package. It calls the team's code
directly: `CardPatient`, `NBDoctor` / `GPT2Doctor`, `run_consultation` and `score`
(consult.py), `revealed_symptoms` / `on_card` (lexicon.py). Nothing is reimplemented.

Run from the repo root:

    uv pip install streamlit
    streamlit run app/streamlit_app.py
    python -m pytest tests/test_app.py -q

Modes: interview a patient (user is the doctor), be the patient (user talks to the
AI doctor), and watch the AI doctor (auto mode, scored with consult.score).

Picked up automatically if present:
- data/processed/cards/cards.jsonl  (otherwise 610 stand-in cards from Symcat)
- checkpoints/doctor/best/          (adds "GPT-2 (fine-tuned)" to the doctor menu)
- models/patient/best/              (adds the fine-tuned patient via simclinic.patient.PatientSimulator;
                                     copy the chosen notebook 08 run folder here)
Symcat is downloaded to data/raw/symcat/ on first run.

Be the patient wraps the doctor in the team's GuardedDoctor (simclinic.refusal) when
checkpoints/doctor/refusal_gate.pkl is present, and every differential can show the team's
grounded explanations (simclinic.explain).

Human sessions for the evaluation: `SIMCLINIC_LOG_DIR=data/human_sessions streamlit run app/streamlit_app.py`
appends each finished session to data/human_sessions/sessions.jsonl, the same file and record format as
`python -m simclinic.session`, so notebook 13 reads both (app records have "source": "app").

Docker (`mkdir -p checkpoints models` first, so both folders exist):

    docker build -t simclinic .
    docker run -p 8501:8501 simclinic      # then open http://localhost:8501

Without models/patient/best the patient is the rule-based CardPatient.
