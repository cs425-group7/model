"""App smoke tests. Run: python -m pytest tests/test_app.py -q"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def start(dark=False):
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["welcomed"] = True        # skip the first-visit welcome dialog
    at.session_state["dark"] = dark
    return at.run()


def test_patient_mode():
    at = start()
    assert not at.exception
    at.chat_input[0].set_value("What brings you in today?").run()
    at.chat_input[0].set_value("Any fever?").run()
    assert len(at.session_state.turns) == 4
    at.multiselect[0].set_value(["Flu"]).run()
    next(b for b in at.button if b.label == "Reveal diagnosis").click().run()
    assert not at.exception and at.session_state.done


def test_doctor_mode():
    at = start()
    at.sidebar.radio[0].set_value("doctor").run()
    assert len(at.session_state.turns) == 1          # the doctor opens
    at.chat_input[0].set_value("I've had a fever and a bad headache since yesterday").run()
    next(b for b in at.button if b.label == "Ask for a diagnosis now").click().run()
    assert not at.exception and at.session_state.done


def test_auto_mode():
    at = start()
    at.sidebar.radio[0].set_value("auto").run()
    at.radio(key="pace").set_value("Fast").run()
    next(b for b in at.button if b.label == "Start consultation").click().run()
    assert not at.exception and at.session_state.done and len(at.session_state.turns) >= 3


def test_auto_steps_matches_run_consultation():
    """The live auto mode must produce exactly what consult.run_consultation (the evaluation) produces."""
    from app import engine

    res = engine.load_resources(load_gpt2=False)
    for card in res.case_pool[:5]:
        doc = lambda: engine.make_doctor(res, res.doctor_names[0], seed=7)
        expected, exp_scores = engine.auto_consult(res, doc(), card, seed=7)
        *_, (_, record, scores) = engine.auto_steps(res, doc(), card, seed=7)
        assert record == expected and scores == exp_scores


def test_model_patient_adapter():
    """The fine-tuned patient plugs into run_consultation through ModelPatient (fake model, no torch)."""
    from app import engine

    class FakeSim:
        def reply(self, card, history, **gen):
            assert history[-1]["role"] == "doctor"
            return "I've had a fever since yesterday."

    res = engine.load_resources(load_gpt2=False)
    res.gpt2_patient = FakeSim()
    card = res.case_pool[0]
    assert isinstance(engine.make_patient(res, card, name=engine.GPT2_PATIENT), engine.ModelPatient)
    assert isinstance(engine.make_patient(res, card, name=engine.RULE_PATIENT), engine.CardPatient)
    record, _ = engine.auto_consult(res, engine.make_doctor(res, res.doctor_names[0]), card,
                                    patient_name=engine.GPT2_PATIENT)
    assert record["turns"][1]["text"].startswith("I've had a fever")


def test_doctor_mode_refuses_out_of_scope_and_ends_on_emergency():
    """With the team's refusal gate present, Be the patient uses GuardedDoctor."""
    pytest.importorskip("sklearn")
    from app import engine
    if not engine.GATE_PATH.exists():
        pytest.skip("no refusal_gate.pkl")
    at = start()
    at.sidebar.radio[0].set_value("doctor").run()
    at.chat_input[0].set_value("what's the weather like today?").run()
    assert at.session_state.turns[-1].get("refused") and not at.session_state.done
    at.chat_input[0].set_value("my chest is crushing and I can't breathe").run()
    assert at.session_state.turns[-1].get("refused") == "emergency" and at.session_state.done
    assert not at.exception


def test_session_log_matches_simclinic_session_format(tmp_path, monkeypatch):
    """App sessions are logged with the same keys as simclinic.session, so notebook 13 reads both."""
    import json
    from app import engine
    monkeypatch.setattr(engine, "LOG_DIR", str(tmp_path))
    engine.log_session({"mode": "patient", "player": "t", "seed": 1, "card": {"disease": "Flu"}, "turns": [],
                        "diagnosis": ["Flu"], "top1": True, "top3": True, "in_character": 4, "minutes": 1.0})
    rec = json.loads((tmp_path / "sessions.jsonl").read_text().splitlines()[0])
    assert {"time", "mode", "player", "seed", "card", "turns", "diagnosis", "top1", "top3", "minutes"} <= set(rec)
    assert rec["source"] == "app"


def test_dark_mode_runs_every_mode():
    at = start(dark=True)
    for mode in ("patient", "doctor", "auto"):
        at.sidebar.radio[0].set_value(mode).run()
        assert not at.exception
    at.sidebar.toggle[0].set_value(False).run()
    assert not at.exception and at.session_state.dark is False


def test_welcome_shows_once():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception and at.session_state.welcomed


def test_theme_from_url():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["theme"] = "dark"
    at.run()
    assert not at.exception and at.session_state.dark is True
