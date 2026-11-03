# Model

Model code for SimClinic

### Dataset source
Med Dialog EN: https://huggingface.co/datasets/UCSD26/medical_dialog/blob/main/medical_dialog.py
Diagnose Me: https://www.kaggle.com/datasets/dsxavier/diagnoise-me?resource=download&select=en_medical_dialog.json
Symcat: https://github.com/teliov/symcat-to-synthea (`symcat/`; downloaded by notebook 03 into `data/raw/symcat/`)

## Data (notebooks 06–07)

| Notebook | What it does |
|---|---|
| `06_data_audit.ipynb` | Re-runs the proposal's data audit from the raw files: Symcat size, the 12-symptom cap, tied percentages and the quality filter (801 / 474 / 9,162 / 687 / 296 / 391), and MedDialog-EN's one-question-one-answer structure, doctor questions and greetings, and patient message length |
| `07_cards_and_dialogues.ipynb` | Builds the 2,013 case cards (61 conditions, 80/10/10 by card, stratified by disease), checks the symptom lexicon, summarises the generation run, applies the final filter and writes `dialogues.jsonl`, plus 100 dialogues for the manual check |

Outputs (git-ignored, under `data/processed/`):
- `cards/cards.jsonl`: `{card_id, disease, symptoms (presenting complaint first), persona, split}`
- `dialogues/dialogues.jsonl`: `{dialogue_id, card_id, split, disease, symptoms, persona, turns, diagnosis (true disease first), source}`. Notebook 03 (doctor) reads it, and the patient model can build its hidden header from the card fields in each record.
- `dialogues/review/`: `review_sample.md` to read and `review_sheet.csv` with the marks from the manual review (findings at the end of notebook 07, section 7)

**Generating the dialogues** (local, no external API; about 6–7 h for ~5,200 dialogues on an RTX 4070; resumable):
```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &
ollama pull qwen2.5:14b-instruct-q4_K_M
python -m simclinic.generate --cards data/processed/cards/cards.jsonl --out-dir data/processed/dialogues
# after changing the lexicon or the filter: re-filter accepted dialogues and regenerate the ones that now fail
# (attempts count across runs and some dialogues have used 12, so keep --max-attempts above that)
python -m simclinic.generate --cards data/processed/cards/cards.jsonl --out-dir data/processed/dialogues --recheck --max-attempts 15
```
Run notebook 07 once before generating (to write the cards) and again afterwards. Generation needs notebook 02's cleaned MedDialog patient messages, which are used as style examples.

**Symptom lexicon.** `simclinic/symptom_lexicon.json` covers every symptom of the 61 conditions. `lexicon.py` matches the listed phrasings plus spoken variants derived from them ("sore throat" → "my throat feels sore"). `Lexicon.on_card()` is the symptom-fidelity test that the dialogue filter uses and the evaluation should use too.

## Doctor model (notebooks 03–05)

| Notebook | What it does |
|---|---|
| `03_doctor_data.ipynb` | Parses Symcat (re-checks the audit numbers), builds case cards and consultations, and tokenizes doctor training strings with the loss only on doctor turns and the diagnosis |
| `04_doctor_train.ipynb` | Fine-tunes GPT-2 small and medium over the learning-rate grid with early stopping, and picks the best run by an auto-mode dev score |
| `05_doctor_eval.ipynb` | Auto-mode and fixed-transcript evaluation of the NB doctor, pretrained GPT-2, fine-tuned GPT-2, and LLaMA-3-8B / Mistral-7B with zero-shot, few-shot and CoT prompts |

Shared code lives in `simclinic/`, so the notebooks and the app use the same implementation:
`symcat.py` (parser and quality filter), `lexicon.py` + `symptom_lexicon.json` (lay phrasings, revealed-symptom extraction and the on-card test), `generate.py` (dialogue generation and filter), `naive_bayes.py`, `cards.py`, `patients.py` (rule-based card patient), `format.py` (training-string format), `doctors.py` (`NBDoctor`, `GPT2Doctor`, `LLMDoctor`), `prompts.py`, `llm.py` (Ollama and Hugging Face backends) and `consult.py` (auto mode and scoring).

**Data.** Notebook 03 reads the cards and dialogues from notebook 07. Setting `CARDS_JSONL` / `DIALOGUES_JSONL` to `None` makes it build the earlier stand-in instead: Symcat-sampled cards plus dialogues between the information-gain NB doctor and the rule-based patient.

**Setup**
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# LLM baselines (local, no external API):
ollama serve &
ollama pull llama3:8b-instruct-q4_K_M && ollama pull mistral:7b-instruct-v0.2-q4_K_M
```
Checkpoints are written to `checkpoints/doctor/` (git-ignored). `checkpoints/doctor/best/` holds the selected model and its `doctor_config.json`.
