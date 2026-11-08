FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app

# CPU-only torch keeps the image small. With this wheel `docker run --gpus all` has no effect.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt streamlit

COPY simclinic/ simclinic/
# Symcat is public, so fetch it at build time rather than shipping it.
RUN python -c "from pathlib import Path; from simclinic.symcat import download; download(Path('data/raw/symcat'))"

COPY app/ app/
COPY .streamlit/ .streamlit/
# Optional, picked up if present (see .dockerignore): the team's cards and the selected checkpoints.
# Both folders must exist, even if empty: mkdir -p checkpoints models
COPY data/ data/
COPY checkpoints/ checkpoints/
COPY models/ models/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
