FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System packages OpenCV/mediapipe/ffmpeg need at runtime (same list the
# devcontainer and Streamlit Cloud install from packages.txt).
COPY packages.txt /app/packages.txt
# The devcontainer base image ships a yarn apt source with an expired/
# unreachable signing key — unrelated to this app, but it fails `apt-get
# update` outright if left in place.
RUN rm -f /etc/apt/sources.list.d/*yarn* \
    && sed -i 's/\r$//' /app/packages.txt \
    && apt-get update \
    && xargs apt-get install -y < /app/packages.txt \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# A real (writable) home directory — Streamlit writes its own config/
# metrics files on startup; a "/nonexistent" home (docker init's default
# hardened-user template) makes every run crash with a PermissionError
# before the app even loads.
RUN adduser --disabled-password --gecos "" --home /home/appuser appuser \
    && mkdir -p /home/appuser/.streamlit \
    && chown -R appuser:appuser /home/appuser /app

USER appuser

COPY --chown=appuser:appuser . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0"]
