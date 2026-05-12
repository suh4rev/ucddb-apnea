# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS=requirements.txt

COPY requirements.txt requirements-dl.txt requirements-mlops.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install -r "${REQUIREMENTS}"

COPY config.py README.md ./
COPY docs ./docs
COPY scripts ./scripts

RUN mkdir -p data/raw data/processed reports/tables reports/figures

CMD ["python", "-m", "compileall", "-q", "scripts"]
