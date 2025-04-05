FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.8.2

WORKDIR /app

# Dependências
RUN apt update \
 && apt install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        git \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

# instalação do Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry
COPY ./pyproject.toml ./poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# Copiando os arquivos
COPY ./api /app/api
COPY ./training /app/training

EXPOSE 8112
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8112"]