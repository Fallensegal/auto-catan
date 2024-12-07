FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates

COPY . .

WORKDIR /opt/app/packages/catan-web

RUN uv sync

CMD ["uv", "run", "uvicorn", "catan_web:app", "--host=0.0.0.0", "--port=8251"]