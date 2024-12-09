FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/app

RUN apt-get update && apt-get install -y \
    libgl1 \
    git \
    libglib2.0-0 \
    curl \
    ca-certificates

COPY . .

WORKDIR /opt/app/packages/catan-ml

RUN uv sync

CMD ["uv", "run", "dramatiq", "catan_ml:redis_broker", "--threads=1", "--processes=1"]
