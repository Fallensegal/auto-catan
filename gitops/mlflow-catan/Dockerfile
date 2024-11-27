FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/app

RUN apt-get update && apt-get install -y \
    libg12-mesa-glx \
    libglib2.0-0 \
    curl \
    ca-certificates

RUN uv pip install mlflow --system

CMD ["mlflow", "server", "--host 0.0.0.0", "--port 8250"] 

