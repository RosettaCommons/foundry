# Base image reference:
# The official Foundry Docker image hosted by Rosetta Commons is available at:
# https://hub.docker.com/r/rosettacommons/foundry
#
# This Dockerfile serves as a reference starting point for building a custom
# Foundry image. You may optionally exclude checkpoint file downloads to
# reduce the final image size.
#
# Authors:
# Hope Woods <hope.woods@omsf.io>
# Rachel Clune <rachel.clune@omsf.io>
# Sergey Lyskov <sergey.lyskov@gmail.com>

ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS BUILDER

COPY --from=ghcr.io/astral-sh/uv /uv /uvx /bin/

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
     ca-certificates \
     git \
     mc \
     build-essential \
     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up app directory and user
WORKDIR /app

# Create a virtual environment and install dependencies with uv
RUN git clone https://github.com/RosettaCommons/foundry /app/foundry \
    && cd /app/foundry \
    && uv venv --python 3.12 \
    && uv pip install ".[all]" \
    && /app/foundry/.venv/bin/foundry --help \
    && /app/foundry/.venv/bin/foundry install base-models --checkpoint-dir /app/foundry/checkpoints \
    && uv cache clean \
    && chmod -R a+rX /root \
    && chmod -R a+rX /app \
    && echo

#FINAL -------------------------------------------------------------------
FROM ${BASE_IMAGE}

COPY --from=BUILDER /root /root
COPY --from=BUILDER /app /app
RUN chmod -R a+rX /root && chmod -R a+rX /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    mc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Activate the venv at container start
ENV PATH="/app/foundry/.venv/bin:$PATH"
WORKDIR /app/foundry

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

# Do not set a default command to run foundry since each models uses a different command
CMD ["bash"]

