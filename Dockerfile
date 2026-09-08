FROM mambaorg/micromamba:1.5.6

LABEL maintainer="nih-nlm"
LABEL org.opencontainers.image.title="pvclust-py"
LABEL org.opencontainers.image.description="Hierarchical clustering with AU p-values via multiscale bootstrap, and its federated form"
LABEL org.opencontainers.image.licenses="GPL-3.0-or-later"

USER root:root

RUN apt-get update && \
    apt-get install -y git procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the repository, mirroring the sibling repos.
#
# NOTE: this builds what is PUSHED, not your working tree, so the build can succeed
# while the image holds older code. Push first, then build. To check a branch before
# it merges:  docker build --build-arg PVCLUST_REF=my-branch -t pvclust-py:test .
ARG PVCLUST_REF=main
RUN git clone --branch "${PVCLUST_REF}" --depth 1 \
        https://github.com/NIH-NLM/pvclust-py.git && \
    chown -R mambauser:mambauser /app/pvclust-py

USER mambauser:mambauser

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    DEBIAN_FRONTEND=noninteractive

# Python + pip; all package dependencies come from pyproject.toml
RUN micromamba install -y -n base -c conda-forge python=3.12 pip && \
    micromamba clean --all --yes

WORKDIR /app/pvclust-py
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir ".[notebooks]"

ENV PYTHONPATH="/app/pvclust-py/src"

# JupyterLab listens here when the notebook entrypoint is used (see below).
EXPOSE 8888

# Default is the CLI, matching the sibling repos, so Nextflow processes can call
# `pvclust-py <command>` directly. To run the notebooks instead:
#
#   docker run --rm -p 8888:8888 -v "$PWD/data:/data" ghcr.io/nih-nlm/pvclust-py:latest \
#     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
#                 --notebook-dir=/app/pvclust-py/ipynb
#
# Mount your data read-only at /data and point the notebooks there; the image
# deliberately contains no study data.
CMD ["pvclust-py", "--help"]
