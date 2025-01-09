# Use the NVIDIA PyTorch container as the base image
FROM nvcr.io/nvidia/pytorch:24.01-py3 AS nvidia-llm-wyoming

# Instructions initially from https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/starthere/intro.html
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install additional dependencies with caching
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y libsndfile1 ffmpeg

# Install Python packages with caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install Cython huggingface_hub[cli] fastapi ipython
# pip install Cython nemo_toolkit['all'] huggingface_hub[cli] fastapi ipython

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install git+https://github.com/NVIDIA/NeMo.git@r2.1.0#egg=nemo_toolkit[asr]

WORKDIR /app

COPY docker/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
