# ██████╗  █████╗ ████████╗ █████╗ ██╗      ██████╗  ██████╗ ██████╗     █████╗ ██╗
# ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██║     ██╔═══██╗██╔═══██╗██╔══██╗   ██╔══██╗██║
# ██║  ██║███████║   ██║   ███████║██║     ██║   ██║██║   ██║██████╔╝   ███████║██║
# ██║  ██║██╔══██║   ██║   ██╔══██║██║     ██║   ██║██║   ██║██╔═══╝    ██╔══██║██║
# ██████╔╝██║  ██║   ██║   ██║  ██║███████╗╚██████╔╝╚██████╔╝██║        ██║  ██║██║
# ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝        ╚═╝  ╚═╝╚═╝

FROM nvcr.io/nvidia/pytorch:24.06-py3

# =================================================================================
# GENERAL REQUIREMENTS 
# =================================================================================

USER root
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Allow user with UID 1000 to use sudo without password
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser \
 && echo 'appuser ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/appuser \
 && chmod 0440 /etc/sudoers.d/appuser

WORKDIR /tmp/app

RUN git clone https://github.com/NVlabs/VILA.git
RUN chown -R 1000:1000 /tmp
RUN chmod -R a+w /tmp

# This is required to enable PEP 660 support
RUN python -m pip install --upgrade pip 

# Install base dependencies
RUN pip install \
    setuptools \
    dtlpy \
    openai \
    transformers \
    httpx==0.27.2

# Install FlashAttention2
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA
WORKDIR /tmp/app/VILA
RUN pip install -e ".[train,eval]"

# Install triton for quantization
RUN pip install triton==3.1.0

# Replace transformers and deepspeed files
RUN site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

# Downgrade protobuf to 3.20 for backward compatibility
RUN pip install protobuf==3.20.*
# Install OpenCV and its dependencies first

RUN pip install opencv-python==4.8.0.74

USER 1000
WORKDIR /tmp/app
# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/villa-model-adapter:0.2.10 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/villa-model-adapter:0.2.10

# docker run -it --gpus all gcr.io/viewo-g/piper/agent/runner/apps/villa-model-adapter:0.2.10 bash