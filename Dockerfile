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
    wget \
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

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda accessible in RUN commands
RUN conda init bash
SHELL ["/bin/bash", "--login", "-c"]

# This is required to enable PEP 660 support
RUN python -m pip install --upgrade pip 

# Install global dependencies
RUN pip install \
    dtlpy \
    openai

# Create Conda environment for VILA
ENV CONDA_ENV_NAME vila_env
RUN conda create -n $CONDA_ENV_NAME python=3.10.14 -y

# Install VILA dependencies within the environment
RUN echo "source activate $CONDA_ENV_NAME" > ~/.bashrc
# Activate conda env and install dependencies
RUN . activate $CONDA_ENV_NAME && \
    pip install --upgrade pip setuptools && \
    # Install FlashAttention2
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    # Install VILA
    pip install -e "/tmp/app/VILA[train,eval]" && \
    # Install triton for quantization
    pip install triton==3.1.0 && \
    # Replace transformers and deepspeed files
    site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    cp -rv /tmp/app/VILA/llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/ && \
    # Downgrade protobuf to 3.20 for backward compatibility
    pip install protobuf==3.20.*

USER 1000
WORKDIR /tmp/app/VILA
# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/villa-model-adapter:0.2.10 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/villa-model-adapter:0.2.10

# docker run -it --gpus all gcr.io/viewo-g/piper/agent/runner/apps/villa-model-adapter:0.2.10 bash
