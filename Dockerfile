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

WORKDIR /app
ENV PYTHONPATH=/app:$PYTHONPATH
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Add conda-forge channel and set retry attempts
RUN conda config --add channels conda-forge \
    && conda config --set remote_connect_timeout_secs 60 \
    && conda config --set remote_read_timeout_secs 120 \
    && conda config --set remote_max_retries 5

# =================================================================================
# DATALOOP AI REQUIREMENTS
# =================================================================================

RUN pip install dtlpy openai transformers

# =================================================================================
# VILA REQUIREMENTS
# =================================================================================
# Using the following Dockerfile as reference:
# https://github.com/NVlabs/VILA/blob/main/Dockerfile

RUN git clone https://github.com/NVlabs/VILA.git . \
    && for i in {1..3}; do bash ./environment_setup.sh vila && break || sleep 15; done