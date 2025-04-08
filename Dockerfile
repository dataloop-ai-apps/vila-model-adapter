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
# =================================================================================
# DATALOOP AI REQUIREMENTS
# =================================================================================

RUN pip install dtlpy openai transformers

# =================================================================================
# VILA REQUIREMENTS
# =================================================================================
# Using the following Dockerfile as reference:
# https://github.com/NVlabs/VILA/blob/main/Dockerfile


RUN git clone https://github.com/NVlabs/VILA.git /app/vila
RUN bash /app/vila/environment_setup.sh vila