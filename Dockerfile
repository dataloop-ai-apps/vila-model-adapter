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

WORKDIR /tmp/app
ENV PYTHONPATH=/tmp/app:$PYTHONPATH
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN usermod -aG sudo $(getent passwd 1000 | cut -d: -f1) && \
echo "$(getent passwd 1000 | cut -d: -f1) ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/1000 && \
chmod 0440 /etc/sudoers.d/1000

RUN chown -R 1000:1000 /tmp/app

USER 1000

# =================================================================================
# DATALOOP AI REQUIREMENTS
# =================================================================================

RUN pip install dtlpy openai transformers

# =================================================================================
# VILA REQUIREMENTS
# =================================================================================
# Using the following Dockerfile as reference:
# https://github.com/NVlabs/VILA/blob/main/Dockerfile

RUN git clone https://github.com/NVlabs/VILA.git .
RUN pip install --upgrade pip setuptools \
    && pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    && pip install -e ".[train,eval]" \
    && pip install triton==3.1.0 \
    && site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') \
    && cp -rv ./llava/train/deepspeed_replace/* "$site_pkg_path/deepspeed/" \
    && pip install protobuf==3.20.*