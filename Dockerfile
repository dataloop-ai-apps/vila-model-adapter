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

ENV PYTHONPATH=/tmp/app:$PYTHONPATH
RUN apt-get update && apt-get install -y python3-dev python3-pip curl git sudo docker.io python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER root

# Allow user with UID 1000 to use sudo without password
RUN echo "%#1000 ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/uid1000
RUN chmod 0440 /etc/sudoers.d/uid1000

WORKDIR /tmp/app
RUN chown -R 1000:1000 /tmp
RUN chmod -R a+w /tmp

USER 1000

RUN git clone https://github.com/NVlabs/VILA.git .
RUN python3 -m venv .venv

# Activate venv for subsequent RUN commands implicitly by using its executables
ENV PATH="/tmp/app/.venv/bin:$PATH"

# Use venv's pip
RUN pip install --upgrade pip setuptools
RUN pip install dtlpy

RUN pip install \
    openai \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    -e ".[train,eval]" \
    triton==3.1.0 \
    && site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])') \
    && cp -rv ./llava/train/deepspeed_replace/* "$site_pkg_path/deepspeed/" \
    && pip install protobuf==3.20.* \
    httpx==0.27.2