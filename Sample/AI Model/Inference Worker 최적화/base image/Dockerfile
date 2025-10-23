# FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV TZ=Asia/Seoul

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    # Install dotnet runtime prerequisites
    wget \
    curl \
    apt-utils \
    ca-certificates \
    # Install gpu prerequisites
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Install tesseract 
    tesseract-ocr tesseract-ocr-kor && \
    # Install Microsoft package feed
    wget -q https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb  && \
    dpkg -i packages-microsoft-prod.deb  && \
    rm packages-microsoft-prod.deb && \
    # Install .NET
    apt-get update && \
    apt-get install -y --no-install-recommends dotnet-runtime-7.0 && \
    # Cleanup
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

COPY pyproject.toml poetry.lock ./

RUN conda init bash
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n daq python=3.10

SHELL ["conda", "run", "-n", "daq", "/bin/bash", "-c"]

RUN pip install poetry==1.7.1 && \
	poetry config virtualenvs.create false && \
	poetry install --without dev --no-cache  --no-interaction  --no-root && \
	# Remove poetry cache to reduce size of docker image 
	rm -Rf ~/.cache/pypoetry