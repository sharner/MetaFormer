ARG PYTORCH="22.06"

FROM nvcr.io/nvidia/pytorch:${PYTORCH}-py3

#ARG USER_ID 1000
#ARG GROUP_ID 1000

RUN TZ=America/Los_Angeles apt-get update && DEBIAN_FRONTEND=noninteractive \
apt-get install -y dialog apt-utils tzdata

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    nodejs \
    npm \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    lsb-release \
    libfreetype6-dev \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    chromium-chromedriver \
    apt-transport-https \
    ca-certificates \
    curl wget \
    redis-tools \
    apt-transport-https gnupg lsb-release

#RUN echo "root:root" | chpasswd

#RUN addgroup --gid 1000 user

#RUN useradd -G video,audio -ms /bin/bash --uid 1000 --gid 1000 user

RUN apt-get update && \
    apt-get -y install sudo

#RUN echo "user:user" | chpasswd && adduser user sudo

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
# RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
#    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#    apt-get update -y && apt-get install google-cloud-sdk -y
# RUN apt-get install -y curl -SL https://sdk.cloud.google.com | bash

WORKDIR /workspace
COPY ./requirements.txt /workspace/

ENV LAYERJOT_HOME="/layerjot"
ENV HOME=/home/forest
ENV PATH=$PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN cd /workspace && pip install -r requirements.txt \
    && python -m pip install --upgrade pip

RUN curl -LO https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y ./google-chrome-stable_current_amd64.deb
RUN rm google-chrome-stable_current_amd64.deb

# RUN pip install numpy --upgrade
# RUN pip install pandas --upgrade
RUN pip install --upgrade gensim

#USER user
