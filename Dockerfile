FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# update ubuntu
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y

# install python3.6
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6

RUN apt-get update \
  && apt-get install python3.6 python3.6-dev python3-pip make curl git sudo cron -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3.6 python

RUN mkdir -p /code
WORKDIR /code

COPY requirements.txt ./
RUN python -m pip install pip --upgrade \
 && python -m pip install -r requirements.txt

COPY . /code