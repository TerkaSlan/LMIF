FROM python:3.6

MAINTAINER Terézia Slanináková, xslanin@mail.muni.cz

RUN apt-get update && apt-get install -y vim coreutils

COPY . /learned-indexes
WORKDIR /learned-indexes

RUN addgroup --gid 1000 user && adduser --gid 1000 --uid 1000 --disabled-password --gecos user user

USER root
RUN chown -R user:user /learned-indexes
RUN chmod 755 /learned-indexes
USER user

# Ignoring warning from pip upgrade to avoid: "You are using pip version xy. a, however version xy. b is available" error
RUN pip install --upgrade pip >/dev/null 2>&1
COPY --chown=user:user requirements.txt requirements.txt
ENV PATH="/home/user/.local/bin:${PATH}"
RUN pip install --user -r requirements.txt
