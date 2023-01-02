FROM python:3.6

MAINTAINER Terezia Slaninakova, xslanin@mail.muni.cz

RUN apt-get update && apt-get install -y vim coreutils
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

COPY . /learned-dynamicity
WORKDIR /learned-dynamicity

RUN addgroup --gid 1000 user && adduser --gid 1000 --uid 1000 --disabled-password --gecos user user

USER root
RUN chown -R user:user /learned-dynamicity
RUN chmod 755 /learned-dynamicity
USER user

# Ignoring warning from pip upgrade to avoid: "You are using pip version xy. a, however version xy. b is available" error
RUN pip install --upgrade pip >/dev/null 2>&1
COPY --chown=user:user requirements.txt requirements.txt
ENV PATH="/home/user/.local/bin:${PATH}"
RUN pip install --user -r requirements.txt
