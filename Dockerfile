FROM python:3.6

MAINTAINER Terézia Slanináková, xslanin@mail.muni.cz

RUN apt-get update && apt-get install -y vim coreutils

COPY . /learned-indexes
WORKDIR /learned-indexes

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN addgroup --gid 1000 user && adduser --gid 1000 --uid 1000 --disabled-password --gecos user user
USER root
RUN chown -R user:user /learned-indexes
RUN chmod 755 /learned-indexes
USER user
