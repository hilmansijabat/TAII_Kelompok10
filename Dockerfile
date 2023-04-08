#FROM python:3.9
#ENV PYTHONUNBUFFERED 1
#RUN mkdir /app
#WORKDIR /app
#ADD requirements.txt /app/
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN pip install -r requirements.txt
#ADD . /app/
#CMD ["uwsgi", "--socket", ":9000", "--workers", "4", "--master", "--enable-threads", "--module", "app.wsgi"]

FROM python:3.10-alpine3.16

ENV PYTHONUNBUFFERED 1

COPY requirements.txt /requirements.txt
RUN apk add --upgrade --no-cache build-base linux-headers && \
    pip install --upgrade pip && \
    pip install -r /requirements.txt

COPY app/ /app
WORKDIR /app

RUN adduser --disabled-password --no-create-home django

USER django

CMD ["uwsgi", "--socket", ":9000", "--workers", "4", "--master", "--enable-threads", "--module", "app.wsgi"]