FROM python:3.9
ENV PYTHONUNBUFFERED 1
RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip && pip install -r requirements.txt
ADD app/ /app/
ADD scripts/ /scripts/
RUN chmod +x /scripts/*
CMD ["/scripts/entrypoint.sh"]