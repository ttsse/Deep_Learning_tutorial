FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app/src/linear_regression
COPY . .
RUN pip3 install -r requirements.txt
USER root
CMD ["python3", "main.py"]