FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app/convolutional_neural_network
COPY . .
RUN pip3 install -r requirements.txt

CMD ["bash", "run.sh"]