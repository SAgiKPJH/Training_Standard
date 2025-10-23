# docker build -f ./Dockerfile -t mirero/daq:ubuntu20.04-python3.10-cuda11.6.2-cudnn8 --no-cache .

docker build -f ./Dockerfile -t mirero/daq:ubuntu20.04-python3.10-cuda11.8-cudnn8-torch2.7-sam2 .
# docker build -f ./Dockerfile -t mirero/daq:ubuntu20.04-python3.10-cuda11.8-cudnn8-torch2.7-sam2 --no-cache .

docker build -f ./Dockerfile.exceptminiconda -t mirero/daq:optimization .
docker build -f ./Dockerfile.exceptminiconda.minipython -t mirero/daq:optimization-exception-minipython .
docker build -f ./Dockerfile.exception -t mirero/daq:optimization-exception .

docker build -f ./Dockerfile.python -t mirero/daq:optimization-python .
docker build -f ./Dockerfile.python0 -t mirero/daq:optimization-python0 .
docker build -f ./Dockerfile.python2 -t mirero/daq:optimization-python2 .