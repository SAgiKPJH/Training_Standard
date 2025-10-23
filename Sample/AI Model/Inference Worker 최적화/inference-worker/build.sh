docker build -f ./Dockerfile -t mirero/daq:inference-base --no-cache . && docker image prune -f && \
docker build -f ./Dockerfile.s2 -t mirero/daq:inference-base-s2 --no-cache . && docker image prune -f
