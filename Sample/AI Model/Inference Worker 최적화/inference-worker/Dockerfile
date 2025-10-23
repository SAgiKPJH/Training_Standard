# Stage 1: envoyproxy/envoy에서 envoy 실행 파일만 추출
FROM mirero/daq:envoy as envoy-extract

# -------------------------------------------------
# Stage 2: Python base with app
FROM mirero/daq:ubuntu20.04-python3.10-cuda11.8-cudnn8-torch2.7-sam2

# -------------------------------------------------
# Stage 3: base 이미지에 envoy 복사
# envoy 바이너리 복사 (gRPC-Web 필터 포함된 정식 바이너리)
COPY --from=envoy-extract /usr/local/bin/envoy /usr/local/bin/envoy

COPY envoy.yaml /etc/envoy/envoy.yaml
