#!/bin/bash

PYTHON=/root/miniconda3/envs/daq/bin/python

# paddle/libs 경로 확인
PADDLE_LIBS=$(${PYTHON} -c 'import paddle, os; print(os.path.join(os.path.dirname(paddle.__file__), "libs"))')

# [cuDNN] Dockerfile에서 versioned .so.8 파일 복사됨 → unversioned .so 심볼릭 링크만 추가
find "${PADDLE_LIBS}" -maxdepth 1 -name "lib*.so.*" | while read versioned; do
    base=$(echo "$versioned" | sed 's/\.so\..*//')
    unversioned="${base}.so"
    [ ! -e "${unversioned}" ] && ln -sf "$(basename "$versioned")" "${unversioned}"
done

# [cublas] targets/x86_64-linux/lib 에 위치 → paddle/libs 에 복사 + unversioned 링크
CUBLAS_SRC=/usr/local/cuda-11.8/targets/x86_64-linux/lib
for lib in libcublas libcublasLt; do
    cp -an "${CUBLAS_SRC}/${lib}.so."* "${PADDLE_LIBS}/" 2>/dev/null || true
    major=$(ls "${PADDLE_LIBS}/${lib}.so."* 2>/dev/null | grep -oP '\.\d+$' | head -1 | tr -d '.')
    [ -n "$major" ] && [ ! -e "${PADDLE_LIBS}/${lib}.so" ] && \
        ln -sf "${lib}.so.${major}" "${PADDLE_LIBS}/${lib}.so"
done

export LD_LIBRARY_PATH=${PADDLE_LIBS}:${CUBLAS_SRC}:${LD_LIBRARY_PATH}

$PYTHON /workspace/main.py
