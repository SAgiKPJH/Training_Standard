# Docker 이미지 최적화 가이드

## 개요
기존 Docker 이미지가 너무 무거워서 Python 모듈만 설치된 가벼운 이미지로 최적화했습니다.

## 주요 개선사항

### 1. 멀티스테이지 빌드 (Multi-stage Build)
- **빌드 스테이지**: 모든 의존성 설치 및 컴파일
- **런타임 스테이지**: 필요한 파일만 복사하여 최종 이미지 생성
- **결과**: 빌드 도구들이 최종 이미지에 포함되지 않아 크기 대폭 감소

### 2. 베이스 이미지 최적화
- **기존**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04` (약 4GB+)
- **CPU 버전**: `python:3.10-slim` (약 200MB)
- **GPU 버전**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04` (필요시만)

### 3. 불필요한 패키지 제거
- **제거된 항목**:
  - .NET Runtime (약 200MB+)
  - Miniconda (약 500MB+)
  - 빌드 도구들
  - 개발 도구들
  - 캐시 파일들

### 4. 레이어 최적화
- 관련 명령어들을 하나의 RUN 명령어로 결합
- 패키지 캐시 정리
- 불필요한 파일 제거

## 사용법

### CPU 버전 빌드 (권장)
```bash
docker build -f ./Dockerfile -t mirero/daq:optimized-cpu .
```

### GPU 버전 빌드 (필요시만)
```bash
docker build -f ./Dockerfile.gpu -t mirero/daq:optimized-gpu .
```

### 자동 빌드 스크립트
```bash
chmod +x build.sh
./build.sh
```

## 예상 크기 감소

| 버전 | 기존 크기 | 최적화 후 크기 | 감소율 |
|------|-----------|----------------|--------|
| CPU 버전 | ~6GB | ~2-3GB | 50-70% |
| GPU 버전 | ~6GB | ~4-5GB | 20-30% |

## 주의사항

1. **GPU 사용시**: GPU 버전을 사용해야 합니다
2. **.NET 의존성**: .NET 관련 기능이 필요한 경우 별도 설치 필요
3. **빌드 시간**: 멀티스테이지 빌드로 인해 초기 빌드 시간이 약간 증가할 수 있음

## 추가 최적화 팁

1. **Alpine Linux 사용**: 더 가벼운 베이스 이미지 (하지만 호환성 주의)
2. **패키지 버전 고정**: 특정 버전을 사용하여 예측 가능성 향상
3. **레이어 캐싱**: 자주 변경되지 않는 레이어를 먼저 배치

## 문제 해결

### 빌드 실패시
```bash
# 캐시 없이 빌드
docker build --no-cache -f ./Dockerfile -t mirero/daq:optimized-cpu .

# 상세 로그 확인
docker build --progress=plain -f ./Dockerfile -t mirero/daq:optimized-cpu .
```

### 이미지 크기 확인
```bash
docker images mirero/daq
```
