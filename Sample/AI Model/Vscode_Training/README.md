# VSCode Web Training Development Environment

이 프로젝트는 Docker를 사용하여 GPU 지원이 되는 VSCode Web 개발 환경을 제공합니다.

## 요구사항
- Docker
- Docker Compose
- NVIDIA GPU (선택사항)

## 사용법

### 1. 환경 설정
현재 GPU 2개를 기준으로 설정되어 있습니다. 다른 GPU 개수를 사용하려면 docker-compose.yml을 직접 수정하세요.

### 2. 서비스 시작
```bash
docker-compose up -d
```

### 3. VSCode Web 접속
2개의 GPU 기반 서비스가 실행되며, 각 서비스는 다른 포트로 접근합니다:

- **training-develop-0** (GPU 0): http://localhost:18080
- **training-develop-1** (GPU 1): http://localhost:18081

사용자명: mirero / 비밀번호: system

### 4. MinIO 접속
- MinIO는 별도에서 실행 중이어야 합니다
- traindata 버킷이 자동으로 생성되며, 각 VSCode 서비스에서 traindata 경로로 접근 가능

### 5. 패키지 정보 확인
- 컨테이너 시작 시 자동으로 `pyproject.toml` 및 `INSTALLED_PACKAGES.md` 파일 생성
- 설치된 Python 패키지 목록과 GPU 정보 확인 가능

### 6. 개발 환경
- Python 개발 환경 (Conda + Poetry)
- Jupyter Notebook 지원
- GPU 지원 TensorFlow, PyTorch
- 각 GPU별로 별도의 작업 폴더가 생성됨 (training-develop-0, training-develop-1, ...)

## 폴더 구조
```
.
├── base_image/          # Python 개발 환경 베이스 이미지
├── vscode-dev/          # VSCode Web 이미지 설정
│   ├── Dockerfile
│   └── entrypoint.sh
├── docker-compose.yml   # 서비스 설정
├── training-develop/    # 공통 개발 작업 폴더
├── training-develop-0/  # GPU 0용 작업 폴더
├── training-develop-1/  # GPU 1용 작업 폴더
├── pyproject.toml       # Python 패키지 설정 (자동 복사)
└── INSTALLED_PACKAGES.md # 설치된 패키지 정보 (자동 생성)
```

## 주요 기능
- 2개의 고정 VSCode 서비스 (training-develop-0, training-develop-1)
- **보장된 GPU 할당**: training-develop-0 → GPU 0, training-develop-1 → GPU 1
- 각 서비스별 포트 할당 (18080, 18081)
- MinIO traindata 버킷 통합 (별도 MinIO 서비스 필요)
- **자동 패키지 문서화**: pyproject.toml 복사 및 INSTALLED_PACKAGES.md 자동 생성
- 단일 사용자 접근 제한
- Python, Jupyter 개발 환경
