$IMAGE     = "mirero/daq-inference-core:ubuntu20.04-python3.10-cuda11.8-cudnn8-torch2.7-ocr"
$WORKSPACE = $PSScriptRoot

# data/ 폴더에 모델이 없으면 안내 후 종료
$dataDir = Join-Path $WORKSPACE "data"
if (-not (Test-Path $dataDir) -or (Get-ChildItem $dataDir -ErrorAction SilentlyContinue).Count -eq 0) {
    Write-Host "[오류] data/ 폴더에 모델이 없습니다. 먼저 download.py를 실행하세요:" -ForegroundColor Red
    Write-Host "  python download.py" -ForegroundColor Yellow
    exit 1
}

docker run --rm --gpus all `
    -v "${WORKSPACE}:/workspace" `
    -w /workspace `
    $IMAGE `
    bash /workspace/run.sh
