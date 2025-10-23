import os, sys, shutil, subprocess, importlib.util as iu

def run(cmd):
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return cp.returncode, (cp.stdout or "").strip()
    except Exception as e:
        return 1, str(e)

summary = {
    "env": {"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH","")},
    "gpu": {}, "torch": {}, "tensorflow": {}, "opencv": {}, "tesseract": {}
}
exit_fail = False

# 1) GPU(NVIDIA) 접근성
smipath = shutil.which("nvidia-smi")
if not smipath:
    summary["gpu"] = {"status":"FAIL","reason":"nvidia-smi not found"}
    exit_fail = True
else:
    rc, out = run([smipath, "-L"])
    summary["gpu"] = {"status":"OK" if rc==0 else "FAIL", "output": out.splitlines()[:5]}
    if rc != 0:
        exit_fail = True

# 2) PyTorch
try:
    has_torch = iu.find_spec("torch") is not None
    if has_torch:
        import torch
        info = {
            "status":"OK",
            "version": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_available": bool(torch.backends.cudnn.is_available())
        }
        info["cudnn_version"] = torch.backends.cudnn.version() if info["cudnn_available"] else None
        if info["cuda_available"]:
            try:
                a = torch.randn(1024,1024, device="cuda")
                b = torch.randn(1024,1024, device="cuda")
                c = torch.mm(a,b)
                torch.cuda.synchronize()
                info["gpu_matmul"] = "OK"
            except Exception as e:
                info["gpu_matmul"] = f"FAIL: {e}"
                exit_fail = True
        else:
            info["status"] = "FAIL"
            exit_fail = True
        summary["torch"] = info
    else:
        summary["torch"] = {"status":"SKIP","reason":"not installed"}
except Exception as e:
    summary["torch"] = {"status":"FAIL","error":str(e)}
    exit_fail = True

# 3) TensorFlow
try:
    has_tf = iu.find_spec("tensorflow") is not None
    if has_tf:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info = {
            "status":"OK",
            "version": tf.__version__,
            "built_with_cuda": bool(tf.test.is_built_with_cuda()),
            "gpus": [g.name for g in gpus]
        }
        if gpus:
            try:
                with tf.device("/GPU:0"):
                    a, b = tf.random.normal([1024,1024]), tf.random.normal([1024,1024])
                    c = tf.matmul(a,b); _ = c.numpy()
                info["gpu_matmul"] = "OK"
            except Exception as e:
                info["gpu_matmul"] = f"FAIL: {e}"
                exit_fail = True
        else:
            info["status"] = "FAIL"
            exit_fail = True
        summary["tensorflow"] = info
    else:
        summary["tensorflow"] = {"status":"SKIP","reason":"not installed"}
except Exception as e:
    summary["tensorflow"] = {"status":"FAIL","error":str(e)}
    exit_fail = True

# 4) OpenCV
try:
    has_cv = iu.find_spec("cv2") is not None
    if has_cv:
        import cv2
        cnt = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2,"cuda") else None
        summary["opencv"] = {"status":"OK","version":cv2.__version__, "cuda_device_count":cnt}
    else:
        summary["opencv"] = {"status":"SKIP","reason":"not installed"}
except Exception as e:
    summary["opencv"] = {"status":"FAIL","error":str(e)}

# 5) Tesseract
tpath = shutil.which("tesseract")
if tpath:
    rc, out = run([tpath, "--version"])
    summary["tesseract"] = {"status":"OK" if rc==0 else "FAIL", "output": (out.splitlines() or [""])[0]}
    if rc != 0:
        exit_fail = True
else:
    summary["tesseract"] = {"status":"SKIP","reason":"not installed"}

# 출력 & 종료 코드
print("=== Smoke Test Summary ===")
for k, v in summary.items():
    print(f"{k}: {v}")

# 둘 다 미설치면 경고만 하고 실패로 치지 않음
if summary["torch"].get("status")=="SKIP" and summary["tensorflow"].get("status")=="SKIP":
    exit_fail = False

sys.exit(1 if exit_fail else 0)