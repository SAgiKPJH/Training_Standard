import tensorflow as tf
import onnx
import onnxruntime as ort
import numpy as np
import os
import argparse
from onnx2tf import convert

def onnx_to_tensorflow(onnx_path, tf_output_dir):
    """
    ONNX 모델을 TensorFlow로 변환
    
    Args:
        onnx_path: ONNX 모델 파일 경로
        tf_output_dir: TensorFlow 모델을 저장할 디렉토리
        
    Returns:
        변환 성공 여부
    """
    
    # ONNX 모델 로드
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 검증 완료")
    
    try:
        # onnx2tf를 사용하여 변환
        convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tf_output_dir,
            copy_onnx_input_output_names_to_tf=True,
            non_verbose=True
        )
        print(f"ONNX 모델이 TensorFlow로 변환되어 {tf_output_dir}에 저장되었습니다.")
        return True
    except Exception as e:
        print(f"변환 중 오류 발생: {e}")
        return False

def verify_conversion(onnx_path, tf_output_dir, input_shape=(1, 299, 299, 3)):
    """
    변환된 TensorFlow 모델 검증
    
    Args:
        onnx_path: 원본 ONNX 모델 파일 경로
        tf_output_dir: TensorFlow 모델이 저장된 디렉토리
        input_shape: 입력 텐서의 형태
    """
    
    # ONNX Runtime으로 추론
    ort_session = ort.InferenceSession(onnx_path)
    
    # TensorFlow 모델 로드
    try:
        tf_model = tf.keras.models.load_model(tf_output_dir)
        print("TensorFlow 모델 로드 완료")
    except Exception as e:
        print(f"TensorFlow 모델 로드 실패: {e}")
        return
    
    # 더미 입력 생성
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # ONNX 추론
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # TensorFlow 추론
    tf_output = tf_model.predict(dummy_input, verbose=0)
    
    # 출력 비교
    np.testing.assert_allclose(
        ort_output, tf_output, rtol=1e-05, atol=1e-05
    )
    print("ONNX와 TensorFlow 출력이 일치합니다.")

def main():
    parser = argparse.ArgumentParser(description='ONNX 모델을 TensorFlow로 변환')
    parser.add_argument('--onnx_path', type=str, required=True,
                       help='ONNX 모델 파일 경로')
    parser.add_argument('--tf_output_dir', type=str, default='tf_model',
                       help='TensorFlow 모델을 저장할 디렉토리')
    parser.add_argument('--input_shape', type=str, default='1,299,299,3',
                       help='입력 텐서 형태 (쉼표로 구분)')
    parser.add_argument('--verify', action='store_true',
                       help='변환 후 모델 검증 수행')
    
    args = parser.parse_args()
    
    # 입력 형태 파싱
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # 파일 존재 확인
    if not os.path.exists(args.onnx_path):
        print(f"ONNX 모델 파일을 찾을 수 없습니다: {args.onnx_path}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(args.tf_output_dir, exist_ok=True)
    
    # ONNX를 TensorFlow로 변환
    success = onnx_to_tensorflow(args.onnx_path, args.tf_output_dir)
    
    # 검증 수행
    if success and args.verify:
        verify_conversion(args.onnx_path, args.tf_output_dir, input_shape)

if __name__ == "__main__":
    main()

