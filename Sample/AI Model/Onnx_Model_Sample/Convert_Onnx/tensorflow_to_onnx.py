import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort
import numpy as np
import os
import argparse

def create_sample_model():
    """샘플 TensorFlow 모델 생성 (실제 사용시에는 본인의 모델로 교체)"""
    model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(299, 299, 3)
    )
    
    # 분류 헤드 추가
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(9, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=model.input, outputs=output)
    return model

def tensorflow_to_onnx(tf_model, onnx_path, input_shape=(1, 299, 299, 3), 
                       opset_version=11):
    """
    TensorFlow 모델을 ONNX로 변환
    
    Args:
        tf_model: TensorFlow 모델
        onnx_path: 저장할 ONNX 파일 경로
        input_shape: 입력 텐서의 형태
        opset_version: ONNX opset 버전
    """
    
    # 모델을 저장 모드로 컴파일
    tf_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # tf2onnx를 사용하여 변환
    onnx_model, _ = tf2onnx.convert.from_keras(
        tf_model, 
        opset=opset_version,
        input_signature=(tf.TensorSpec(input_shape, tf.float32),)
    )
    
    # ONNX 모델 저장
    onnx.save(onnx_model, onnx_path)
    print(f"TensorFlow 모델이 {onnx_path}로 변환되었습니다.")

def verify_onnx_model(onnx_path, tf_model, input_shape=(1, 299, 299, 3)):
    """
    변환된 ONNX 모델 검증
    
    Args:
        onnx_path: ONNX 모델 파일 경로
        tf_model: 원본 TensorFlow 모델
        input_shape: 입력 텐서의 형태
    """
    
    # ONNX 모델 로드
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX 모델 검증 완료")
    
    # ONNX Runtime으로 추론
    ort_session = ort.InferenceSession(onnx_path)
    
    # 더미 입력 생성
    dummy_input = np.random.random(input_shape).astype(np.float32)
    
    # TensorFlow 추론
    tf_output = tf_model.predict(dummy_input, verbose=0)
    
    # ONNX 추론
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # 출력 비교
    np.testing.assert_allclose(
        ort_output, tf_output, rtol=1e-05, atol=1e-05
    )
    print("TensorFlow와 ONNX 출력이 일치합니다.")

def load_tensorflow_model(model_path):
    """
    TensorFlow 모델 로드
    
    Args:
        model_path: 모델 파일 경로 또는 디렉토리
        
    Returns:
        TensorFlow 모델
    """
    if os.path.isdir(model_path):
        # SavedModel 형식
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"SavedModel을 로드했습니다: {model_path}")
            return model
        except Exception as e:
            print(f"SavedModel 로드 실패: {e}")
            return None
    elif model_path.endswith('.h5') or model_path.endswith('.keras'):
        # HDF5/Keras 형식
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Keras 모델을 로드했습니다: {model_path}")
            return model
        except Exception as e:
            print(f"Keras 모델 로드 실패: {e}")
            return None
    else:
        print(f"지원되지 않는 모델 형식: {model_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description='TensorFlow 모델을 ONNX로 변환')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='TensorFlow 모델 파일 경로 또는 디렉토리')
    parser.add_argument('--onnx_path', type=str, default='model.onnx',
                       help='저장할 ONNX 파일 경로')
    parser.add_argument('--input_shape', type=str, default='1,299,299,3',
                       help='입력 텐서 형태 (쉼표로 구분)')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset 버전')
    parser.add_argument('--verify', action='store_true',
                       help='변환 후 모델 검증 수행')
    
    args = parser.parse_args()
    
    # 입력 형태 파싱
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # 모델 로드 또는 생성
    if args.model_path and os.path.exists(args.model_path):
        model = load_tensorflow_model(args.model_path)
        if model is None:
            print("샘플 TensorFlow 모델을 생성합니다.")
            model = create_sample_model()
    else:
        print("샘플 TensorFlow 모델을 생성합니다.")
        model = create_sample_model()
    
    # ONNX로 변환
    tensorflow_to_onnx(model, args.onnx_path, input_shape, opset_version=args.opset_version)
    
    # 검증 수행
    if args.verify:
        verify_onnx_model(args.onnx_path, model, input_shape)

if __name__ == "__main__":
    main()

