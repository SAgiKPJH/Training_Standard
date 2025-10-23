import os
import sys
import json
import importlib.util
import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Tuple, Optional

class ModelHandler:
    """
    시간 기반 동적 Model Handler (Dynamic Loading 방식) - Swin Transformer 버전
    
    폴더 구조:
    /swin_local_multifolder_one/
    ├── model_handler.py (이 파일)
    ├── 1/
    │   ├── model_handler.py
    │   ├── model.pth
    │   └── label.txt
    └── 2/
        ├── model_handler.py
        ├── model.pth
        └── label.txt
    
    __init__ 시점에는 초기화만 수행
    __call__ 시점에 현재 시간의 초 단위에 따라 적절한 폴더 선택하고 동적 로딩
    """
    
    def __init__(self, data: Any, context: Dict[str, str]):
        self.logger = logging.getLogger(__name__)
        
        # 📁 현재 파일의 디렉토리 경로
        self._current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 🗂️ 현재 로드된 Model Handler 인스턴스
        self._current_handler: Optional[Any] = None
        self._current_folder: Optional[str] = None
        
        # ⏱️ 성능 측정
        self._load_times: Dict[str, float] = {}
        
        # 📊 폴더 선택 및 로딩 통계
        self._folder_select_stats = {
            'total_selects': 0,
            'total_loads': 0,
            'folder_1_count': 0,
            'folder_2_count': 0,
            'folder_changes': 0,
            'total_time': 0.0,
            'total_load_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'avg_time': 0.0,
            'avg_load_time': 0.0
        }
        
        self.logger.info("🚀 Time-based Dynamic Swin Transformer ModelHandler - Ready for dynamic loading...")
        self.logger.info("✅ Initialization completed! Ready to load handlers on demand.")
    
    def _clear_current_handler(self):
        """현재 로드된 핸들러를 메모리에서 제거"""
        if self._current_handler is not None:
            self.logger.info(f"[HANDLER CLEAR] 기존 Swin 핸들러 ({self._current_folder}) 메모리에서 제거")
            
            # 핸들러의 모델 초기화 함수 호출 (모델 메모리 정리)
            try:
                if hasattr(self._current_handler, 'clear_model'):
                    self._current_handler.clear_model()
                elif hasattr(self._current_handler, '__del__'):
                    self._current_handler.__del__()
            except Exception as e:
                self.logger.warning(f"[HANDLER CLEAR] Swin 핸들러 정리 중 오류: {e}")
            
            # 핸들러 객체 삭제
            del self._current_handler
            self._current_handler = None
            self._current_folder = None
            
            self.logger.info(f"[HANDLER CLEAR] Swin 핸들러 제거 완료")
    
    def _load_handler_if_needed(self, target_folder: str, data: Any, context: Dict[str, str]):
        """필요한 경우에만 핸들러 로드 (폴더가 다르면 기존 핸들러 제거 후 새로 로드)"""
        # 동일한 폴더면 로딩 건너뛰기
        if self._current_folder == target_folder and self._current_handler is not None:
            self.logger.info(f"[HANDLER SKIP] 동일한 Swin 폴더 ({target_folder}) - 로딩 건너뛰기")
            return self._current_handler
        
        # 폴더가 다르면 기존 핸들러 제거 후 새로 로드
        if self._current_folder != target_folder:
            if self._current_folder is not None:
                self.logger.info(f"[HANDLER CHANGE] Swin 핸들러 변경 시작: {self._current_folder} -> {target_folder}")
                self._folder_select_stats['folder_changes'] += 1
            
            # 기존 핸들러 제거
            self._clear_current_handler()
            
            # 새 핸들러 로드
            return self._load_model_handler_from_folder(target_folder, data, context)
        
        return self._current_handler

    def _load_model_handler_from_folder(self, folder_name: str, data: Any, context: Dict[str, str]) -> Any:
        """선택된 폴더에서 실제 Swin ModelHandler 동적 로드 (각 폴더의 working directory에서 초기화)"""
        self.logger.info(f"[HANDLER LOAD] 새로운 Swin 핸들러 로드 시작: 폴더 {folder_name}")
        load_start = time.time()
        
        try:
            # 📁 경로 구성
            model_folder_path = os.path.join(self._current_dir, folder_name)
            model_handler_path = os.path.join(model_folder_path, "model_handler.py")
            
            if not os.path.exists(model_handler_path):
                raise FileNotFoundError(f"Swin Model handler not found: {model_handler_path}")
            
            self.logger.debug(f"📂 Loading Swin from: {model_handler_path}")
            
            # ⚡ 해당 폴더의 working directory에서 초기화 (중요!)
            with self._working_directory(model_folder_path):
                self.logger.debug(f"🔄 Working directory for Swin initialization: {os.getcwd()}")
                
                # 동적 모듈 로드
                module_name = f"swin_model_handler_{folder_name}_{int(time.time())}"
                spec = importlib.util.spec_from_file_location(module_name, model_handler_path)
                module = importlib.util.module_from_spec(spec)
                
                # 모듈 실행
                spec.loader.exec_module(module)
                
                # ModelHandler 인스턴스 생성 (해당 폴더의 working directory에서!)
                if hasattr(module, 'ModelHandler'):
                    self.logger.debug(f"🏗️ Creating Swin ModelHandler instance in folder: {folder_name}")
                    handler_instance = module.ModelHandler(data, context)
                    
                    # 현재 상태 업데이트
                    self._current_handler = handler_instance
                    self._current_folder = folder_name
                    
                    # 로드 시간 측정
                    load_end = time.time()
                    load_duration = load_end - load_start
                    self._load_times[folder_name] = load_duration
                    
                    # 통계 업데이트
                    self._folder_select_stats['total_loads'] += 1
                    self._folder_select_stats['total_load_time'] += load_duration
                    self._folder_select_stats['avg_load_time'] = self._folder_select_stats['total_load_time'] / self._folder_select_stats['total_loads']
                    
                    self.logger.info(f"✅ Swin ModelHandler from folder {folder_name} loaded in {load_duration:.4f}s")
                    return handler_instance
                else:
                    raise AttributeError(f"Swin ModelHandler class not found in {model_handler_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load swin model handler from folder {folder_name}: {e}")
            # 실패시 현재 핸들러 상태 초기화
            self._current_handler = None
            self._current_folder = None
            raise
    
    @contextmanager
    def _working_directory(self, path: str):
        """임시 작업 디렉토리 전환 컨텍스트 매니저"""
        original_cwd = os.getcwd()
        original_sys_path = sys.path.copy()
        
        try:
            os.chdir(path)
            sys.path.insert(0, path)
            yield path
        finally:
            # 환경 복원
            os.chdir(original_cwd)
            sys.path = original_sys_path
    
    def _determine_model_folder(self) -> str:
        """
        현재 시간의 초 단위를 기반으로 사용할 Swin 모델 폴더 결정
        
        로직:
        - 현재 시간의 초 % 2
        - 0이면 폴더 1
        - 1이면 폴더 2
        """
        current_time = time.time()
        seconds = int(current_time) % 60  # 현재 초 (0-59)
        folder_selector = seconds % 2
        
        if folder_selector == 0:
            selected_folder = "1"
        else:
            selected_folder = "2"
        
        self.logger.info(f"[SWIN FOLDER SELECT] 시간: {current_time:.3f}, 초: {seconds}, %2 = {folder_selector} -> 폴더 {selected_folder}")
        
        return selected_folder
    
    def __call__(self, data: Any, context: Dict[str, str]) -> Tuple[Any, Dict[str, str]]:
        """
        실제 Swin 추론 수행 - 시간에 따라 적절한 폴더 선택하고 동적으로 핸들러 로드
        """
        call_start = time.time()
        
        try:
            # 🎯 현재 시간 기반으로 폴더 결정
            select_start = time.time()
            required_folder = self._determine_model_folder()
            select_duration = time.time() - select_start
            
            # 📊 폴더 선택 통계 업데이트
            self._folder_select_stats['total_selects'] += 1
            self._folder_select_stats['total_time'] += select_duration
            self._folder_select_stats['min_time'] = min(self._folder_select_stats['min_time'], select_duration)
            self._folder_select_stats['max_time'] = max(self._folder_select_stats['max_time'], select_duration)
            self._folder_select_stats['avg_time'] = self._folder_select_stats['total_time'] / self._folder_select_stats['total_selects']
            
            if required_folder == "1":
                self._folder_select_stats['folder_1_count'] += 1
            else:
                self._folder_select_stats['folder_2_count'] += 1
            
            self.logger.info(f"🔍 Current Swin request requires model folder: {required_folder}")
            self.logger.info(f"📊 Swin folder selection stats: 1번={self._folder_select_stats['folder_1_count']}, "
                           f"2번={self._folder_select_stats['folder_2_count']}, "
                           f"변경={self._folder_select_stats['folder_changes']}회, "
                           f"평균선택시간={self._folder_select_stats['avg_time']:.6f}초")
            
            # 🎯 필요한 경우에만 핸들러 로드 (동적 로딩)
            load_start = time.time()
            actual_handler = self._load_handler_if_needed(required_folder, data, context)
            load_duration = time.time() - load_start
            
            if actual_handler is None:
                raise RuntimeError(f"Failed to load Swin handler for folder {required_folder}")
            
            # 🚀 로드된 적절한 handler로 추론 수행
            inference_start = time.time()
            result, output_context = actual_handler(data, context)
            inference_time = time.time() - inference_start
            
            total_call_time = time.time() - call_start
            
            # 📊 결과에 사용된 모델 정보 및 성능 정보 추가
            output_context["used_model_folder"] = required_folder
            output_context["used_model_type"] = "swin_transformer"
            output_context["time_based_selection"] = "true"
            output_context["dynamic_loading"] = "true"
            output_context["performance"] = json.dumps({
                "total_call_time": round(total_call_time, 4),
                "inference_time": round(inference_time, 4),
                "folder_select_time": round(select_duration, 6),
                "handler_load_time": round(load_duration, 4),
                "dynamic_loaded": True,
                "model_type": "swin_transformer"
            })
            output_context["folder_stats"] = json.dumps(self._folder_select_stats)
            
            self.logger.info(f"✅ Swin inference completed using dynamic model folder: {required_folder}")
            self.logger.info(f"⏱️ Swin call performance: total={total_call_time:.4f}s, inference={inference_time:.4f}s, "
                           f"select={select_duration:.6f}s, load={load_duration:.4f}s (dynamic)")
            
            return result, output_context
            
        except Exception as e:
            self.logger.error(f"❌ Swin inference failed: {e}")
            # 에러 정보도 context에 포함
            error_context = dict(context)
            error_context["error"] = str(e)
            error_context["used_model_folder"] = required_folder if 'required_folder' in locals() else "unknown"
            error_context["used_model_type"] = "swin_transformer"
            raise
    
    def test(self):
        """테스트 함수 - Swin 폴더 선택 테스트"""
        self.logger.info("[SWIN TEST] 시간 기반 폴더 선택 테스트 시작...")
        
        for i in range(10):
            selected_folder = self._determine_model_folder()
            self.logger.info(f"[SWIN TEST] {i+1}번째 테스트 - 선택된 폴더: {selected_folder}")
            time.sleep(1)  # 1초 대기로 초 변경 확인
        
        self.logger.info("[SWIN TEST] 폴더 선택 테스트 완료")
        self.logger.info(f"[SWIN TEST STATS] 폴더 선택 통계: {self._folder_select_stats}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 정보 반환"""
        return {
            "model_type": "swin_transformer",
            "current_handler_folder": self._current_folder,
            "load_times": self._load_times,
            "is_handler_loaded": self._current_handler is not None,
            "folder_select_stats": self._folder_select_stats
        }
    
    def reinitialize_current_model(self):
        """현재 로드된 Swin 핸들러의 모델을 재초기화"""
        if self._current_handler is not None and hasattr(self._current_handler, 'initialize_model'):
            self.logger.info(f"[SWIN MODEL REINIT] 현재 Swin 핸들러 ({self._current_folder}) 모델 재초기화")
            self._current_handler.initialize_model()
        else:
            self.logger.warning("[SWIN MODEL REINIT] 재초기화할 Swin 핸들러가 없거나 initialize_model 메서드가 없음")

    def __del__(self):
        """소멸자에서 현재 Swin 핸들러 정리"""
        try:
            self._clear_current_handler()
            self.logger.info("[CLEANUP] Dynamic MultiFolder Swin ModelHandler 정리 완료")
        except Exception as e:
            self.logger.error(f"[CLEANUP ERROR] Swin 정리 중 오류: {e}")

if __name__ == "__main__":
    import pickle
    import numpy as np
    
    # 더미 데이터 생성 (384x384 크기의 랜덤 이미지)
    dummy_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    data = pickle.dumps(dummy_image)
    
    # Swin MultiFolder Test 실행
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    try:
        handler = ModelHandler(None, {})
        
        # 폴더 선택 테스트
        handler.test()
        
        # 여러 번 실행하여 동적 폴더 선택 테스트
        for i in range(5):
            logger.info(f"\n[SWIN TEST RUN] {i+1}번째 추론 테스트 실행")
            result, context = handler(data=data, context={})
            logger.info(f"[SWIN TEST RESULT] 사용된 폴더: {context['used_model_folder']}")
            logger.info(f"[SWIN TEST PERFORMANCE] {context['performance']}")
            time.sleep(2)  # 2초 대기로 폴더 변경 확인
            
    except Exception as e:
        logger.error(f"[SWIN TEST ERROR] 테스트 실행 중 오류: {e}")
        logger.info("💡 1번, 2번 폴더와 각각의 model_handler.py, model.pth, label.txt 파일이 필요합니다.")