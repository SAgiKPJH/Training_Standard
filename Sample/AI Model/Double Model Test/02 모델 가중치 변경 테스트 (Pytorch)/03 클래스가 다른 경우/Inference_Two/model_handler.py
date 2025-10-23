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
    MasterData 기반 동적 Model Handler (Pre-loading 방식)
    
    폴더 구조:
    /model_handler/
    ├── model_handler.py (이 파일)
    ├── 1/
    │   └── model_handler.py
    └── 2/
        └── model_handler.py
    
    __init__ 시점에 모든 폴더의 model_handler를 미리 로드
    __call__ 시점에는 context의 기준정보에 따라 적절한 인스턴스 선택
    """
    
    def __init__(self, data: Any, context: Dict[str, str]):
        self.logger = logging.getLogger(__name__)
        
        # 📁 현재 파일의 디렉토리 경로
        self._current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 🗂️ 미리 로드된 Model Handler 인스턴스들
        self._preloaded_handlers: Dict[str, Any] = {}
        
        # ⏱️ 성능 측정
        self._load_times: Dict[str, float] = {}
        
        self.logger.info("🚀 Pre-loading Dynamic ModelHandler - Loading all sub-handlers...")
        
        # 🔥 모든 폴더의 model_handler를 미리 로드
        self._preload_all_handlers(data, context)
        
        self.logger.info(f"✅ Pre-loading completed! Loaded handlers: {list(self._preloaded_handlers.keys())}")
    
    def _preload_all_handlers(self, data: Any, context: Dict[str, str]):
        """모든 폴더의 model_handler를 미리 로드"""
        folders_to_load = ["1", "2"]  # 필요에 따라 더 추가 가능
        
        for folder_name in folders_to_load:
            try:
                load_start = time.time()
                self.logger.info(f"📥 Pre-loading handler from folder: {folder_name}")
                
                handler_instance = self._load_model_handler_from_folder(folder_name, data, context)
                self._preloaded_handlers[folder_name] = handler_instance
                
                load_time = time.time() - load_start
                self._load_times[folder_name] = load_time
                
                self.logger.info(f"✅ Handler from folder {folder_name} loaded in {load_time:.4f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to preload handler from folder {folder_name}: {e}")
                # 실패해도 계속 진행 (다른 폴더들은 로드)
    
    def _load_model_handler_from_folder(self, folder_name: str, data: Any, context: Dict[str, str]) -> Any:
        """선택된 폴더에서 실제 ModelHandler 동적 로드 (각 폴더의 working directory에서 초기화)"""
        try:
            # 📁 경로 구성
            model_folder_path = os.path.join(self._current_dir, folder_name)
            model_handler_path = os.path.join(model_folder_path, "model_handler.py")
            
            if not os.path.exists(model_handler_path):
                raise FileNotFoundError(f"Model handler not found: {model_handler_path}")
            
            self.logger.debug(f"📂 Loading from: {model_handler_path}")
            
            # ⚡ 해당 폴더의 working directory에서 초기화 (중요!)
            with self._working_directory(model_folder_path):
                self.logger.debug(f"🔄 Working directory for initialization: {os.getcwd()}")
                
                # 동적 모듈 로드
                module_name = f"model_handler_{folder_name}_{int(time.time())}"
                spec = importlib.util.spec_from_file_location(module_name, model_handler_path)
                module = importlib.util.module_from_spec(spec)
                
                # 모듈 실행
                spec.loader.exec_module(module)
                
                # ModelHandler 인스턴스 생성 (해당 폴더의 working directory에서!)
                if hasattr(module, 'ModelHandler'):
                    self.logger.debug(f"🏗️ Creating ModelHandler instance in folder: {folder_name}")
                    handler_instance = module.ModelHandler(data, context)
                    
                    self.logger.info(f"✅ ModelHandler from folder {folder_name} initialized successfully")
                    return handler_instance
                else:
                    raise AttributeError(f"ModelHandler class not found in {model_handler_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load model handler from folder {folder_name}: {e}")
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
    
    def _extract_master_data(self, context: Dict[str, str]) -> Dict[str, Any]:
        """Context에서 MasterData 필드들만 추출 (flat structure)"""
        # Master Data 관련 필드 목록
        master_data_fields = {
            'equip_id', 'device_id', 'line_id', 
            'step_id', 'setup_id'
        }
        try:
            # gRPC ScalarMapContainer인 경우 dict로 변환
            if hasattr(context, '__iter__') and hasattr(context, 'items'):
                context_dict = dict(context)
            else:
                context_dict = context
            
            # Master Data 필드들만 추출
            extracted_master_data = {}
            for key, value in context_dict.items():
                if key in master_data_fields:
                    extracted_master_data[key] = value
            
            self.logger.info(f"🔍 Extracted master_data from context: {extracted_master_data}")
            self.logger.debug(f"📋 Full context keys: {list(context_dict.keys())}")
            
            return extracted_master_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract master_data from context: {e}")
            return {}
    
    def _determine_model_folder(self, master_data: Dict[str, Any]) -> str:
        """
        MasterData를 기반으로 사용할 모델 폴더 결정
        
        예시 로직:
        - setup_id가 'SETUP_A'이면 폴더 1
        - setup_id가 'SETUP_B'이면 폴더 2
        - step_id가 특정 값이면 특정 폴더
        - 기본값은 폴더 1
        """
        setup_id = master_data.get("setup_id", "")
        step_id = master_data.get("step_id", "")
        line_id = master_data.get("line_id", "")
        
        # 🔍 여기서 비즈니스 로직에 따라 폴더 선택
        if setup_id in ["SETUP_A", "SETUP_001"]:
            return "1"
        elif setup_id in ["SETUP_B", "SETUP_002"]:
            return "2"
        elif step_id.startswith("STEP_HIGH"):
            return "2"
        elif line_id == "LINE2":
            return "2"
        else:
            # 기본값
            return "1"
    
    def __call__(self, data: Any, context: Dict[str, str]) -> Tuple[Any, Dict[str, str]]:
        """
        실제 추론 수행 - 미리 로드된 handler 중에서 context에 따라 적절한 것 선택
        """
        call_start = time.time()
        
        try:
            # 🎯 현재 요청의 MasterData 추출
            current_master_data = self._extract_master_data(context)
            
            # 📁 현재 요청에 적합한 모델 폴더 결정
            required_folder = self._determine_model_folder(current_master_data)
            
            self.logger.info(f"🔍 Current request requires model folder: {required_folder}")
            self.logger.info(f"📋 Current MasterData: {current_master_data}")
            
            # 🎯 미리 로드된 handler 확인
            if required_folder not in self._preloaded_handlers:
                raise ValueError(f"Handler for folder {required_folder} was not preloaded. Available: {list(self._preloaded_handlers.keys())}")
            
            # 🚀 미리 로드된 적절한 handler로 추론 수행 (매우 빠름!)
            inference_start = time.time()
            actual_handler = self._preloaded_handlers[required_folder]
            result, output_context = actual_handler(data, context)
            inference_time = time.time() - inference_start
            
            total_call_time = time.time() - call_start
            
            # 📊 결과에 사용된 모델 정보 및 성능 정보 추가
            output_context["used_model_folder"] = required_folder
            output_context["master_data_info"] = json.dumps(current_master_data)
            output_context["preloaded"] = "true"
            output_context["performance"] = json.dumps({
                "total_call_time": round(total_call_time, 4),
                "inference_time": round(inference_time, 4),
                "preload_time": round(self._load_times.get(required_folder, 0), 4),
                "preloaded": True
            })
            
            self.logger.info(f"✅ Inference completed using preloaded model folder: {required_folder}")
            self.logger.info(f"⏱️ Call performance: total={total_call_time:.4f}s, inference={inference_time:.4f}s (preloaded)")
            
            self.logger.info(f"🔍 Current context: {context}")

            return result, output_context
            
        except Exception as e:
            self.logger.error(f"❌ Inference failed: {e}")
            # 에러 정보도 context에 포함
            error_context = dict(context)
            error_context["error"] = str(e)
            error_context["used_model_folder"] = required_folder if 'required_folder' in locals() else "unknown"
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 정보 반환"""
        return {
            "preloaded_handlers": list(self._preloaded_handlers.keys()),
            "preload_times": self._load_times,
            "handler_count": len(self._preloaded_handlers)
        }
    
    def add_handler_folder(self, folder_name: str, data: Any, context: Dict[str, str]):
        """런타임에 새로운 폴더의 handler 추가 (선택적)"""
        try:
            self.logger.info(f"📥 Adding new handler from folder: {folder_name}")
            handler_instance = self._load_model_handler_from_folder(folder_name, data, context)
            self._preloaded_handlers[folder_name] = handler_instance
            self.logger.info(f"✅ Handler from folder {folder_name} added successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to add handler from folder {folder_name}: {e}")
            raise
