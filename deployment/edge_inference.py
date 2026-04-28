"""
Edge Device Inference
Run inference on edge devices (Raspberry Pi, Jetson Nano, mobile)
"""

import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
import threading
import queue


class EdgeInference:
    """Run inference on edge devices"""
    
    def __init__(self, model_path: str, device_type: str = 'cpu'):
        """
        Initialize edge inference
        
        Args:
            model_path: Path to model (TFLite or ONNX)
            device_type: 'cpu', 'gpu', 'npu', 'tpu'
        """
        self.model_path = model_path
        self.device_type = device_type
        self.model = None
        self.interpreter = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model based on format and device"""
        if self.model_path.endswith('.tflite'):
            self._load_tflite()
        elif self.model_path.endswith('.onnx'):
            self._load_onnx()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")
    
    def _load_tflite(self):
        """Load TFLite model"""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            print(f"TFLite model loaded from {self.model_path}")
        except ImportError:
            print("TensorFlow not installed. Install with: pip install tensorflow")
    
    def _load_onnx(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
            
            providers = ['CPUExecutionProvider']
            if self.device_type == 'gpu':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif self.device_type == 'npu':
                providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
            
            self.model = ort.InferenceSession(self.model_path, providers=providers)
            print(f"ONNX model loaded from {self.model_path}")
        except ImportError:
            print("ONNX Runtime not installed. Install with: pip install onnxruntime")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference
        
        Args:
            input_data: Input tensor
            
        Returns:
            Inference output
        """
        if self.interpreter is not None:
            return self._infer_tflite(input_data)
        elif self.model is not None:
            return self._infer_onnx(input_data)
        else:
            raise RuntimeError("No model loaded")
    
    def _infer_tflite(self, input_data: np.ndarray) -> np.ndarray:
        """TFLite inference"""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        output = self.interpreter.get_tensor(output_details[0]['index'])
        return output
    
    def _infer_onnx(self, input_data: np.ndarray) -> np.ndarray:
        """ONNX Runtime inference"""
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        
        result = self.model.run([output_name], {input_name: input_data})
        return result[0]
    
    def benchmark(self, 
                 input_shape: Tuple = (1, 3, 640, 640),
                 num_iterations: int = 100,
                 warmup: int = 10) -> Dict:
        """
        Benchmark inference performance
        
        Args:
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            warmup: Warmup iterations
            
        Returns:
            Benchmark results
        """
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            _ = self.infer(dummy_input)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.infer(dummy_input)
            end = time.time()
            times.append((end - start) * 1000)  # ms
        
        times = np.array(times)
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'median_latency_ms': np.median(times),
            'fps': 1000 / np.mean(times),
        }


class RaspberryPiInference(EdgeInference):
    """Optimized inference for Raspberry Pi"""
    
    def __init__(self, model_path: str, use_gpu: bool = False):
        """
        Initialize RPi inference
        
        Args:
            model_path: Path to model
            use_gpu: Use GPU acceleration (Coral TPU)
        """
        super().__init__(model_path, device_type='cpu')
        self.use_gpu = use_gpu
        
        if use_gpu:
            self._setup_coral_tpu()
    
    def _setup_coral_tpu(self):
        """Setup Coral TPU accelerator"""
        try:
            from pycoral.pybind import _pywrap_tensorflow_lite_python_interpreter_wrapper
            print("Coral TPU detected and ready")
        except ImportError:
            print("Coral TPU not detected. Install with: pip install pycoral")


class JetsonInference(EdgeInference):
    """Optimized inference for NVIDIA Jetson"""
    
    def __init__(self, model_path: str, use_tensorrt: bool = True):
        """
        Initialize Jetson inference
        
        Args:
            model_path: Path to model
            use_tensorrt: Use TensorRT optimization
        """
        device_type = 'gpu' if use_tensorrt else 'cpu'
        super().__init__(model_path, device_type=device_type)
        self.use_tensorrt = use_tensorrt
        
        if use_tensorrt:
            self._setup_tensorrt()
    
    def _setup_tensorrt(self):
        """Setup TensorRT optimization"""
        try:
            import tensorrt
            print(f"TensorRT {tensorrt.__version__} detected")
        except ImportError:
            print("TensorRT not detected")


class MobileInference(EdgeInference):
    """Inference for mobile devices (Android/iOS)"""
    
    def __init__(self, model_path: str, model_type: str = 'tflite'):
        """
        Initialize mobile inference
        
        Args:
            model_path: Path to model
            model_type: 'tflite' or 'coreml'
        """
        super().__init__(model_path, device_type='cpu')
        self.model_type = model_type
    
    def export_for_android(self, output_dir: str):
        """Export model for Android"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model_path.endswith('.tflite'):
            # Copy TFLite model
            import shutil
            shutil.copy(self.model_path, output_dir / 'model.tflite')
            print(f"Model exported for Android to {output_dir}")
        else:
            print("Convert to TFLite format for Android deployment")
    
    def export_for_ios(self, output_dir: str):
        """Export model for iOS"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("iOS deployment requires CoreML format")
        print("Convert using: coremltools.convert()")


class StreamingInference:
    """Real-time inference from camera stream"""
    
    def __init__(self, model_path: str, camera_id: int = 0):
        """
        Initialize streaming inference
        
        Args:
            model_path: Path to model
            camera_id: Camera device ID
        """
        self.inference = EdgeInference(model_path)
        self.camera_id = camera_id
        self.running = False
        self.result_queue = queue.Queue(maxsize=1)
    
    def start_stream(self, confidence_threshold: float = 0.5):
        """
        Start inference stream
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        try:
            import cv2
        except ImportError:
            print("OpenCV not installed. Install with: pip install opencv-python")
            return
        
        self.running = True
        
        # Start inference thread
        inference_thread = threading.Thread(
            target=self._inference_loop,
            args=(confidence_threshold,)
        )
        inference_thread.daemon = True
        inference_thread.start()
        
        # Start display loop
        self._display_loop()
    
    def _inference_loop(self, confidence_threshold: float):
        """Inference loop running in separate thread"""
        import cv2
        
        cap = cv2.VideoCapture(self.camera_id)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            input_data = self._preprocess(frame)
            
            # Run inference
            start = time.time()
            output = self.inference.infer(input_data)
            latency = (time.time() - start) * 1000
            
            # Postprocess results
            detections = self._postprocess(output, confidence_threshold)
            
            # Store result
            try:
                self.result_queue.put_nowait({
                    'frame': frame,
                    'detections': detections,
                    'latency_ms': latency,
                })
            except queue.Full:
                pass
        
        cap.release()
    
    def _display_loop(self):
        """Display inference results"""
        import cv2
        
        while self.running:
            try:
                result = self.result_queue.get(timeout=1)
                frame = result['frame']
                detections = result['detections']
                latency = result['latency_ms']
                
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    confidence = det['confidence']
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{confidence:.2f}', (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Display FPS
                fps = 1000 / latency if latency > 0 else 0
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Edge Inference', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    
            except queue.Empty:
                continue
        
        cv2.destroyAllWindows()
    
    def _preprocess(self, frame):
        """Preprocess video frame"""
        import cv2
        
        # Resize to model input size
        resized = cv2.resize(frame, (640, 640))
        
        # Normalize
        normalized = resized / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0).astype(np.float32)
        
        return input_data
    
    def _postprocess(self, output, confidence_threshold: float) -> List[Dict]:
        """Postprocess model output"""
        # This is a placeholder implementation
        # Adapt based on your model's output format
        
        detections = []
        
        # Example: output shape (1, 25200, 85) for YOLOv8
        if len(output.shape) == 3:
            predictions = output[0]
            
            for pred in predictions:
                conf = pred[4]  # Confidence score
                
                if conf > confidence_threshold:
                    x, y, w, h = pred[:4]
                    x1, y1 = x - w/2, y - h/2
                    x2, y2 = x + w/2, y + h/2
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'confidence': float(conf),
                    })
        
        return detections
    
    def stop_stream(self):
        """Stop inference stream"""
        self.running = False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run edge device inference')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--device', choices=['cpu', 'raspberry_pi', 'jetson', 'mobile'],
                       default='cpu', help='Target device')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarking')
    parser.add_argument('--stream', action='store_true', help='Run camera stream')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    
    args = parser.parse_args()
    
    if args.device == 'raspberry_pi':
        inference = RaspberryPiInference(args.model, use_gpu=False)
    elif args.device == 'jetson':
        inference = JetsonInference(args.model, use_tensorrt=True)
    elif args.device == 'mobile':
        inference = MobileInference(args.model)
    else:
        inference = EdgeInference(args.model)
    
    if args.benchmark:
        results = inference.benchmark()
        print("\nBenchmark Results:")
        print(f"  Mean Latency: {results['mean_latency_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.1f}")
    
    if args.stream:
        streaming = StreamingInference(args.model, args.camera_id)
        streaming.start_stream()
    
    print("Inference complete!")
