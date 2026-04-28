"""
Convert PyTorch models to TensorFlow Lite for edge deployment
"""

import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import subprocess
from typing import Dict
import onnx
import onnxruntime as ort
import numpy as np


class TFLiteConverter:
    """Convert PyTorch models to TensorFlow Lite format"""
    
    @staticmethod
    def pytorch_to_onnx(model_path: str, 
                       output_path: str,
                       input_shape: tuple = (1, 3, 640, 640)) -> str:
        """
        Convert PyTorch model to ONNX format
        
        Args:
            model_path: Path to PyTorch model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            
        Returns:
            Path to ONNX model
        """
        print(f"Converting PyTorch model to ONNX...")
        
        # Load model
        model = torch.jit.load(model_path) if model_path.endswith('.pt') else torch.load(model_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=12,
            do_constant_folding=True,
            verbose=False,
        )
        
        print(f"ONNX model saved to {output_path}")
        return output_path
    
    @staticmethod
    def onnx_to_tflite(onnx_path: str,
                      output_path: str,
                      quantization: bool = True,
                      quantization_type: str = 'int8') -> str:
        """
        Convert ONNX to TensorFlow Lite
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TFLite model
            quantization: Whether to apply quantization
            quantization_type: 'int8', 'float16', or 'dynamic'
            
        Returns:
            Path to TFLite model
        """
        try:
            import tensorflow as tf
            from onnx_tf.backend import prepare
        except ImportError:
            print("Install required packages: pip install tensorflow onnx-tf")
            return None
        
        print(f"Converting ONNX to TensorFlow Lite with {quantization_type} quantization...")
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        with tempfile.TemporaryDirectory() as temp_dir:
            saved_model_path = Path(temp_dir) / 'saved_model'
            tf_rep.export_graph(str(saved_model_path))
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            
            if quantization:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                if quantization_type == 'int8':
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                    ]
                    converter.inference_input_type = tf.uint8
                    converter.inference_output_type = tf.uint8
                elif quantization_type == 'float16':
                    converter.target_spec.supported_types = [tf.float16]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TFLite model saved to {output_path}")
            
            return output_path
    
    @staticmethod
    def validate_tflite(tflite_path: str, 
                       input_shape: tuple = (1, 3, 640, 640)) -> Dict:
        """
        Validate TFLite model
        
        Args:
            tflite_path: Path to TFLite model
            input_shape: Input shape
            
        Returns:
            Validation results
        """
        try:
            import tensorflow as tf
        except ImportError:
            return {'error': 'TensorFlow not installed'}
        
        print(f"Validating TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test inference
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return {
            'valid': True,
            'input_shape': input_details[0]['shape'].tolist(),
            'output_shape': output_details[0]['shape'].tolist(),
            'model_size_mb': Path(tflite_path).stat().st_size / (1024 * 1024),
        }


class ONNXExporter:
    """Export models to ONNX format"""
    
    @staticmethod
    def export_to_onnx(model_path: str,
                      output_path: str,
                      input_shape: tuple = (1, 3, 640, 640),
                      opset_version: int = 12) -> str:
        """
        Export model to ONNX format
        
        Args:
            model_path: Path to model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            
        Returns:
            Path to ONNX model
        """
        print(f"Exporting model to ONNX (opset {opset_version})...")
        
        # Load model
        model = torch.load(model_path)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['images'],
            output_names=['output'],
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False,
        )
        
        print(f"ONNX model saved to {output_path}")
        return output_path
    
    @staticmethod
    def validate_onnx(onnx_path: str) -> Dict:
        """Validate ONNX model"""
        print(f"Validating ONNX model...")
        
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        return {
            'valid': True,
            'model_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024),
        }
    
    @staticmethod
    def run_onnx_inference(onnx_path: str,
                          input_data: np.ndarray) -> np.ndarray:
        """Run inference using ONNX Runtime"""
        sess = ort.InferenceSession(onnx_path)
        
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        
        result = sess.run([output_name], {input_name: input_data})
        
        return result[0]


class ModelOptimizer:
    """Optimize models for edge deployment"""
    
    @staticmethod
    def optimize_for_mobile(model_path: str,
                           output_path: str) -> str:
        """
        Optimize PyTorch model for mobile deployment
        
        Args:
            model_path: Path to PyTorch model
            output_path: Path to save optimized model
            
        Returns:
            Path to optimized model
        """
        print("Optimizing model for mobile deployment...")
        
        model = torch.load(model_path)
        model.eval()
        
        # Convert to TorchScript
        scripted_model = torch.jit.script(model)
        
        # Optimize TorchScript
        optimized_model = torch.jit.optimize_for_mobile(scripted_model)
        
        # Save optimized model
        optimized_model._save_for_lite_interpreter(output_path)
        
        print(f"Optimized model saved to {output_path}")
        return output_path
    
    @staticmethod
    def compare_model_sizes(original_path: str, optimized_path: str) -> Dict:
        """Compare model sizes before and after optimization"""
        original_size = Path(original_path).stat().st_size / (1024 * 1024)
        optimized_size = Path(optimized_path).stat().st_size / (1024 * 1024)
        
        compression = (1 - optimized_size / original_size) * 100
        
        return {
            'original_size_mb': original_size,
            'optimized_size_mb': optimized_size,
            'compression_percentage': compression,
        }


class DeploymentPackager:
    """Package models for edge device deployment"""
    
    @staticmethod
    def create_deployment_bundle(model_path: str,
                                output_dir: str,
                                include_onnx: bool = True,
                                include_tflite: bool = True) -> Dict:
        """
        Create deployment package with multiple formats
        
        Args:
            model_path: Path to model
            output_dir: Output directory
            include_onnx: Include ONNX format
            include_tflite: Include TFLite format
            
        Returns:
            Deployment package info
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        package_info = {
            'timestamp': str(Path(model_path).stat().st_mtime),
            'formats': {}
        }
        
        # ONNX export
        if include_onnx:
            onnx_path = output_dir / 'model.onnx'
            ONNXExporter.export_to_onnx(model_path, str(onnx_path))
            package_info['formats']['onnx'] = {
                'path': str(onnx_path),
                'size_mb': onnx_path.stat().st_size / (1024 * 1024),
            }
        
        # TFLite export
        if include_tflite:
            tflite_path = output_dir / 'model.tflite'
            onnx_temp = output_dir / 'model_temp.onnx'
            
            if not (output_dir / 'model.onnx').exists():
                ONNXExporter.export_to_onnx(model_path, str(onnx_temp))
                onnx_to_convert = str(onnx_temp)
            else:
                onnx_to_convert = str(output_dir / 'model.onnx')
            
            try:
                TFLiteConverter.onnx_to_tflite(onnx_to_convert, str(tflite_path))
                package_info['formats']['tflite'] = {
                    'path': str(tflite_path),
                    'size_mb': tflite_path.stat().st_size / (1024 * 1024),
                }
            except Exception as e:
                print(f"Warning: TFLite conversion failed: {e}")
        
        return package_info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert models for edge deployment')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--output-dir', default='deployment', help='Output directory')
    parser.add_argument('--format', choices=['onnx', 'tflite', 'all'], default='all')
    parser.add_argument('--quantization', action='store_true', help='Apply quantization')
    
    args = parser.parse_args()
    
    if args.format in ['onnx', 'all']:
        ONNXExporter.export_to_onnx(args.model, f'{args.output_dir}/model.onnx')
        ONNXExporter.validate_onnx(f'{args.output_dir}/model.onnx')
    
    if args.format in ['tflite', 'all']:
        packager = DeploymentPackager()
        package = packager.create_deployment_bundle(args.model, args.output_dir)
        print(f"Deployment package created: {package}")
