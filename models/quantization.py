"""
Model Quantization Module
INT8 and FP16 quantization for edge deployment
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, prepare_qat, convert
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple
import numpy as np


class QuantizationConfig:
    """Configuration for quantization"""
    
    def __init__(self, 
                 quantization_type: str = 'dynamic',
                 dtype: torch.dtype = torch.qint8,
                 calibration_samples: int = 100):
        """
        Initialize quantization config
        
        Args:
            quantization_type: 'dynamic' or 'static'
            dtype: torch.qint8 or torch.float16
            calibration_samples: Number of samples for calibration
        """
        self.quantization_type = quantization_type
        self.dtype = dtype
        self.calibration_samples = calibration_samples


class PyTorchQuantizer:
    """PyTorch model quantization"""
    
    @staticmethod
    def quantize_dynamic(model, qconfig_spec=None) -> nn.Module:
        """
        Dynamic quantization (post-training)
        
        Args:
            model: Model to quantize
            qconfig_spec: Quantization configuration
            
        Returns:
            Quantized model
        """
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear},
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def quantize_static(model, 
                       train_loader,
                       val_loader,
                       num_calibration_batches: int = 32) -> nn.Module:
        """
        Static quantization (QAT - Quantization Aware Training)
        
        Args:
            model: Model to quantize
            train_loader: Training data loader
            val_loader: Validation data loader
            num_calibration_batches: Number of batches for calibration
            
        Returns:
            Quantized model
        """
        # Prepare model for QAT
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        
        # Calibration
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                if batch_idx >= num_calibration_batches:
                    break
                _ = model(data)
        
        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)
        
        return model
    
    @staticmethod
    def get_model_size(model) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = (param_size + buffer_size) / (1024 * 1024)
        return total_size


class TensorFlowQuantizer:
    """TensorFlow model quantization for TFLite conversion"""
    
    @staticmethod
    def convert_to_tflite(model_path: str, 
                         output_path: str,
                         quantization_type: str = 'dynamic',
                         representative_dataset = None):
        """
        Convert PyTorch model to TFLite with quantization
        
        Args:
            model_path: Path to PyTorch model
            output_path: Path to save TFLite model
            quantization_type: 'dynamic' or 'full_integer'
            representative_dataset: Dataset for quantization calibration
        """
        # Note: This requires ONNX conversion first
        # PyTorch -> ONNX -> TensorFlow -> TFLite
        
        import onnx
        import onnx_tf
        
        # For full implementation, use ultralytics export
        print(f"Converting {model_path} to TFLite with {quantization_type} quantization")
        
        # Placeholder for actual conversion
        # In practice, use: model.export(format='tflite', int8=True)
        
        return output_path
    
    @staticmethod
    def quantize_tflite(tflite_model_path: str,
                       representative_dataset,
                       output_path: str):
        """
        Quantize TFLite model
        
        Args:
            tflite_model_path: Path to TFLite model
            representative_dataset: Calibration dataset
            output_path: Path to save quantized model
        """
        converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_data_gen():
            for input_data in representative_dataset:
                yield [input_data.astype(np.float32)]
        
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        quantized_tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(quantized_tflite_model)
        
        print(f"Quantized TFLite model saved to {output_path}")


class FP16Quantizer:
    """FP16 (half-precision) quantization"""
    
    @staticmethod
    def quantize_fp16(model) -> nn.Module:
        """
        Convert model to FP16
        
        Args:
            model: Model to convert
            
        Returns:
            FP16 model
        """
        model = model.half()
        return model
    
    @staticmethod
    def mixed_precision_training(model, device: str = 'cuda'):
        """
        Enable automatic mixed precision training
        
        Args:
            model: Model to enable AMP for
            device: Device type
            
        Returns:
            GradScaler for loss scaling
        """
        if device == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            return scaler
        else:
            raise ValueError("Mixed precision only supported on CUDA")


class QuantizationAnalyzer:
    """Analyze quantization effects"""
    
    @staticmethod
    def compare_models(original_model, quantized_model, test_loader, device: str = 'cuda'):
        """
        Compare original and quantized models
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_loader: Test data loader
            device: Device to use
            
        Returns:
            Comparison metrics
        """
        original_model.eval()
        quantized_model.eval()
        
        original_correct = 0
        quantized_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                
                # Original model
                original_output = original_model(data)
                original_pred = original_output.argmax(dim=1)
                original_correct += (original_pred == target).sum().item()
                
                # Quantized model
                quantized_output = quantized_model(data)
                quantized_pred = quantized_output.argmax(dim=1)
                quantized_correct += (quantized_pred == target).sum().item()
                
                total += target.size(0)
        
        original_accuracy = original_correct / total
        quantized_accuracy = quantized_correct / total
        accuracy_drop = (original_accuracy - quantized_accuracy) * 100
        
        original_size = PyTorchQuantizer.get_model_size(original_model)
        quantized_size = PyTorchQuantizer.get_model_size(quantized_model)
        compression_ratio = (1 - quantized_size / original_size) * 100
        
        return {
            'original_accuracy': original_accuracy,
            'quantized_accuracy': quantized_accuracy,
            'accuracy_drop': accuracy_drop,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': compression_ratio,
        }
    
    @staticmethod
    def get_weight_distribution(model) -> Dict:
        """
        Analyze weight distribution
        
        Args:
            model: Model to analyze
            
        Returns:
            Weight statistics
        """
        stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
        }
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                stats['mean'].append(param.data.mean().item())
                stats['std'].append(param.data.std().item())
                stats['min'].append(param.data.min().item())
                stats['max'].append(param.data.max().item())
        
        return {
            'mean': np.mean(stats['mean']),
            'std': np.mean(stats['std']),
            'min': np.min(stats['min']),
            'max': np.max(stats['max']),
        }


class QuantizationHelper:
    """Helper functions for quantization"""
    
    @staticmethod
    def save_quantized_model(model, path: str):
        """Save quantized model"""
        torch.save(model.state_dict(), path)
        print(f"Quantized model saved to {path}")
    
    @staticmethod
    def load_quantized_model(model, path: str):
        """Load quantized model"""
        model.load_state_dict(torch.load(path))
        print(f"Quantized model loaded from {path}")
        return model
    
    @staticmethod
    def benchmark_quantization(original_model, quantized_model, input_size: Tuple, 
                              num_iterations: int = 100, device: str = 'cuda'):
        """
        Benchmark quantization performance
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            input_size: Input tensor size
            num_iterations: Number of iterations for benchmarking
            device: Device to benchmark on
            
        Returns:
            Benchmark results
        """
        import time
        
        original_model.eval()
        quantized_model.eval()
        
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = original_model(dummy_input)
                _ = quantized_model(dummy_input)
        
        # Benchmark original
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = original_model(dummy_input)
        torch.cuda.synchronize() if device == 'cuda' else None
        original_time = time.time() - start
        
        # Benchmark quantized
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = quantized_model(dummy_input)
        torch.cuda.synchronize() if device == 'cuda' else None
        quantized_time = time.time() - start
        
        speedup = original_time / quantized_time
        
        return {
            'original_time_ms': (original_time / num_iterations) * 1000,
            'quantized_time_ms': (quantized_time / num_iterations) * 1000,
            'speedup': speedup,
        }
