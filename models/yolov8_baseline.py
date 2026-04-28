"""
YOLOv8 Baseline Model Implementation
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Tuple, List
import yaml


class YOLOv8Baseline:
    """YOLOv8 baseline model for object detection"""
    
    def __init__(self, model_size: str = 'n', pretrained: bool = True):
        """
        Initialize YOLOv8 model
        
        Args:
            model_size: Model size ('n' for nano, 's' for small, 'm' for medium, 'l' for large)
            pretrained: Use pretrained weights
        """
        self.model_size = model_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLOv8 model
        model_name = f'yolov8{model_size}.pt' if pretrained else f'yolov8{model_size}.yaml'
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.model_info = self._get_model_info()
    
    def _get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'size': self.model_size,
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'device': str(self.device),
        }
    
    def train(self, 
              data_path: str,
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              device: int = 0,
              patience: int = 20,
              save_dir: str = 'runs/detect') -> Dict:
        """
        Train YOLOv8 model
        
        Args:
            data_path: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Image size
            device: GPU device ID
            patience: Early stopping patience
            save_dir: Directory to save results
            
        Returns:
            Training results dictionary
        """
        results = self.model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            project=save_dir,
            name='baseline',
            verbose=True,
            workers=4,
            cache=True,
            amp=True,  # Automatic Mixed Precision
        )
        
        return {
            'status': 'completed',
            'save_dir': save_dir,
            'results': results
        }
    
    def validate(self, data_path: str, imgsz: int = 640) -> Dict:
        """
        Validate model
        
        Args:
            data_path: Path to dataset YAML file
            imgsz: Image size
            
        Returns:
            Validation metrics
        """
        metrics = self.model.val(
            data=data_path,
            imgsz=imgsz,
            device=self.device,
            verbose=True
        )
        
        return {
            'map50': metrics.box.map50,
            'map': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr,
        }
    
    def predict(self, source, conf: float = 0.5, iou: float = 0.45) -> List:
        """
        Run inference on images
        
        Args:
            source: Image path, directory, or list of images
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            List of predictions
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        
        return results
    
    def get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in self.model.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = (param_size + buffer_size) / (1024 * 1024)
        return total_size
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        self.model = YOLO(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")
    
    def get_model(self):
        """Get the underlying YOLO model"""
        return self.model


class YOLOv8ModelConfig:
    """Configuration for different YOLOv8 variants"""
    
    VARIANTS = {
        'n': {  # Nano
            'depth_multiple': 0.33,
            'width_multiple': 0.25,
            'max_channels': 1024,
        },
        's': {  # Small
            'depth_multiple': 0.33,
            'width_multiple': 0.50,
            'max_channels': 1024,
        },
        'm': {  # Medium
            'depth_multiple': 0.67,
            'width_multiple': 0.75,
            'max_channels': 1024,
        },
        'l': {  # Large
            'depth_multiple': 1.0,
            'width_multiple': 1.0,
            'max_channels': 1024,
        },
        'x': {  # Extra Large
            'depth_multiple': 1.33,
            'width_multiple': 1.25,
            'max_channels': 1024,
        }
    }
    
    @classmethod
    def get_config(cls, variant: str) -> Dict:
        """Get configuration for a variant"""
        return cls.VARIANTS.get(variant, cls.VARIANTS['n'])
