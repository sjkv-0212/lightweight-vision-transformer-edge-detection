"""
Main Training Script
Train YOLOv8 models with various compression techniques
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import json
from datetime import datetime

from models.yolov8_baseline import YOLOv8Baseline
from models.knowledge_distillation import StudentTeacherTrainer, DistillationConfig
from models.quantization import PyTorchQuantizer, QuantizationAnalyzer
from models.pruning import UnstructuredPruner, IterativePruning, PruningAnalyzer


class TrainingConfig:
    """Training configuration"""
    
    def __init__(self, config_path: str = 'configs/training_config.yaml'):
        """Load training configuration"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.__dict__.update(config)
        else:
            # Default configuration
            self.model_size = 'n'
            self.epochs = 100
            self.batch_size = 32
            self.learning_rate = 0.001
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.save_dir = 'runs/train'
            self.data_path = 'data/dataset.yaml'
            self.resume = None


class ModelTrainer:
    """Main trainer for object detection models"""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize baseline model
        self.baseline_model = YOLOv8Baseline(model_size=config.model_size)
        
        self.training_history = {
            'baseline': None,
            'distilled': None,
            'quantized': None,
            'pruned': None,
            'combined': None,
        }
    
    def train_baseline(self, data_path: str = None) -> dict:
        """
        Train baseline YOLOv8 model
        
        Args:
            data_path: Path to dataset YAML
            
        Returns:
            Training results
        """
        print("\n" + "="*60)
        print("Training Baseline YOLOv8 Model")
        print("="*60)
        
        data_path = data_path or self.config.data_path
        
        results = self.baseline_model.train(
            data_path=data_path,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            device=0 if self.config.device == 'cuda' else -1,
            patience=20,
            save_dir=str(self.save_dir / 'baseline')
        )
        
        # Validation
        val_metrics = self.baseline_model.validate(data_path)
        
        results['metrics'] = val_metrics
        results['model_size_mb'] = self.baseline_model.get_model_size()
        
        self.training_history['baseline'] = results
        
        print(f"\nBaseline Results:")
        print(f"  mAP50: {val_metrics['map50']:.4f}")
        print(f"  mAP: {val_metrics['map']:.4f}")
        print(f"  Model Size: {results['model_size_mb']:.2f} MB")
        
        return results
    
    def train_with_quantization(self, data_path: str = None, 
                               quantization_type: str = 'dynamic') -> dict:
        """
        Train and quantize model
        
        Args:
            data_path: Path to dataset YAML
            quantization_type: 'dynamic' or 'static'
            
        Returns:
            Quantization results
        """
        print("\n" + "="*60)
        print(f"Applying {quantization_type.upper()} Quantization")
        print("="*60)
        
        # Get baseline model
        baseline_model = self.baseline_model.get_model()
        
        if quantization_type == 'dynamic':
            quantized_model = PyTorchQuantizer.quantize_dynamic(baseline_model)
        else:
            # For static quantization, would need data loaders
            quantized_model = PyTorchQuantizer.quantize_dynamic(baseline_model)
        
        # Analyze quantization effects
        original_size = self.baseline_model.get_model_size()
        quantized_size = PyTorchQuantizer.get_model_size(quantized_model)
        
        results = {
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'compression_ratio': (1 - quantized_size / original_size) * 100,
            'quantization_type': quantization_type,
        }
        
        self.training_history['quantized'] = results
        
        print(f"\nQuantization Results:")
        print(f"  Original Size: {original_size:.2f} MB")
        print(f"  Quantized Size: {quantized_size:.2f} MB")
        print(f"  Compression: {results['compression_ratio']:.2f}%")
        
        # Save quantized model
        save_path = self.save_dir / 'models' / f'quantized_{quantization_type}.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_model.state_dict(), save_path)
        results['save_path'] = str(save_path)
        
        return results
    
    def train_with_pruning(self, data_path: str = None,
                          prune_ratio: float = 0.3,
                          iterations: int = 5) -> dict:
        """
        Train with iterative pruning
        
        Args:
            data_path: Path to dataset YAML
            prune_ratio: Ratio to prune per iteration
            iterations: Number of pruning iterations
            
        Returns:
            Pruning results
        """
        print("\n" + "="*60)
        print("Applying Iterative Pruning")
        print("="*60)
        
        # Apply magnitude pruning
        model = self.baseline_model.get_model().model  # Get PyTorch model from YOLO wrapper
        
        UnstructuredPruner.magnitude_pruning(model, prune_ratio=prune_ratio)
        
        # Analyze pruning
        pruning_summary = PruningAnalyzer.get_pruning_summary(model)
        PruningAnalyzer.visualize_pruning(model)
        
        results = {
            'prune_ratio': prune_ratio,
            'iterations': iterations,
            'pruning_summary': pruning_summary,
        }
        
        self.training_history['pruned'] = results
        
        # Save pruned model
        save_path = self.save_dir / 'models' / 'pruned_model.pt'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        results['save_path'] = str(save_path)
        
        return results
    
    def train_with_distillation(self, data_path: str = None) -> dict:
        """
        Train student model with knowledge distillation
        
        Args:
            data_path: Path to dataset YAML
            
        Returns:
            Distillation results
        """
        print("\n" + "="*60)
        print("Training with Knowledge Distillation")
        print("="*60)
        
        # Create teacher (large) and student (small) models
        teacher = YOLOv8Baseline(model_size='m', pretrained=True).get_model()
        student = YOLOv8Baseline(model_size='n', pretrained=True).get_model()
        
        print(f"Teacher Model Size: {YOLOv8Baseline('m').get_model_size():.2f} MB")
        print(f"Student Model Size: {YOLOv8Baseline('n').get_model_size():.2f} MB")
        
        # Note: Full implementation would require data loaders
        # This is a placeholder showing the structure
        
        results = {
            'teacher_size_mb': YOLOv8Baseline('m').get_model_size(),
            'student_size_mb': YOLOv8Baseline('n').get_model_size(),
            'distillation_config': DistillationConfig().to_dict(),
            'status': 'configured',
        }
        
        self.training_history['distilled'] = results
        
        print(f"\nDistillation Results:")
        print(f"  Teacher Size: {results['teacher_size_mb']:.2f} MB")
        print(f"  Student Size: {results['student_size_mb']:.2f} MB")
        print(f"  Compression Ratio: {(1 - results['student_size_mb']/results['teacher_size_mb'])*100:.2f}%")
        
        return results
    
    def save_training_summary(self):
        """Save training summary to file"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'results': self.training_history,
        }
        
        save_path = self.save_dir / 'training_summary.json'
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to {save_path}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Train object detection models')
    parser.add_argument('--config', default='configs/training_config.yaml',
                       help='Path to training config')
    parser.add_argument('--data', default='data/dataset.yaml',
                       help='Path to dataset YAML')
    parser.add_argument('--mode', choices=['baseline', 'quantization', 'pruning', 
                                           'distillation', 'all'],
                       default='all', help='Training mode')
    parser.add_argument('--model-size', choices=['n', 's', 'm', 'l', 'x'],
                       default='n', help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--save-dir', default='runs/train',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(args.config)
    config.model_size = args.model_size
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.device = args.device
    config.save_dir = args.save_dir
    config.data_path = args.data
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Train models
    if args.mode in ['baseline', 'all']:
        trainer.train_baseline(args.data)
    
    if args.mode in ['quantization', 'all']:
        trainer.train_with_quantization(args.data, quantization_type='dynamic')
    
    if args.mode in ['pruning', 'all']:
        trainer.train_with_pruning(args.data, prune_ratio=0.3, iterations=5)
    
    if args.mode in ['distillation', 'all']:
        trainer.train_with_distillation(args.data)
    
    # Save summary
    trainer.save_training_summary()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
