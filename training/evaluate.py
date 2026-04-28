"""
Evaluation and Metrics Computation
Calculate mAP, FPS, latency, and other metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
from datetime import datetime


class MetricsCalculator:
    """Calculate evaluation metrics for object detection"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize metrics calculator
        
        Args:
            device: Device to use for computation
        """
        self.device = torch.device(device)
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection area
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        """
        Calculate Average Precision using 11-point interpolation
        
        Args:
            recall: Array of recall values
            precision: Array of precision values
            
        Returns:
            Average Precision
        """
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.
        
        return ap
    
    def calculate_map(self, predictions: List[Dict], 
                     ground_truth: List[Dict],
                     iou_threshold: float = 0.5) -> Dict:
        """
        Calculate mean Average Precision (mAP)
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth annotations
            iou_threshold: IoU threshold for positive match
            
        Returns:
            mAP metrics
        """
        # Group by class
        predictions_by_class = {}
        gt_by_class = {}
        
        for pred in predictions:
            class_id = pred['class_id']
            if class_id not in predictions_by_class:
                predictions_by_class[class_id] = []
            predictions_by_class[class_id].append(pred)
        
        for gt in ground_truth:
            class_id = gt['class_id']
            if class_id not in gt_by_class:
                gt_by_class[class_id] = []
            gt_by_class[class_id].append(gt)
        
        # Calculate AP for each class
        class_aps = {}
        for class_id in predictions_by_class.keys():
            preds = predictions_by_class[class_id]
            gts = gt_by_class.get(class_id, [])
            
            # Sort predictions by confidence
            preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)
            
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            
            gt_matched = set()
            
            for pred_idx, pred in enumerate(preds):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_idx in gt_matched:
                        continue
                    
                    iou = self.calculate_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp[pred_idx] = 1
                    gt_matched.add(best_gt_idx)
                else:
                    fp[pred_idx] = 1
            
            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / max(len(gts), 1)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            ap = self.calculate_ap(recalls, precisions)
            class_aps[class_id] = ap
        
        mAP = np.mean(list(class_aps.values())) if class_aps else 0.0
        
        return {
            'mAP': mAP,
            'class_aps': class_aps,
        }


class PerformanceBenchmark:
    """Benchmark model performance on edge devices"""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize benchmarker
        
        Args:
            model: Model to benchmark
            device: Device to benchmark on
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
    
    def measure_latency(self, 
                       input_shape: Tuple = (1, 3, 640, 640),
                       num_iterations: int = 100,
                       warmup: int = 10) -> Dict:
        """
        Measure inference latency
        
        Args:
            input_shape: Input tensor shape
            num_iterations: Number of iterations for averaging
            warmup: Number of warmup iterations
            
        Returns:
            Latency statistics
        """
        self.model.eval()
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
        
        # Synchronize
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        return {
            'mean_latency_ms': np.mean(times),
            'std_latency_ms': np.std(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'median_latency_ms': np.median(times),
        }
    
    def measure_fps(self,
                   input_shape: Tuple = (1, 3, 640, 640),
                   num_iterations: int = 100) -> Dict:
        """
        Measure Frames Per Second (FPS)
        
        Args:
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            
        Returns:
            FPS statistics
        """
        latency = self.measure_latency(input_shape, num_iterations)
        
        mean_fps = 1000 / latency['mean_latency_ms']
        min_fps = 1000 / latency['max_latency_ms']
        max_fps = 1000 / latency['min_latency_ms']
        
        return {
            'mean_fps': mean_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'latency_ms': latency['mean_latency_ms'],
        }
    
    def measure_memory_usage(self, 
                            input_shape: Tuple = (1, 3, 640, 640)) -> Dict:
        """
        Measure GPU memory usage
        
        Args:
            input_shape: Input tensor shape
            
        Returns:
            Memory statistics
        """
        if self.device.type != 'cuda':
            return {'error': 'Memory measurement only supported on CUDA'}
        
        self.model.eval()
        
        # Get baseline memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        dummy_input = torch.randn(input_shape).to(self.device)
        
        with torch.no_grad():
            _ = self.model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        return {
            'peak_memory_mb': peak_memory,
            'model_size_mb': sum(p.numel() * p.element_size() for p in 
                                 self.model.parameters()) / (1024 ** 2),
        }
    
    def measure_model_size(self) -> float:
        """Measure model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        total_size = (param_size + buffer_size) / (1024 ** 2)
        
        return total_size


class Evaluator:
    """Complete evaluation pipeline"""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize evaluator
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = device
        self.metrics_calc = MetricsCalculator(device)
        self.benchmark = PerformanceBenchmark(model, device)
    
    def evaluate_full(self,
                     val_loader: DataLoader = None,
                     input_shape: Tuple = (1, 3, 640, 640),
                     num_benchmark_iters: int = 100) -> Dict:
        """
        Full evaluation including accuracy and performance metrics
        
        Args:
            val_loader: Validation data loader (optional)
            input_shape: Input tensor shape for benchmarking
            num_benchmark_iters: Number of iterations for benchmarking
            
        Returns:
            Complete evaluation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {},
            'performance': {},
        }
        
        # Model info
        results['model_info']['size_mb'] = self.benchmark.measure_model_size()
        
        # Performance benchmarks
        print("Measuring latency...")
        latency = self.benchmark.measure_latency(input_shape, num_benchmark_iters)
        results['performance']['latency'] = latency
        
        print("Measuring FPS...")
        fps = self.benchmark.measure_fps(input_shape, num_benchmark_iters)
        results['performance']['fps'] = fps
        
        print("Measuring memory...")
        memory = self.benchmark.measure_memory_usage(input_shape)
        results['performance']['memory'] = memory
        
        # Validation accuracy if data loader provided
        if val_loader is not None:
            print("Evaluating on validation set...")
            # This would require task-specific evaluation
            # Placeholder for actual implementation
            results['validation'] = {
                'status': 'validation_data_required_for_accuracy_metrics'
            }
        
        return results
    
    def save_results(self, results: Dict, save_path: str):
        """Save evaluation results"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {save_path}")


def evaluate_models(baseline_results: Dict, 
                   compressed_results: Dict) -> Dict:
    """
    Compare baseline and compressed models
    
    Args:
        baseline_results: Baseline model evaluation results
        compressed_results: Compressed model evaluation results
        
    Returns:
        Comparison results
    """
    comparison = {
        'baseline': baseline_results,
        'compressed': compressed_results,
        'improvements': {
            'speedup': (baseline_results['performance']['latency']['mean_latency_ms'] / 
                       compressed_results['performance']['latency']['mean_latency_ms']),
            'size_reduction': ((baseline_results['model_info']['size_mb'] - 
                               compressed_results['model_info']['size_mb']) / 
                              baseline_results['model_info']['size_mb'] * 100),
        }
    }
    
    return comparison


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate object detection models')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-iterations', type=int, default=100)
    parser.add_argument('--save-dir', default='runs/eval')
    
    args = parser.parse_args()
    
    # Load model (adjust based on model type)
    print(f"Loading model from {args.model}")
    
    # Placeholder for actual model loading
    # model = torch.load(args.model)
    
    # Evaluate
    # evaluator = Evaluator(model, device=args.device)
    # results = evaluator.evaluate_full(num_benchmark_iters=args.num_iterations)
    
    # Save results
    # save_path = f"{args.save_dir}/evaluation_results.json"
    # evaluator.save_results(results, save_path)
    
    print("Evaluation complete!")
