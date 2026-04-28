"""
Comprehensive Benchmarking Script
Compare different model compression techniques
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List
import json
from datetime import datetime
from tabulate import tabulate


class BenchmarkSuite:
    """Run comprehensive benchmarks on multiple models"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize benchmark suite
        
        Args:
            device: Device to benchmark on
        """
        self.device = torch.device(device)
        self.results = {}
    
    def benchmark_model(self, model_path: str, model_name: str,
                       input_shape: tuple = (1, 3, 640, 640),
                       num_iterations: int = 100) -> Dict:
        """
        Benchmark a single model
        
        Args:
            model_path: Path to model
            model_name: Name of model
            input_shape: Input tensor shape
            num_iterations: Number of iterations
            
        Returns:
            Benchmark results
        """
        print(f"\nBenchmarking {model_name}...")
        
        try:
            # Load model
            model = torch.load(model_path, map_location=self.device)
            if isinstance(model, dict):
                # StateDict
                model = self._create_model_from_statedict(model)
            
            model.eval()
            model.to(self.device)
            
            # Get model info
            model_size = self._get_model_size(model)
            param_count = sum(p.numel() for p in model.parameters())
            
            # Warmup
            dummy_input = torch.randn(input_shape).to(self.device)
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_iterations):
                    start = time.time()
                    _ = model(dummy_input)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.time()
                    times.append((end - start) * 1000)
            
            times = np.array(times)
            
            results = {
                'model_name': model_name,
                'model_size_mb': model_size,
                'param_count': param_count,
                'latency_ms': {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times),
                },
                'fps': 1000 / np.mean(times),
                'throughput_samples_per_sec': (input_shape[0] * 1000) / np.mean(times),
            }
            
            self.results[model_name] = results
            
            print(f"  Model Size: {results['model_size_mb']:.2f} MB")
            print(f"  Parameters: {results['param_count']:,}")
            print(f"  Latency: {results['latency_ms']['mean']:.2f} ± {results['latency_ms']['std']:.2f} ms")
            print(f"  FPS: {results['fps']:.1f}")
            
            return results
            
        except Exception as e:
            print(f"  Error benchmarking {model_name}: {e}")
            return None
    
    def compare_models(self, baseline_name: str) -> Dict:
        """
        Compare all models against baseline
        
        Args:
            baseline_name: Name of baseline model
            
        Returns:
            Comparison results
        """
        if baseline_name not in self.results:
            raise ValueError(f"Baseline {baseline_name} not found")
        
        baseline = self.results[baseline_name]
        comparison = {}
        
        for model_name, results in self.results.items():
            if model_name == baseline_name:
                continue
            
            comparison[model_name] = {
                'size_reduction_%': (1 - results['model_size_mb'] / baseline['model_size_mb']) * 100,
                'param_reduction_%': (1 - results['param_count'] / baseline['param_count']) * 100,
                'speedup': baseline['latency_ms']['mean'] / results['latency_ms']['mean'],
                'fps_improvement_%': (results['fps'] / baseline['fps'] - 1) * 100,
            }
        
        return comparison
    
    def generate_report(self, output_path: str = 'benchmark_report.json'):
        """
        Generate benchmark report
        
        Args:
            output_path: Path to save report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'models': self.results,
            'comparisons': self.compare_models(list(self.results.keys())[0]) if len(self.results) > 1 else {},
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nBenchmark report saved to {output_path}")
        
        return report
    
    def print_summary(self):
        """
        Print summary of all benchmarks
        """
        if not self.results:
            print("No benchmark results to display")
            return
        
        # Prepare table data
        table_data = []
        for model_name, results in self.results.items():
            table_data.append([
                model_name,
                f"{results['model_size_mb']:.2f}",
                f"{results['param_count']:,}",
                f"{results['latency_ms']['mean']:.2f}",
                f"{results['fps']:.1f}",
            ])
        
        headers = ['Model', 'Size (MB)', 'Parameters', 'Latency (ms)', 'FPS']
        print("\n" + "="*80)
        print("Benchmark Summary")
        print("="*80)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    @staticmethod
    def _get_model_size(model) -> float:
        """Get model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    @staticmethod
    def _create_model_from_statedict(statedict):
        """Create a simple model from statedict (placeholder)"""
        # This is a placeholder - actual implementation depends on model architecture
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def forward(self, x):
                return x
        
        return SimpleModel()


def run_comprehensive_benchmark(models_dir: str, output_dir: str = 'benchmark_results'):
    """
    Run comprehensive benchmark on all models in directory
    
    Args:
        models_dir: Directory containing models
        output_dir: Output directory for results
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suite = BenchmarkSuite(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find all model files
    model_files = list(models_dir.glob('**/*.pt')) + list(models_dir.glob('**/*.pth'))
    
    print(f"\nFound {len(model_files)} models to benchmark")
    
    for model_file in model_files:
        model_name = model_file.stem
        suite.benchmark_model(str(model_file), model_name)
    
    # Generate report
    suite.print_summary()
    report_path = output_dir / 'benchmark_report.json'
    suite.generate_report(str(report_path))
    
    return suite.results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive benchmarks')
    parser.add_argument('--models-dir', required=True, help='Directory containing models')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    
    args = parser.parse_args()
    
    results = run_comprehensive_benchmark(args.models_dir, args.output_dir)
    print("\nBenchmarking complete!")
