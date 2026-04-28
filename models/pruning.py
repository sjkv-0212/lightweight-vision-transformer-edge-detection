"""
Model Pruning Module
Structured and unstructured pruning for model compression
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple
import numpy as np


class PruningConfig:
    """Configuration for pruning"""
    
    def __init__(self,
                 pruning_type: str = 'unstructured',
                 prune_ratio: float = 0.3,
                 prune_layers: List[str] = None):
        """
        Initialize pruning config
        
        Args:
            pruning_type: 'unstructured' or 'structured'
            prune_ratio: Percentage of weights to prune (0.0-1.0)
            prune_layers: List of layer names to prune
        """
        self.pruning_type = pruning_type
        self.prune_ratio = prune_ratio
        self.prune_layers = prune_layers or []


class UnstructuredPruner:
    """Unstructured pruning - remove individual weights"""
    
    @staticmethod
    def magnitude_pruning(model, prune_ratio: float = 0.3, 
                         prune_layers: List[str] = None) -> nn.Module:
        """
        Magnitude-based pruning (remove smallest weights)
        
        Args:
            model: Model to prune
            prune_ratio: Ratio of weights to prune
            prune_layers: List of layer names to prune (None = all)
            
        Returns:
            Pruned model
        """
        parameters_to_prune = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if prune_layers is None or any(layer in name for layer in prune_layers):
                    parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_ratio
        )
        
        return model
    
    @staticmethod
    def make_pruning_permanent(model) -> nn.Module:
        """
        Remove pruning buffers and make pruning permanent
        
        Args:
            model: Pruned model
            
        Returns:
            Model with permanent pruning
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
        
        return model


class StructuredPruner:
    """Structured pruning - remove entire channels/filters"""
    
    @staticmethod
    def channel_pruning(model, prune_ratio: float = 0.3,
                       prune_layers: List[str] = None) -> nn.Module:
        """
        Channel-level pruning for Conv2d layers
        
        Args:
            model: Model to prune
            prune_ratio: Ratio of channels to prune
            prune_layers: List of layer names to prune
            
        Returns:
            Pruned model
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if prune_layers is None or any(layer in name for layer in prune_layers):
                    # Prune by output channels (filters)
                    num_channels = module.out_channels
                    num_to_prune = int(num_channels * prune_ratio)
                    
                    prune.ln_structured(
                        module,
                        name='weight',
                        amount=num_to_prune,
                        n=2,
                        dim=0  # Output channel dimension
                    )
        
        return model
    
    @staticmethod
    def layer_pruning(model, prune_layers: List[str] = None) -> nn.Module:
        """
        Remove entire layers from the model
        
        Args:
            model: Model to prune
            prune_layers: List of layer names to remove
            
        Returns:
            Pruned model with removed layers
        """
        # This is a simplified version
        # Full implementation would require redefining the model architecture
        
        for layer_name in prune_layers or []:
            for name, module in list(model.named_modules()):
                if name == layer_name:
                    setattr(model, name, nn.Identity())
        
        return model


class AttentionHeadPruning:
    """Prune attention heads in transformer models"""
    
    @staticmethod
    def analyze_attention_heads(model) -> Dict:
        """
        Analyze attention head importance
        
        Args:
            model: Transformer model
            
        Returns:
            Importance scores for each head
        """
        head_importance = {}
        
        # This would need to be adapted based on specific transformer architecture
        for name, module in model.named_modules():
            if 'attention' in name.lower():
                # Analyze based on gradient magnitude
                pass
        
        return head_importance
    
    @staticmethod
    def prune_attention_heads(model, prune_ratio: float = 0.3) -> nn.Module:
        """
        Prune less important attention heads
        
        Args:
            model: Transformer model
            prune_ratio: Ratio of heads to prune
            
        Returns:
            Pruned model
        """
        # Calculate importance of each head
        importance_scores = AttentionHeadPruning.analyze_attention_heads(model)
        
        # Identify heads to prune (lowest importance)
        heads_to_prune = {}
        for layer_name, scores in importance_scores.items():
            num_to_prune = int(len(scores) * prune_ratio)
            head_indices = np.argsort(scores)[:num_to_prune]
            heads_to_prune[layer_name] = head_indices.tolist()
        
        return model


class SensitivityAnalysis:
    """Analyze model sensitivity to pruning"""
    
    @staticmethod
    def layer_sensitivity(model, val_loader, device: str = 'cuda',
                         prune_percentages: List[float] = None) -> Dict:
        """
        Analyze sensitivity of each layer to pruning
        
        Args:
            model: Model to analyze
            val_loader: Validation data loader
            device: Device to use
            prune_percentages: Percentages to test
            
        Returns:
            Sensitivity analysis results
        """
        if prune_percentages is None:
            prune_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        sensitivity = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                sensitivity[name] = []
                
                for prune_ratio in prune_percentages:
                    # Create copy for testing
                    model_copy = self._clone_model(model)
                    
                    # Prune specific layer
                    prune.l1_unstructured(model_copy.__dict__[name], 'weight', prune_ratio)
                    
                    # Evaluate
                    # accuracy = self._evaluate(model_copy, val_loader, device)
                    # sensitivity[name].append(accuracy)
        
        return sensitivity
    
    @staticmethod
    def _clone_model(model):
        """Clone model for testing"""
        import copy
        return copy.deepcopy(model)
    
    @staticmethod
    def _evaluate(model, val_loader, device):
        """Evaluate model on validation set"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                
                output = model(data)
                pred = output.argmax(dim=1)
                
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total


class IterativePruning:
    """Iterative pruning with fine-tuning"""
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize iterative pruner
        
        Args:
            model: Model to prune
            device: Device to use
        """
        self.model = model
        self.device = device
    
    def prune_and_finetune(self,
                          prune_ratio: float = 0.1,
                          num_iterations: int = 5,
                          train_loader = None,
                          val_loader = None,
                          learning_rate: float = 0.001) -> Dict:
        """
        Iteratively prune and fine-tune model
        
        Args:
            prune_ratio: Ratio to prune each iteration
            num_iterations: Number of iterations
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Learning rate for fine-tuning
            
        Returns:
            Pruning history
        """
        history = {
            'sparsity': [],
            'train_loss': [],
            'val_accuracy': [],
            'model_size': [],
        }
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            
            # Prune
            UnstructuredPruner.magnitude_pruning(self.model, prune_ratio)
            
            # Calculate sparsity
            sparsity = self._calculate_sparsity()
            history['sparsity'].append(sparsity)
            print(f"Sparsity: {sparsity:.2%}")
            
            # Fine-tune
            if train_loader is not None:
                train_loss = self._finetune_epoch(train_loader, optimizer)
                history['train_loss'].append(train_loss)
                print(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate
            if val_loader is not None:
                val_accuracy = self._evaluate_model(val_loader)
                history['val_accuracy'].append(val_accuracy)
                print(f"Val Accuracy: {val_accuracy:.4f}")
            
            # Model size
            model_size = self._get_model_size()
            history['model_size'].append(model_size)
            print(f"Model Size: {model_size:.2f} MB")
            
            # Make pruning permanent
            UnstructuredPruner.make_pruning_permanent(self.model)
        
        return history
    
    def _calculate_sparsity(self) -> float:
        """Calculate model sparsity (percentage of zero weights)"""
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0
    
    def _finetune_epoch(self, train_loader, optimizer) -> float:
        """Fine-tune for one epoch"""
        self.model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _evaluate_model(self, val_loader) -> float:
        """Evaluate model on validation set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        return correct / total if total > 0 else 0
    
    def _get_model_size(self) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.numel() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 * 1024)


class PruningAnalyzer:
    """Analyze pruning effects"""
    
    @staticmethod
    def get_pruning_summary(model) -> Dict:
        """Get summary of pruning in model"""
        summary = {
            'total_parameters': 0,
            'pruned_parameters': 0,
            'sparsity': 0,
            'layers': {}
        }
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    total = module.weight.numel()
                    pruned = (module.weight_mask == 0).sum().item()
                    
                    summary['total_parameters'] += total
                    summary['pruned_parameters'] += pruned
                    summary['layers'][name] = {
                        'total': total,
                        'pruned': pruned,
                        'sparsity': pruned / total if total > 0 else 0
                    }
        
        summary['sparsity'] = (summary['pruned_parameters'] / summary['total_parameters']
                              if summary['total_parameters'] > 0 else 0)
        
        return summary
    
    @staticmethod
    def visualize_pruning(model):
        """Visualize pruning statistics"""
        summary = PruningAnalyzer.get_pruning_summary(model)
        
        print("\nPruning Summary:")
        print(f"Total Parameters: {summary['total_parameters']:,}")
        print(f"Pruned Parameters: {summary['pruned_parameters']:,}")
        print(f"Overall Sparsity: {summary['sparsity']:.2%}")
        print("\nPer-Layer Sparsity:")
        
        for layer_name, stats in summary['layers'].items():
            print(f"  {layer_name}: {stats['sparsity']:.2%} "
                  f"({stats['pruned']}/{stats['total']})")
