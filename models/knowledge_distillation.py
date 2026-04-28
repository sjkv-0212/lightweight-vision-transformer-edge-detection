"""
Knowledge Distillation Module
Teacher-Student learning for model compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple
import numpy as np


class KnowledgeDistillationLoss(nn.Module):
    """KL divergence loss for knowledge distillation"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Initialize KD loss
        
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight between distillation and task loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, ground_truth=None, task_loss=None):
        """
        Calculate KD loss
        
        Args:
            student_logits: Student model output
            teacher_logits: Teacher model output
            ground_truth: Ground truth labels (optional)
            task_loss: Task-specific loss (optional)
            
        Returns:
            Total loss
        """
        # Soften the probabilities
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence loss
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # If task loss is provided, combine both
        if task_loss is not None:
            return self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        
        return distillation_loss


class FeatureDistillation(nn.Module):
    """Feature-level knowledge distillation"""
    
    def __init__(self, student_channels: int, teacher_channels: int):
        """
        Initialize feature distillation
        
        Args:
            student_channels: Number of channels in student features
            teacher_channels: Number of channels in teacher features
        """
        super().__init__()
        self.adapter = nn.Conv2d(student_channels, teacher_channels, 1)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_features, teacher_features):
        """
        Calculate feature distillation loss
        
        Args:
            student_features: Features from student model
            teacher_features: Features from teacher model
            
        Returns:
            Feature distillation loss
        """
        # Adapt student features to match teacher dimensions
        adapted_features = self.adapter(student_features)
        
        # Normalize features
        adapted_features = F.normalize(adapted_features, p=2, dim=1)
        teacher_features = F.normalize(teacher_features, p=2, dim=1)
        
        # Calculate MSE loss between features
        loss = self.mse_loss(adapted_features, teacher_features)
        
        return loss


class StudentTeacherTrainer:
    """Trainer for knowledge distillation"""
    
    def __init__(self, 
                 student_model,
                 teacher_model,
                 device: str = 'cuda',
                 temperature: float = 4.0,
                 alpha: float = 0.7):
        """
        Initialize trainer
        
        Args:
            student_model: Student model to train
            teacher_model: Teacher model (frozen)
            device: Device to train on
            temperature: KD temperature
            alpha: Weight between distillation and task loss
        """
        self.student = student_model
        self.teacher = teacher_model
        self.device = device
        
        # Freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
        self.feature_distillation = FeatureDistillation(512, 512)  # Adjust channels as needed
    
    def train_epoch(self,
                   train_loader: DataLoader,
                   optimizer,
                   epoch: int) -> Dict:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Epoch number
            
        Returns:
            Training metrics
        """
        self.student.train()
        self.teacher.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        
        for images, targets in progress_bar:
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # Forward pass
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_logits = self.teacher(images)
            
            student_logits = self.student(images)
            
            # Calculate KD loss
            loss = self.kd_loss(student_logits, teacher_logits)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            progress_bar.set_postfix({'loss': total_loss / total_samples})
        
        avg_loss = total_loss / total_samples
        
        return {
            'epoch': epoch,
            'train_loss': avg_loss,
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.student.eval()
        self.teacher.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Validating'):
                images = images.to(self.device)
                batch_size = images.size(0)
                
                teacher_logits = self.teacher(images)
                student_logits = self.student(images)
                
                loss = self.kd_loss(student_logits, teacher_logits)
                
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        
        return {
            'val_loss': avg_loss,
        }
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              learning_rate: float = 0.001,
              save_path: str = 'student_model.pt') -> Dict:
        """
        Complete training loop with knowledge distillation
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        optimizer = Adam(self.student.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_metrics['val_loss'])
            
            scheduler.step()
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                torch.save(self.student.state_dict(), save_path)
                print(f"Model saved at epoch {epoch}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:.4f} | "
                  f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Load best model
        self.student.load_state_dict(torch.load(save_path))
        
        return history


class DistillationConfig:
    """Configuration for knowledge distillation"""
    
    def __init__(self):
        self.temperature = 4.0
        self.alpha = 0.7
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        self.patience = 20
    
    def to_dict(self) -> Dict:
        return {
            'temperature': self.temperature,
            'alpha': self.alpha,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
        }
