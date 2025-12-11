"""Training utilities and trainer class for graph pooling methods."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, AUROC
from tqdm import tqdm
import wandb
from omegaconf import DictConfig

from ..utils import EarlyStopping, get_device, set_seed, format_time
from ..data import GraphDataset


class Trainer:
    """Trainer class for graph neural network models with pooling."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: DictConfig,
        device: Optional[torch.device] = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            test_loader: Test data loader.
            config: Training configuration.
            device: Device to use for training.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or get_device(config.get("device", "auto"))
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        
        # Setup metrics
        self._setup_metrics()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.get("early_stopping", {}).get("patience", 20),
            min_delta=config.training.get("early_stopping", {}).get("min_delta", 0.001),
            mode=config.training.get("early_stopping", {}).get("mode", "max"),
            restore_best_weights=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = None
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        optimizer_config = self.config.training.get("optimizer", {})
        
        if optimizer_config.get("_target_") == "torch.optim.Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config.get("lr", 0.01),
                weight_decay=optimizer_config.get("weight_decay", 5e-4),
                betas=optimizer_config.get("betas", [0.9, 0.999])
            )
        else:
            # Default Adam optimizer
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=0.01,
                weight_decay=5e-4
            )
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        scheduler_config = self.config.training.get("scheduler", {})
        
        if scheduler_config.get("_target_") == "torch.optim.lr_scheduler.ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=scheduler_config.get("mode", "min"),
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
                min_lr=scheduler_config.get("min_lr", 1e-6)
            )
        else:
            self.scheduler = None
    
    def _setup_loss_function(self) -> None:
        """Setup loss function."""
        loss_function = self.config.training.get("loss_function", "cross_entropy")
        
        if loss_function == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == "nll_loss":
            self.criterion = nn.NLLLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def _setup_metrics(self) -> None:
        """Setup evaluation metrics."""
        self.train_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=self._get_num_classes()),
            "f1": F1Score(task="multiclass", num_classes=self._get_num_classes(), average="macro")
        }
        
        self.val_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=self._get_num_classes()),
            "f1": F1Score(task="multiclass", num_classes=self._get_num_classes(), average="macro"),
            "auroc": AUROC(task="multiclass", num_classes=self._get_num_classes())
        }
        
        self.test_metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=self._get_num_classes()),
            "f1": F1Score(task="multiclass", num_classes=self._get_num_classes(), average="macro"),
            "auroc": AUROC(task="multiclass", num_classes=self._get_num_classes())
        }
        
        # Move metrics to device
        for metrics_dict in [self.train_metrics, self.val_metrics, self.test_metrics]:
            for metric in metrics_dict.values():
                metric.to(self.device)
    
    def _get_num_classes(self) -> int:
        """Get number of classes from the dataset."""
        # This is a simplified approach - in practice, you might want to pass this explicitly
        return 2  # Default for MUTAG dataset
    
    def _setup_logging(self) -> None:
        """Setup logging (wandb, tensorboard, etc.)."""
        self.use_wandb = self.config.logging.get("use_wandb", False)
        self.use_tensorboard = self.config.logging.get("use_tensorboard", False)
        
        if self.use_wandb:
            wandb.init(
                project=self.config.experiment.get("name", "graph_pooling"),
                tags=self.config.experiment.get("tags", []),
                notes=self.config.experiment.get("notes", ""),
                config=self.config
            )
        
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = os.path.join(self.config.paths.get("log_dir", "logs"), 
                                 self.config.experiment.get("name", "experiment"))
            self.writer = SummaryWriter(log_dir)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(logits, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        
        # Update metrics
        train_metrics = {"loss": avg_loss}
        for name, metric in self.train_metrics.items():
            metric_value = metric(torch.tensor(all_predictions), torch.tensor(all_targets))
            train_metrics[name] = metric_value.item()
        
        return train_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(logits, batch.y)
                
                # Update metrics
                total_loss += loss.item()
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        
        # Update metrics
        val_metrics = {"loss": avg_loss}
        for name, metric in self.val_metrics.items():
            if name == "auroc":
                metric_value = metric(torch.tensor(all_probabilities), torch.tensor(all_targets))
            else:
                metric_value = metric(torch.tensor(all_predictions), torch.tensor(all_targets))
            val_metrics[name] = metric_value.item()
        
        return val_metrics
    
    def test_epoch(self) -> Dict[str, float]:
        """Test for one epoch.
        
        Returns:
            Dictionary of test metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(logits, batch.y)
                
                # Update metrics
                total_loss += loss.item()
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.test_loader)
        
        # Update metrics
        test_metrics = {"loss": avg_loss}
        for name, metric in self.test_metrics.items():
            if name == "auroc":
                metric_value = metric(torch.tensor(all_probabilities), torch.tensor(all_targets))
            else:
                metric_value = metric(torch.tensor(all_predictions), torch.tensor(all_targets))
            test_metrics[name] = metric_value.item()
        
        return test_metrics
    
    def train(self) -> Dict[str, Any]:
        """Train the model.
        
        Returns:
            Dictionary containing training results.
        """
        start_time = time.time()
        
        print(f"Starting training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.config.training.get("max_epochs", 100)):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Check early stopping
            if self.early_stopping(val_metrics["accuracy"], self.model):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            if val_metrics["accuracy"] > (self.best_val_score or 0):
                self.best_val_score = val_metrics["accuracy"]
                self._save_checkpoint(epoch, val_metrics)
        
        # Final test
        test_metrics = self.test_epoch()
        
        training_time = time.time() - start_time
        
        results = {
            "best_val_score": self.best_val_score,
            "test_metrics": test_metrics,
            "training_time": training_time,
            "total_epochs": self.current_epoch + 1
        }
        
        print(f"Training completed in {format_time(training_time)}")
        print(f"Best validation accuracy: {self.best_val_score:.4f}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        
        return results
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log metrics to various backends.
        
        Args:
            epoch: Current epoch.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
        """
        # Console logging
        print(f"Epoch {epoch:3d}: "
              f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {}
            for key, value in train_metrics.items():
                log_dict[f"train/{key}"] = value
            for key, value in val_metrics.items():
                log_dict[f"val/{key}"] = value
            wandb.log(log_dict, step=epoch)
        
        # Tensorboard logging
        if self.use_tensorboard:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            metrics: Current metrics.
        """
        checkpoint_dir = self.config.paths.get("checkpoint_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Metrics: {checkpoint['metrics']}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.use_wandb:
            wandb.finish()
        
        if self.use_tensorboard:
            self.writer.close()
