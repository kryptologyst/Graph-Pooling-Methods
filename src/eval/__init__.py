"""Evaluation utilities and metrics for graph pooling methods."""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, AUROC, Precision, Recall
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils import get_device


class ModelEvaluator:
    """Comprehensive model evaluator for graph classification tasks."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        num_classes: int = 2
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained model to evaluate.
            test_loader: Test data loader.
            device: Device to use for evaluation.
            num_classes: Number of classes.
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or get_device("auto")
        self.num_classes = num_classes
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup metrics
        self._setup_metrics()
    
    def _setup_metrics(self) -> None:
        """Setup evaluation metrics."""
        self.metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
            "f1_macro": F1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
            "f1_micro": F1Score(task="multiclass", num_classes=self.num_classes, average="micro"),
            "f1_weighted": F1Score(task="multiclass", num_classes=self.num_classes, average="weighted"),
            "precision_macro": Precision(task="multiclass", num_classes=self.num_classes, average="macro"),
            "precision_micro": Precision(task="multiclass", num_classes=self.num_classes, average="micro"),
            "recall_macro": Recall(task="multiclass", num_classes=self.num_classes, average="macro"),
            "recall_micro": Recall(task="multiclass", num_classes=self.num_classes, average="micro"),
            "auroc": AUROC(task="multiclass", num_classes=self.num_classes)
        }
        
        # Move metrics to device
        for metric in self.metrics.values():
            metric.to(self.device)
    
    def evaluate(self) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Returns:
            Dictionary containing all evaluation results.
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_losses = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_losses.append(loss.item())
        
        # Convert to tensors
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
        probabilities_tensor = torch.tensor(all_probabilities)
        
        # Calculate metrics
        results = {}
        
        # Basic metrics
        results["loss"] = np.mean(all_losses)
        
        # TorchMetrics
        for name, metric in self.metrics.items():
            if name == "auroc":
                results[name] = metric(probabilities_tensor, targets_tensor).item()
            else:
                results[name] = metric(predictions_tensor, targets_tensor).item()
        
        # Additional analysis
        results["confusion_matrix"] = confusion_matrix(all_targets, all_predictions)
        results["classification_report"] = classification_report(
            all_targets, all_predictions, output_dict=True
        )
        
        # Per-class metrics
        results["per_class_metrics"] = self._calculate_per_class_metrics(
            all_targets, all_predictions, all_probabilities
        )
        
        return results
    
    def _calculate_per_class_metrics(
        self, 
        targets: List[int], 
        predictions: List[int], 
        probabilities: List[List[float]]
    ) -> Dict[str, Any]:
        """Calculate per-class metrics.
        
        Args:
            targets: True labels.
            predictions: Predicted labels.
            probabilities: Predicted probabilities.
            
        Returns:
            Dictionary containing per-class metrics.
        """
        per_class_metrics = {}
        
        for class_id in range(self.num_classes):
            # Binary classification for this class
            binary_targets = [1 if t == class_id else 0 for t in targets]
            binary_predictions = [1 if p == class_id else 0 for p in predictions]
            binary_probabilities = [prob[class_id] for prob in probabilities]
            
            # Calculate metrics
            tp = sum(1 for t, p in zip(binary_targets, binary_predictions) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(binary_targets, binary_predictions) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(binary_targets, binary_predictions) if t == 1 and p == 0)
            tn = sum(1 for t, p in zip(binary_targets, binary_predictions) if t == 0 and p == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[f"class_{class_id}"] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(binary_targets)
            }
        
        return per_class_metrics
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray, 
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array.
            class_names: Names of classes.
            save_path: Path to save the plot.
        """
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_class_distribution(
        self, 
        targets: List[int], 
        predictions: List[int],
        save_path: Optional[str] = None
    ) -> None:
        """Plot class distribution comparison.
        
        Args:
            targets: True labels.
            predictions: Predicted labels.
            save_path: Path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        unique_targets, counts_targets = np.unique(targets, return_counts=True)
        ax1.bar(unique_targets, counts_targets, alpha=0.7, color="blue")
        ax1.set_title("True Class Distribution")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        
        # Predicted distribution
        unique_predictions, counts_predictions = np.unique(predictions, return_counts=True)
        ax2.bar(unique_predictions, counts_predictions, alpha=0.7, color="red")
        ax2.set_title("Predicted Class Distribution")
        ax2.set_xlabel("Class")
        ax2.set_ylabel("Count")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()


class ModelComparison:
    """Compare multiple models and create leaderboard."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize model comparison.
        
        Args:
            results_dir: Directory to save comparison results.
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results = []
    
    def add_model_result(
        self,
        model_name: str,
        model_config: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        training_time: float,
        model_size_mb: float
    ) -> None:
        """Add model result to comparison.
        
        Args:
            model_name: Name of the model.
            model_config: Model configuration.
            evaluation_results: Evaluation results.
            training_time: Training time in seconds.
            model_size_mb: Model size in MB.
        """
        result = {
            "model_name": model_name,
            "model_config": model_config,
            "evaluation_results": evaluation_results,
            "training_time": training_time,
            "model_size_mb": model_size_mb
        }
        
        self.results.append(result)
    
    def create_leaderboard(self) -> pd.DataFrame:
        """Create model leaderboard.
        
        Returns:
            DataFrame with model comparison results.
        """
        leaderboard_data = []
        
        for result in self.results:
            eval_results = result["evaluation_results"]
            
            leaderboard_data.append({
                "Model": result["model_name"],
                "Accuracy": eval_results["accuracy"],
                "F1-Macro": eval_results["f1_macro"],
                "F1-Micro": eval_results["f1_micro"],
                "AUROC": eval_results["auroc"],
                "Precision-Macro": eval_results["precision_macro"],
                "Recall-Macro": eval_results["recall_macro"],
                "Training Time (s)": result["training_time"],
                "Model Size (MB)": result["model_size_mb"]
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        # Sort by accuracy (descending)
        leaderboard = leaderboard.sort_values("Accuracy", ascending=False)
        
        return leaderboard
    
    def save_results(self, filename: str = "model_comparison.json") -> None:
        """Save comparison results to file.
        
        Args:
            filename: Name of the file to save.
        """
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")
    
    def plot_comparison(
        self, 
        metrics: List[str] = ["accuracy", "f1_macro", "auroc"],
        save_path: Optional[str] = None
    ) -> None:
        """Plot model comparison.
        
        Args:
            metrics: List of metrics to plot.
            save_path: Path to save the plot.
        """
        leaderboard = self.create_leaderboard()
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            metric_col = metric.replace("_", "-").title()
            if metric_col in leaderboard.columns:
                axes[i].bar(leaderboard["Model"], leaderboard[metric_col])
                axes[i].set_title(f"{metric_col} Comparison")
                axes[i].set_ylabel(metric_col)
                axes[i].tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate comprehensive comparison report.
        
        Returns:
            String containing the report.
        """
        leaderboard = self.create_leaderboard()
        
        report = "=" * 80 + "\n"
        report += "MODEL COMPARISON REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Summary statistics
        report += "SUMMARY STATISTICS:\n"
        report += "-" * 40 + "\n"
        report += f"Number of models compared: {len(self.results)}\n"
        report += f"Best accuracy: {leaderboard['Accuracy'].max():.4f}\n"
        report += f"Best F1-Macro: {leaderboard['F1-Macro'].max():.4f}\n"
        report += f"Best AUROC: {leaderboard['AUROC'].max():.4f}\n\n"
        
        # Leaderboard
        report += "LEADERBOARD:\n"
        report += "-" * 40 + "\n"
        report += leaderboard.to_string(index=False) + "\n\n"
        
        # Detailed results
        report += "DETAILED RESULTS:\n"
        report += "-" * 40 + "\n"
        
        for result in self.results:
            report += f"\nModel: {result['model_name']}\n"
            report += f"Training Time: {result['training_time']:.2f}s\n"
            report += f"Model Size: {result['model_size_mb']:.2f}MB\n"
            
            eval_results = result["evaluation_results"]
            report += f"Accuracy: {eval_results['accuracy']:.4f}\n"
            report += f"F1-Macro: {eval_results['f1_macro']:.4f}\n"
            report += f"AUROC: {eval_results['auroc']:.4f}\n"
            report += "-" * 40 + "\n"
        
        return report
