#!/usr/bin/env python3
"""Main training script for graph pooling methods."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from src.data import GraphDataset, create_data_loaders, get_dataset_stats
from src.models import GCNWithPooling, GATWithPooling, GraphSAGEWithPooling
from src.train import Trainer
from src.eval import ModelEvaluator, ModelComparison
from src.utils import set_seed, get_device, get_model_size_mb, format_time
from src.utils.visualization import GraphVisualizer


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function.
    
    Args:
        config: Hydra configuration object.
    """
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.paths.data_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(config.paths.log_dir, exist_ok=True)
    os.makedirs(config.paths.assets_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = hydra.utils.instantiate(config.data)
    dataset_stats = get_dataset_stats(dataset)
    
    print(f"Dataset statistics:")
    for key, value in dataset_stats.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model comparison
    model_comparison = ModelComparison(results_dir=config.paths.assets_dir)
    
    # Define models to compare
    model_configs = [
        ("GCN-TopK", "gcn_topk"),
        ("GCN-SAG", "gcn_sag"),
        ("GCN-DiffPool", "gcn_diff"),
    ]
    
    # Train and evaluate each model
    for model_name, model_config_name in model_configs:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # Load model configuration
        model_config = OmegaConf.load(f"configs/model/{model_config_name}.yaml")
        
        # Create model
        model = hydra.utils.instantiate(
            model_config,
            input_dim=dataset.num_node_features,
            num_classes=dataset.num_classes
        )
        
        print(f"Model created:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Size: {get_model_size_mb(model):.2f} MB")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=device
        )
        
        # Train model
        start_time = time.time()
        training_results = trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate model
        print(f"\nEvaluating {model_name}...")
        evaluator = ModelEvaluator(
            model=model,
            test_loader=test_loader,
            device=device,
            num_classes=dataset.num_classes
        )
        
        evaluation_results = evaluator.evaluate()
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"  Test Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"  Test F1-Macro: {evaluation_results['f1_macro']:.4f}")
        print(f"  Test AUROC: {evaluation_results['auroc']:.4f}")
        print(f"  Training Time: {format_time(training_time)}")
        
        # Add to comparison
        model_comparison.add_model_result(
            model_name=model_name,
            model_config=OmegaConf.to_container(model_config),
            evaluation_results=evaluation_results,
            training_time=training_time,
            model_size_mb=get_model_size_mb(model)
        )
        
        # Cleanup trainer
        trainer.cleanup()
    
    # Create and save comparison results
    print(f"\n{'='*60}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*60}")
    
    leaderboard = model_comparison.create_leaderboard()
    print(leaderboard.to_string(index=False))
    
    # Save results
    model_comparison.save_results("model_comparison.json")
    
    # Generate report
    report = model_comparison.generate_report()
    report_path = os.path.join(config.paths.assets_dir, "comparison_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\nComparison report saved to: {report_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = GraphVisualizer(save_dir=os.path.join(config.paths.assets_dir, "visualizations"))
    
    # Plot model comparison
    model_results = {}
    for result in model_comparison.results:
        model_results[result["model_name"]] = result["evaluation_results"]
    
    visualizer.plot_model_comparison(
        model_results,
        metrics=["accuracy", "f1_macro", "auroc"],
        save_path=os.path.join(config.paths.assets_dir, "visualizations", "model_comparison.png")
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
