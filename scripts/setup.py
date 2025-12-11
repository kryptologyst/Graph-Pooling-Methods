#!/usr/bin/env python3
"""Quick start script for graph pooling methods."""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def run_quick_test():
    """Run a quick test to verify installation."""
    print("Running quick test...")
    try:
        # Test imports
        from src.data import GraphDataset
        from src.models import GCNWithPooling
        from src.utils import set_seed, get_device
        
        # Create a small test dataset
        dataset = GraphDataset(
            dataset_name="MUTAG",
            dataset_root="data",
            use_synthetic=True,
            synthetic_config={
                "num_graphs": 10,
                "min_nodes": 5,
                "max_nodes": 10,
                "num_classes": 2,
                "edge_prob": 0.3,
                "feature_dim": 7
            },
            seed=42
        )
        
        # Create a simple model
        model = GCNWithPooling(
            input_dim=dataset.num_node_features,
            num_classes=dataset.num_classes,
            pooling_method="topk"
        )
        
        print("✓ Quick test passed")
        return True
        
    except Exception as e:
        print(f"Error in quick test: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ["data", "checkpoints", "logs", "assets", "assets/visualizations"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main setup function."""
    print("=" * 60)
    print("Graph Pooling Methods - Quick Start")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Install dependencies
    print("\nInstalling dependencies...")
    install_dependencies()
    
    # Run quick test
    print("\nRunning quick test...")
    if run_quick_test():
        print("\n" + "=" * 60)
        print("✓ Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run training: python scripts/train.py")
        print("2. Launch demo: streamlit run demo/streamlit_app.py")
        print("3. Run tests: pytest tests/")
        print("\nFor more information, see README.md")
    else:
        print("\n" + "=" * 60)
        print("✗ Setup completed with errors")
        print("=" * 60)
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
