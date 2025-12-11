# Graph Pooling Methods

A comprehensive implementation of graph pooling methods for graph classification tasks. This project provides state-of-the-art graph pooling techniques including TopK, SAGPool, DiffPool, ASAP, and advanced variants like hierarchical, adaptive, and multi-scale pooling.

## Features

- **Multiple Pooling Methods**: TopK, SAGPool, DiffPool, ASAP, Global, Hierarchical, Adaptive, Multi-scale
- **Modern Architecture**: Built with PyTorch Geometric 2.x and Python 3.10+
- **Comprehensive Evaluation**: Extensive metrics and model comparison tools
- **Interactive Demo**: Streamlit-based web application for exploration
- **Production Ready**: Clean code, type hints, comprehensive testing
- **Flexible Configuration**: Hydra-based configuration system
- **Visualization Tools**: Rich visualization capabilities for graphs and results

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Graph-Pooling-Methods.git
cd Graph-Pooling-Methods

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
from src.data import GraphDataset, create_data_loaders
from src.models import GCNWithPooling
from src.train import Trainer

# Load dataset
dataset = GraphDataset(dataset_name="MUTAG", use_synthetic=False)
train_loader, val_loader, test_loader = create_data_loaders(dataset)

# Create model
model = GCNWithPooling(
    input_dim=dataset.num_node_features,
    num_classes=dataset.num_classes,
    pooling_method="topk"
)

# Train model
trainer = Trainer(model, train_loader, val_loader, test_loader, config)
results = trainer.train()
```

### Command Line Training

```bash
# Train with default configuration
python scripts/train.py

# Train with custom configuration
python scripts/train.py model=gcn_sag data.dataset_name=PROTEINS

# Train with synthetic data
python scripts/train.py data.use_synthetic=true
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py
```

## Project Structure

```
graph-pooling-methods/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   ├── layers/                   # Custom pooling layers
│   ├── data/                     # Data loading and preprocessing
│   ├── train/                    # Training utilities
│   ├── eval/                     # Evaluation metrics
│   └── utils/                    # Utility functions
├── configs/                      # Configuration files
│   ├── model/                    # Model configurations
│   ├── data/                     # Data configurations
│   └── training/                 # Training configurations
├── scripts/                      # Training and evaluation scripts
├── demo/                         # Interactive demo application
├── tests/                        # Unit tests
├── assets/                       # Generated results and visualizations
└── data/                         # Dataset storage
```

## Supported Pooling Methods

### Traditional Methods

1. **TopK Pooling**: Selects top-k nodes based on learned scores
2. **SAG Pooling**: Self-attention based pooling with learnable attention weights
3. **DiffPool**: Differentiable pooling with cluster assignment
4. **ASAP**: Adaptive structure-aware pooling

### Advanced Methods

1. **Global Pooling**: Mean, max, sum, and attention-based global aggregation
2. **Hierarchical Pooling**: Multi-level pooling with residual connections
3. **Adaptive Pooling**: Learns to select optimal pooling strategy
4. **Multi-scale Pooling**: Combines multiple pooling ratios

## Model Architectures

- **GCN**: Graph Convolutional Networks with various pooling methods
- **GAT**: Graph Attention Networks with attention-based pooling
- **GraphSAGE**: GraphSAGE with inductive pooling capabilities

## Datasets

### Real Datasets
- MUTAG: Molecular graphs for mutagenicity prediction
- PROTEINS: Protein graphs for protein function prediction
- NCI1/NCI109: Chemical compounds for cancer activity prediction
- ENZYMES: Protein structures for enzyme classification
- IMDB-BINARY/IMDB-MULTI: Movie collaboration networks

### Synthetic Datasets
- Configurable synthetic graph generation
- Control over graph size, connectivity, and features
- Useful for testing and demonstration

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/`: Model-specific configurations
- `configs/data/`: Dataset configurations
- `configs/training/`: Training configurations

### Example Configuration

```yaml
# config.yaml
defaults:
  - model: gcn_topk
  - data: mutag
  - training: default

experiment:
  name: "graph_pooling_experiment"
  tags: ["pooling", "graph_classification"]

device: "auto"
seed: 42

paths:
  data_dir: "data"
  checkpoint_dir: "checkpoints"
  log_dir: "logs"
  assets_dir: "assets"
```

## Evaluation Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score (macro/micro/weighted)
- Area Under ROC Curve (AUROC)
- Confusion Matrix and Classification Report

### Model Comparison
- Comprehensive leaderboard with multiple metrics
- Training time and model size comparison
- Statistical significance testing

### Visualization
- Training curves and loss plots
- Model comparison charts
- Graph visualizations (static and interactive)
- Attention weight visualizations
- Feature importance plots

## Advanced Features

### Logging and Monitoring
- Weights & Biases integration
- TensorBoard support
- Comprehensive logging system

### Checkpointing
- Automatic model checkpointing
- Best model selection
- Resume training capability

### Early Stopping
- Configurable patience and monitoring metrics
- Best weights restoration

### Device Support
- Automatic device detection (CUDA/MPS/CPU)
- Cross-platform compatibility

## Development

### Code Quality
- Type hints throughout
- Comprehensive docstrings
- Black code formatting
- Ruff linting
- MyPy type checking

### Testing
```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Performance Considerations

### Memory Optimization
- Efficient data loading with proper batching
- Gradient accumulation for large models
- Mixed precision training support

### Scalability
- Neighbor sampling for large graphs
- Distributed training support
- Model parallelism capabilities

## Ethical Considerations

### Privacy
- No personal data collection
- Synthetic data generation for demonstrations
- Secure data handling practices

### Bias and Fairness
- Comprehensive evaluation across different graph types
- Fairness metrics for model comparison
- Transparent reporting of limitations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{graph_pooling_methods,
  title={Graph Pooling Methods: A Comprehensive Implementation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Graph-Pooling-Methods}
}
```

## Acknowledgments

- PyTorch Geometric team for the excellent graph neural network framework
- The graph neural network research community
- Contributors and users of this project

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation
- Review the examples in the demo application

## Changelog

### Version 0.1.0
- Initial release
- Support for multiple pooling methods
- Comprehensive evaluation framework
- Interactive demo application
- Modern codebase with type hints and testing
# Graph-Pooling-Methods
