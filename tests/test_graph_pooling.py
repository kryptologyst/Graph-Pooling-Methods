"""Tests for graph pooling methods."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from src.data import GraphDataset, create_data_loaders
from src.models import GCNWithPooling, GATWithPooling, GraphSAGEWithPooling
from src.layers import GlobalPooling, HierarchicalPooling, AdaptivePooling, MultiScalePooling
from src.utils import set_seed, get_device, count_parameters, get_model_size_mb
from src.eval import ModelEvaluator, ModelComparison


class TestData:
    """Test data utilities."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        x = torch.randn(10, 7)  # 10 nodes, 7 features
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        ], dtype=torch.long)
        y = torch.tensor([0], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return GraphDataset(
            dataset_name="MUTAG",
            dataset_root="data",
            use_synthetic=True,
            synthetic_config={
                "num_graphs": 50,
                "min_nodes": 5,
                "max_nodes": 15,
                "num_classes": 2,
                "edge_prob": 0.3,
                "feature_dim": 7
            },
            seed=42
        )


class TestGraphDataset(TestData):
    """Test GraphDataset functionality."""
    
    def test_dataset_creation(self, sample_dataset):
        """Test dataset creation."""
        assert len(sample_dataset) > 0
        assert sample_dataset.num_node_features > 0
        assert sample_dataset.num_classes > 0
    
    def test_data_splits(self, sample_dataset):
        """Test train/val/test splits."""
        train_data = sample_dataset.get_train_data()
        val_data = sample_dataset.get_val_data()
        test_data = sample_dataset.get_test_data()
        
        assert len(train_data) > 0
        assert len(val_data) > 0
        assert len(test_data) > 0
        assert len(train_data) + len(val_data) + len(test_data) == len(sample_dataset)
    
    def test_data_loader_creation(self, sample_dataset):
        """Test data loader creation."""
        train_loader, val_loader, test_loader = create_data_loaders(
            sample_dataset,
            batch_size=8,
            num_workers=0
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0
    
    def test_graph_properties(self, sample_graph):
        """Test graph properties."""
        assert sample_graph.x.size(0) == 10
        assert sample_graph.x.size(1) == 7
        assert sample_graph.edge_index.size(1) == 20  # Bidirectional edges
        assert sample_graph.y.size(0) == 1


class TestModels:
    """Test model implementations."""
    
    @pytest.fixture
    def model_config(self):
        """Model configuration for testing."""
        return {
            "input_dim": 7,
            "hidden_dim": 32,
            "num_classes": 2,
            "num_layers": 2,
            "dropout": 0.5,
            "use_batch_norm": True,
            "pooling_method": "topk",
            "pooling_ratio": 0.8
        }
    
    def test_gcn_model_creation(self, model_config):
        """Test GCN model creation."""
        model = GCNWithPooling(**model_config)
        
        assert isinstance(model, GCNWithPooling)
        assert count_parameters(model) > 0
        assert get_model_size_mb(model) > 0
    
    def test_gat_model_creation(self, model_config):
        """Test GAT model creation."""
        model_config["pooling_method"] = "sag"
        model = GATWithPooling(**model_config)
        
        assert isinstance(model, GATWithPooling)
        assert count_parameters(model) > 0
    
    def test_graphsage_model_creation(self, model_config):
        """Test GraphSAGE model creation."""
        model_config["pooling_method"] = "global"
        model = GraphSAGEWithPooling(**model_config)
        
        assert isinstance(model, GraphSAGEWithPooling)
        assert count_parameters(model) > 0
    
    def test_model_forward_pass(self, model_config, sample_graph):
        """Test model forward pass."""
        model = GCNWithPooling(**model_config)
        model.eval()
        
        with torch.no_grad():
            output = model(sample_graph.x, sample_graph.edge_index, torch.zeros(10, dtype=torch.long))
        
        assert output.size(0) == 1  # Batch size
        assert output.size(1) == model_config["num_classes"]
        assert torch.allclose(torch.exp(output).sum(dim=1), torch.ones(1), atol=1e-6)  # Probabilities sum to 1


class TestPoolingLayers:
    """Test pooling layer implementations."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for pooling tests."""
        x = torch.randn(20, 16)
        edge_index = torch.randint(0, 20, (2, 40))
        batch = torch.zeros(20, dtype=torch.long)
        
        return x, edge_index, batch
    
    def test_global_pooling(self, sample_data):
        """Test global pooling."""
        x, edge_index, batch = sample_data
        
        # Test different global pooling methods
        for method in ["mean", "max", "sum", "attention"]:
            pooling = GlobalPooling(method=method)
            output = pooling(x, batch)
            
            assert output.size(0) == 1  # Single graph
            assert output.size(1) == x.size(1)
    
    def test_hierarchical_pooling(self, sample_data):
        """Test hierarchical pooling."""
        x, edge_index, batch = sample_data
        
        pooling = HierarchicalPooling(
            input_dim=x.size(1),
            hidden_dim=16,
            pooling_method="topk",
            num_levels=2
        )
        
        pooled_x, pooled_edge_index, pooled_batch = pooling(x, edge_index, batch)
        
        assert pooled_x.size(1) == 16
        assert pooled_x.size(0) <= x.size(0)  # Should reduce nodes
    
    def test_adaptive_pooling(self, sample_data):
        """Test adaptive pooling."""
        x, edge_index, batch = sample_data
        
        pooling = AdaptivePooling(
            input_dim=x.size(1),
            hidden_dim=16
        )
        
        pooled_x, pooled_edge_index, pooled_batch = pooling(x, edge_index, batch)
        
        assert pooled_x.size(1) == 16
        assert pooled_x.size(0) <= x.size(0)
    
    def test_multiscale_pooling(self, sample_data):
        """Test multi-scale pooling."""
        x, edge_index, batch = sample_data
        
        pooling = MultiScalePooling(
            input_dim=x.size(1),
            hidden_dim=16,
            pooling_ratios=[0.8, 0.6, 0.4]
        )
        
        pooled_x, pooled_edge_index, pooled_batch = pooling(x, edge_index, batch)
        
        assert pooled_x.size(1) == 16


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed affects random generation
        torch.manual_seed(42)
        a = torch.randn(10)
        
        set_seed(42)
        torch.manual_seed(42)
        b = torch.randn(10)
        
        assert torch.allclose(a, b)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        # Test specific devices
        cpu_device = get_device("cpu")
        assert cpu_device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        param_count = count_parameters(model)
        
        assert param_count == 55  # 10*5 + 5 bias
    
    def test_get_model_size_mb(self):
        """Test model size calculation."""
        model = torch.nn.Linear(1000, 1000)
        size_mb = get_model_size_mb(model)
        
        assert size_mb > 0
        assert isinstance(size_mb, float)


class TestEvaluation:
    """Test evaluation functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 2)
            
            def forward(self, x, edge_index, batch):
                return torch.log_softmax(self.linear(x.mean(dim=0, keepdim=True)), dim=1)
        
        return MockModel()
    
    @pytest.fixture
    def mock_data_loader(self):
        """Mock data loader for testing."""
        data_list = []
        for _ in range(10):
            x = torch.randn(5, 10)
            edge_index = torch.randint(0, 5, (2, 8))
            y = torch.randint(0, 2, (1,))
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        
        from torch_geometric.loader import DataLoader
        return DataLoader(data_list, batch_size=2)
    
    def test_model_evaluator(self, mock_model, mock_data_loader):
        """Test model evaluator."""
        evaluator = ModelEvaluator(
            model=mock_model,
            test_loader=mock_data_loader,
            num_classes=2
        )
        
        results = evaluator.evaluate()
        
        assert "accuracy" in results
        assert "f1_macro" in results
        assert "auroc" in results
        assert "loss" in results
        
        # Check that metrics are reasonable
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1_macro"] <= 1
        assert 0 <= results["auroc"] <= 1
    
    def test_model_comparison(self):
        """Test model comparison."""
        comparison = ModelComparison()
        
        # Add mock results
        comparison.add_model_result(
            model_name="Test Model",
            model_config={"test": "config"},
            evaluation_results={"accuracy": 0.8, "f1_macro": 0.75, "auroc": 0.85},
            training_time=100.0,
            model_size_mb=5.0
        )
        
        leaderboard = comparison.create_leaderboard()
        
        assert len(leaderboard) == 1
        assert leaderboard.iloc[0]["Model"] == "Test Model"
        assert leaderboard.iloc[0]["Accuracy"] == 0.8


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create small dataset
        dataset = GraphDataset(
            dataset_name="MUTAG",
            dataset_root="data",
            use_synthetic=True,
            synthetic_config={
                "num_graphs": 20,
                "min_nodes": 5,
                "max_nodes": 10,
                "num_classes": 2,
                "edge_prob": 0.3,
                "feature_dim": 7
            },
            seed=42
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset,
            batch_size=4,
            num_workers=0
        )
        
        # Create model
        model = GCNWithPooling(
            input_dim=dataset.num_node_features,
            num_classes=dataset.num_classes,
            pooling_method="topk",
            hidden_dim=16
        )
        
        # Test forward pass
        model.eval()
        for batch in train_loader:
            with torch.no_grad():
                output = model(batch.x, batch.edge_index, batch.batch)
            assert output.size(0) == batch.y.size(0)
            assert output.size(1) == dataset.num_classes
            break  # Just test one batch


if __name__ == "__main__":
    pytest.main([__file__])
