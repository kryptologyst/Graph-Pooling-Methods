"""Graph pooling layer implementations."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    TopKPooling, SAGPooling, DiffPool, ASAPooling
)
from torch_geometric.utils import softmax, to_dense_batch


class GlobalPooling(nn.Module):
    """Global pooling layer with multiple aggregation methods."""
    
    def __init__(self, method: str = "mean"):
        """Initialize global pooling.
        
        Args:
            method: Pooling method ("mean", "max", "sum", "attention").
        """
        super().__init__()
        self.method = method
        
        if method == "attention":
            self.attention_net = nn.Sequential(
                nn.Linear(1, 1),
                nn.Tanh(),
                nn.Linear(1, 1)
            )
    
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            batch: Batch assignment for each node.
            
        Returns:
            Graph-level representation.
        """
        if self.method == "mean":
            return global_mean_pool(x, batch)
        elif self.method == "max":
            return global_max_pool(x, batch)
        elif self.method == "sum":
            return global_add_pool(x, batch)
        elif self.method == "attention":
            return self._attention_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.method}")
    
    def _attention_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Attention-based global pooling.
        
        Args:
            x: Node features.
            batch: Batch assignment for each node.
            
        Returns:
            Graph-level representation.
        """
        # Compute attention weights
        attention_scores = self.attention_net(x.mean(dim=1, keepdim=True))
        attention_weights = softmax(attention_scores, batch)
        
        # Weighted sum
        weighted_x = x * attention_weights
        return global_add_pool(weighted_x, batch)


class HierarchicalPooling(nn.Module):
    """Hierarchical pooling with multiple levels."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pooling_method: str = "topk",
        pooling_ratio: float = 0.8,
        num_levels: int = 2,
        use_residual: bool = True
    ):
        """Initialize hierarchical pooling.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            pooling_method: Pooling method ("topk", "sag", "diff", "asap").
            pooling_ratio: Pooling ratio.
            num_levels: Number of pooling levels.
            use_residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pooling_method = pooling_method
        self.pooling_ratio = pooling_ratio
        self.num_levels = num_levels
        self.use_residual = use_residual
        
        # Create pooling layers
        self.pooling_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        current_dim = input_dim
        for i in range(num_levels):
            # Convolution layer
            conv = GCNConv(current_dim, hidden_dim)
            self.conv_layers.append(conv)
            
            # Pooling layer
            if pooling_method == "topk":
                pool = TopKPooling(hidden_dim, ratio=pooling_ratio)
            elif pooling_method == "sag":
                pool = SAGPooling(hidden_dim, ratio=pooling_ratio)
            elif pooling_method == "diff":
                pool = DiffPool(hidden_dim, hidden_dim, num_clusters=10)
            elif pooling_method == "asap":
                pool = ASAPooling(hidden_dim, ratio=pooling_ratio)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")
            
            self.pooling_layers.append(pool)
            current_dim = hidden_dim
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment for each node.
            
        Returns:
            Tuple of (final_features, final_edge_index, final_batch).
        """
        current_x = x
        current_edge_index = edge_index
        current_batch = batch
        
        for i in range(self.num_levels):
            # Convolution
            current_x = F.relu(self.conv_layers[i](current_x, current_edge_index))
            
            # Pooling
            if self.pooling_method == "topk":
                current_x, current_edge_index, _, current_batch, _, _ = self.pooling_layers[i](
                    current_x, current_edge_index, batch=current_batch
                )
            elif self.pooling_method == "sag":
                current_x, current_edge_index, _, current_batch, _, _ = self.pooling_layers[i](
                    current_x, current_edge_index, batch=current_batch
                )
            elif self.pooling_method == "diff":
                current_x, current_edge_index, current_batch = self.pooling_layers[i](
                    current_x, current_edge_index, batch=current_batch
                )
            elif self.pooling_method == "asap":
                current_x, current_edge_index, _, current_batch, _, _ = self.pooling_layers[i](
                    current_x, current_edge_index, batch=current_batch
                )
        
        return current_x, current_edge_index, current_batch


class AdaptivePooling(nn.Module):
    """Adaptive pooling that learns the pooling strategy."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_pooling_methods: int = 4,
        temperature: float = 1.0
    ):
        """Initialize adaptive pooling.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            num_pooling_methods: Number of pooling methods to choose from.
            temperature: Temperature for softmax.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_pooling_methods = num_pooling_methods
        self.temperature = temperature
        
        # Method selector
        self.method_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_pooling_methods)
        )
        
        # Pooling methods
        self.pooling_methods = nn.ModuleList([
            TopKPooling(input_dim, ratio=0.8),
            SAGPooling(input_dim, ratio=0.8),
            DiffPool(input_dim, hidden_dim, num_clusters=10),
            ASAPooling(input_dim, ratio=0.8)
        ])
        
        # Feature transformation
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment for each node.
            
        Returns:
            Tuple of (pooled_features, pooled_edge_index, pooled_batch).
        """
        # Compute method selection weights
        graph_features = global_mean_pool(x, batch)
        method_weights = F.softmax(
            self.method_selector(graph_features) / self.temperature, 
            dim=1
        )
        
        # Apply each pooling method
        pooled_results = []
        for i, pooling_method in enumerate(self.pooling_methods):
            if isinstance(pooling_method, TopKPooling):
                pooled_x, pooled_edge_index, _, pooled_batch, _, _ = pooling_method(
                    x, edge_index, batch=batch
                )
            elif isinstance(pooling_method, SAGPooling):
                pooled_x, pooled_edge_index, _, pooled_batch, _, _ = pooling_method(
                    x, edge_index, batch=batch
                )
            elif isinstance(pooling_method, DiffPool):
                pooled_x, pooled_edge_index, pooled_batch = pooling_method(
                    x, edge_index, batch=batch
                )
            elif isinstance(pooling_method, ASAPooling):
                pooled_x, pooled_edge_index, _, pooled_batch, _, _ = pooling_method(
                    x, edge_index, batch=batch
                )
            
            pooled_results.append((pooled_x, pooled_edge_index, pooled_batch))
        
        # Weighted combination
        final_x = torch.zeros_like(pooled_results[0][0])
        final_edge_index = pooled_results[0][1]
        final_batch = pooled_results[0][2]
        
        for i, (pooled_x, _, _) in enumerate(pooled_results):
            weight = method_weights[:, i:i+1].expand_as(pooled_x)
            final_x += weight * pooled_x
        
        return final_x, final_edge_index, final_batch


class MultiScalePooling(nn.Module):
    """Multi-scale pooling that combines different pooling ratios."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        pooling_ratios: list = [0.8, 0.6, 0.4],
        pooling_method: str = "topk"
    ):
        """Initialize multi-scale pooling.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            pooling_ratios: List of pooling ratios.
            pooling_method: Pooling method.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pooling_ratios = pooling_ratios
        self.pooling_method = pooling_method
        
        # Create pooling layers for different scales
        self.pooling_layers = nn.ModuleList()
        for ratio in pooling_ratios:
            if pooling_method == "topk":
                pool = TopKPooling(input_dim, ratio=ratio)
            elif pooling_method == "sag":
                pool = SAGPooling(input_dim, ratio=ratio)
            elif pooling_method == "asap":
                pool = ASAPooling(input_dim, ratio=ratio)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")
            
            self.pooling_layers.append(pool)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * len(pooling_ratios), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment for each node.
            
        Returns:
            Tuple of (fused_features, edge_index, batch).
        """
        pooled_features = []
        
        for pooling_layer in self.pooling_layers:
            if self.pooling_method == "topk":
                pooled_x, _, _, _, _, _ = pooling_layer(x, edge_index, batch=batch)
            elif self.pooling_method == "sag":
                pooled_x, _, _, _, _, _ = pooling_layer(x, edge_index, batch=batch)
            elif self.pooling_method == "asap":
                pooled_x, _, _, _, _, _ = pooling_layer(x, edge_index, batch=batch)
            
            pooled_features.append(pooled_x)
        
        # Concatenate features from different scales
        concatenated_features = torch.cat(pooled_features, dim=1)
        
        # Fuse features
        fused_features = self.fusion(concatenated_features)
        
        return fused_features, edge_index, batch
