"""Graph neural network models with various pooling methods."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import softmax

from ..layers import (
    GlobalPooling, HierarchicalPooling, AdaptivePooling, MultiScalePooling
)


class GCNWithPooling(nn.Module):
    """GCN-based model with various pooling methods for graph classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        pooling_method: str = "topk",
        pooling_ratio: float = 0.8,
        pooling_layers: int = 2,
        use_residual: bool = True,
        **kwargs
    ):
        """Initialize GCN with pooling.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            num_classes: Number of output classes.
            num_layers: Number of GCN layers.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
            pooling_method: Pooling method ("topk", "sag", "diff", "asap", "global", "hierarchical", "adaptive", "multiscale").
            pooling_ratio: Pooling ratio for hierarchical pooling.
            pooling_layers: Number of pooling layers.
            use_residual: Whether to use residual connections.
            **kwargs: Additional arguments.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling_method = pooling_method
        self.pooling_ratio = pooling_ratio
        self.pooling_layers = pooling_layers
        self.use_residual = use_residual
        
        # GCN layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer
        if num_layers > 1:
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling layers
        self._setup_pooling()
        
        # Classifier
        self._setup_classifier()
    
    def _setup_pooling(self) -> None:
        """Setup pooling layers based on the pooling method."""
        if self.pooling_method == "global":
            self.pooling = GlobalPooling(method="mean")
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "hierarchical":
            self.pooling = HierarchicalPooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                pooling_method="topk",
                pooling_ratio=self.pooling_ratio,
                num_levels=self.pooling_layers,
                use_residual=self.use_residual
            )
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "adaptive":
            self.pooling = AdaptivePooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim
            )
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "multiscale":
            self.pooling = MultiScalePooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                pooling_ratios=[0.8, 0.6, 0.4],
                pooling_method="topk"
            )
            self.pooling_dim = self.hidden_dim
        else:
            # Traditional pooling methods (topk, sag, diff, asap)
            self.pooling_dim = self.hidden_dim
    
    def _setup_classifier(self) -> None:
        """Setup the final classifier."""
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment for each node.
            
        Returns:
            Log probabilities for each class.
        """
        # Apply GCN layers
        for i, conv in enumerate(self.conv_layers):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size(1) == residual.size(1):
                x = x + residual
        
        # Apply pooling
        if self.pooling_method == "global":
            graph_repr = self.pooling(x, batch)
        elif self.pooling_method in ["hierarchical", "adaptive", "multiscale"]:
            pooled_x, pooled_edge_index, pooled_batch = self.pooling(x, edge_index, batch)
            graph_repr = global_mean_pool(pooled_x, pooled_batch)
        else:
            # Traditional pooling methods
            graph_repr = global_mean_pool(x, batch)
        
        # Classify
        logits = self.classifier(graph_repr)
        return F.log_softmax(logits, dim=1)


class GATWithPooling(nn.Module):
    """GAT-based model with various pooling methods for graph classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        pooling_method: str = "topk",
        pooling_ratio: float = 0.8,
        pooling_layers: int = 2,
        use_residual: bool = True,
        **kwargs
    ):
        """Initialize GAT with pooling.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            num_classes: Number of output classes.
            num_layers: Number of GAT layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
            pooling_method: Pooling method.
            pooling_ratio: Pooling ratio for hierarchical pooling.
            pooling_layers: Number of pooling layers.
            use_residual: Whether to use residual connections.
            **kwargs: Additional arguments.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling_method = pooling_method
        self.pooling_ratio = pooling_ratio
        self.pooling_layers = pooling_layers
        self.use_residual = use_residual
        
        # GAT layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Last layer
        if num_layers > 1:
            self.conv_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling layers
        self._setup_pooling()
        
        # Classifier
        self._setup_classifier()
    
    def _setup_pooling(self) -> None:
        """Setup pooling layers based on the pooling method."""
        if self.pooling_method == "global":
            self.pooling = GlobalPooling(method="attention")
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "hierarchical":
            self.pooling = HierarchicalPooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                pooling_method="sag",
                pooling_ratio=self.pooling_ratio,
                num_levels=self.pooling_layers,
                use_residual=self.use_residual
            )
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "adaptive":
            self.pooling = AdaptivePooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim
            )
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "multiscale":
            self.pooling = MultiScalePooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                pooling_ratios=[0.8, 0.6, 0.4],
                pooling_method="sag"
            )
            self.pooling_dim = self.hidden_dim
        else:
            # Traditional pooling methods
            self.pooling_dim = self.hidden_dim
    
    def _setup_classifier(self) -> None:
        """Setup the final classifier."""
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment for each node.
            
        Returns:
            Log probabilities for each class.
        """
        # Apply GAT layers
        for i, conv in enumerate(self.conv_layers):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size(1) == residual.size(1):
                x = x + residual
        
        # Apply pooling
        if self.pooling_method == "global":
            graph_repr = self.pooling(x, batch)
        elif self.pooling_method in ["hierarchical", "adaptive", "multiscale"]:
            pooled_x, pooled_edge_index, pooled_batch = self.pooling(x, edge_index, batch)
            graph_repr = global_mean_pool(pooled_x, pooled_batch)
        else:
            # Traditional pooling methods
            graph_repr = global_mean_pool(x, batch)
        
        # Classify
        logits = self.classifier(graph_repr)
        return F.log_softmax(logits, dim=1)


class GraphSAGEWithPooling(nn.Module):
    """GraphSAGE-based model with various pooling methods for graph classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        pooling_method: str = "topk",
        pooling_ratio: float = 0.8,
        pooling_layers: int = 2,
        use_residual: bool = True,
        **kwargs
    ):
        """Initialize GraphSAGE with pooling.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden feature dimension.
            num_classes: Number of output classes.
            num_layers: Number of GraphSAGE layers.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
            pooling_method: Pooling method.
            pooling_ratio: Pooling ratio for hierarchical pooling.
            pooling_layers: Number of pooling layers.
            use_residual: Whether to use residual connections.
            **kwargs: Additional arguments.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.pooling_method = pooling_method
        self.pooling_ratio = pooling_ratio
        self.pooling_layers = pooling_layers
        self.use_residual = use_residual
        
        # GraphSAGE layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(SAGEConv(input_dim, hidden_dim))
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Last layer
        if num_layers > 1:
            self.conv_layers.append(SAGEConv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Pooling layers
        self._setup_pooling()
        
        # Classifier
        self._setup_classifier()
    
    def _setup_pooling(self) -> None:
        """Setup pooling layers based on the pooling method."""
        if self.pooling_method == "global":
            self.pooling = GlobalPooling(method="max")
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "hierarchical":
            self.pooling = HierarchicalPooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                pooling_method="asap",
                pooling_ratio=self.pooling_ratio,
                num_levels=self.pooling_layers,
                use_residual=self.use_residual
            )
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "adaptive":
            self.pooling = AdaptivePooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim
            )
            self.pooling_dim = self.hidden_dim
        elif self.pooling_method == "multiscale":
            self.pooling = MultiScalePooling(
                input_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                pooling_ratios=[0.8, 0.6, 0.4],
                pooling_method="asap"
            )
            self.pooling_dim = self.hidden_dim
        else:
            # Traditional pooling methods
            self.pooling_dim = self.hidden_dim
    
    def _setup_classifier(self) -> None:
        """Setup the final classifier."""
        self.classifier = nn.Sequential(
            nn.Linear(self.pooling_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            batch: Batch assignment for each node.
            
        Returns:
            Log probabilities for each class.
        """
        # Apply GraphSAGE layers
        for i, conv in enumerate(self.conv_layers):
            residual = x if self.use_residual and i > 0 else None
            
            x = conv(x, edge_index)
            
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if residual is not None and x.size(1) == residual.size(1):
                x = x + residual
        
        # Apply pooling
        if self.pooling_method == "global":
            graph_repr = self.pooling(x, batch)
        elif self.pooling_method in ["hierarchical", "adaptive", "multiscale"]:
            pooled_x, pooled_edge_index, pooled_batch = self.pooling(x, edge_index, batch)
            graph_repr = global_mean_pool(pooled_x, pooled_batch)
        else:
            # Traditional pooling methods
            graph_repr = global_mean_pool(x, batch)
        
        # Classify
        logits = self.classifier(graph_repr)
        return F.log_softmax(logits, dim=1)
