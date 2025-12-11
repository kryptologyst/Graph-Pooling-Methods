"""Visualization utilities for graph pooling methods."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from ..utils import get_device


class GraphVisualizer:
    """Visualization utilities for graph data and pooling results."""
    
    def __init__(self, save_dir: str = "assets/visualizations"):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")
    
    def visualize_graph(
        self,
        data: Data,
        node_colors: Optional[List[str]] = None,
        edge_colors: Optional[List[str]] = None,
        node_labels: Optional[List[str]] = None,
        title: str = "Graph Visualization",
        save_path: Optional[str] = None,
        layout: str = "spring"
    ) -> None:
        """Visualize a single graph.
        
        Args:
            data: PyTorch Geometric Data object.
            node_colors: Colors for nodes.
            edge_colors: Colors for edges.
            node_labels: Labels for nodes.
            title: Title of the plot.
            save_path: Path to save the plot.
            layout: Layout algorithm ("spring", "circular", "random").
        """
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        plt.figure(figsize=(10, 8))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors or "lightblue",
            node_size=500,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors or "gray",
            alpha=0.6,
            width=1
        )
        
        # Draw labels
        if node_labels:
            labels = {i: node_labels[i] for i in range(len(node_labels))}
            nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title(title)
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def visualize_pooling_comparison(
        self,
        original_data: Data,
        pooled_data: Data,
        pooling_method: str,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize before and after pooling.
        
        Args:
            original_data: Original graph data.
            pooled_data: Pooled graph data.
            pooling_method: Name of pooling method.
            save_path: Path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original graph
        G_orig = to_networkx(original_data, to_undirected=True)
        pos_orig = nx.spring_layout(G_orig, k=1, iterations=50)
        
        nx.draw_networkx_nodes(G_orig, pos_orig, node_color="lightblue", node_size=300, ax=ax1)
        nx.draw_networkx_edges(G_orig, pos_orig, alpha=0.6, ax=ax1)
        ax1.set_title(f"Original Graph ({G_orig.number_of_nodes()} nodes)")
        ax1.axis("off")
        
        # Pooled graph
        G_pooled = to_networkx(pooled_data, to_undirected=True)
        pos_pooled = nx.spring_layout(G_pooled, k=1, iterations=50)
        
        nx.draw_networkx_nodes(G_pooled, pos_pooled, node_color="lightcoral", node_size=300, ax=ax2)
        nx.draw_networkx_edges(G_pooled, pos_pooled, alpha=0.6, ax=ax2)
        ax2.set_title(f"After {pooling_method} ({G_pooled.number_of_nodes()} nodes)")
        ax2.axis("off")
        
        plt.suptitle(f"Graph Pooling Comparison: {pooling_method}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accuracies: List[float],
        val_accuracies: List[float],
        save_path: Optional[str] = None
    ) -> None:
        """Plot training curves.
        
        Args:
            train_losses: Training losses.
            val_losses: Validation losses.
            train_accuracies: Training accuracies.
            val_accuracies: Validation accuracies.
            save_path: Path to save the plot.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, label="Training Loss", color="blue")
        ax1.plot(epochs, val_losses, label="Validation Loss", color="red")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, train_accuracies, label="Training Accuracy", color="blue")
        ax2.plot(epochs, val_accuracies, label="Validation Accuracy", color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_model_comparison(
        self,
        model_results: Dict[str, Dict[str, float]],
        metrics: List[str] = ["accuracy", "f1_macro", "auroc"],
        save_path: Optional[str] = None
    ) -> None:
        """Plot model comparison.
        
        Args:
            model_results: Dictionary of model results.
            metrics: List of metrics to plot.
            save_path: Path to save the plot.
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        model_names = list(model_results.keys())
        
        for i, metric in enumerate(metrics):
            values = [model_results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f"{metric.replace('_', ' ').title()} Comparison")
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis="x", rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f"{value:.3f}", ha="center", va="bottom")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def create_interactive_graph(
        self,
        data: Data,
        node_colors: Optional[List[str]] = None,
        node_labels: Optional[List[str]] = None,
        title: str = "Interactive Graph",
        save_path: Optional[str] = None
    ) -> None:
        """Create interactive graph visualization using PyVis.
        
        Args:
            data: PyTorch Geometric Data object.
            node_colors: Colors for nodes.
            node_labels: Labels for nodes.
            title: Title of the graph.
            save_path: Path to save the HTML file.
        """
        # Convert to NetworkX
        G = to_networkx(data, to_undirected=True)
        
        # Create PyVis network
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        
        # Add nodes
        for node in G.nodes():
            label = node_labels[node] if node_labels and node < len(node_labels) else str(node)
            color = node_colors[node] if node_colors and node < len(node_colors) else "#97c2fc"
            
            net.add_node(node, label=label, color=color, size=20)
        
        # Add edges
        for edge in G.edges():
            net.add_edge(edge[0], edge[1], width=2)
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        # Save or show
        if save_path:
            net.save_graph(save_path)
            print(f"Interactive graph saved to: {save_path}")
        else:
            net.show("graph.html")
    
    def plot_attention_weights(
        self,
        attention_weights: torch.Tensor,
        edge_index: torch.Tensor,
        title: str = "Attention Weights",
        save_path: Optional[str] = None
    ) -> None:
        """Plot attention weights on edges.
        
        Args:
            attention_weights: Attention weights tensor.
            edge_index: Edge indices.
            title: Title of the plot.
            save_path: Path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        
        # Create edge list with weights
        edges = edge_index.t().cpu().numpy()
        weights = attention_weights.cpu().numpy()
        
        # Create scatter plot
        scatter = plt.scatter(edges[:, 0], edges[:, 1], c=weights, cmap="viridis", s=100, alpha=0.7)
        
        plt.colorbar(scatter, label="Attention Weight")
        plt.xlabel("Source Node")
        plt.ylabel("Target Node")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_importance: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> None:
        """Plot feature importance.
        
        Args:
            feature_importance: Feature importance scores.
            feature_names: Names of features.
            title: Title of the plot.
            save_path: Path to save the plot.
        """
        plt.figure(figsize=(10, 6))
        
        importance_scores = feature_importance.cpu().numpy()
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance_scores))]
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_scores = importance_scores[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create bar plot
        bars = plt.bar(range(len(sorted_scores)), sorted_scores, alpha=0.7)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title(title)
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha="right")
        
        # Add value labels on bars
        for bar, score in zip(bars, sorted_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{score:.3f}", ha="center", va="bottom")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.show()
    
    def plot_embedding_2d(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        title: str = "2D Embedding Visualization",
        save_path: Optional[str] = None
    ) -> None:
        """Plot 2D embeddings using t-SNE or UMAP.
        
        Args:
            embeddings: Node or graph embeddings.
            labels: Labels for coloring.
            title: Title of the plot.
            save_path: Path to save the plot.
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            
            # Reduce dimensionality
            if embeddings.size(1) > 2:
                # First reduce with PCA if too many dimensions
                if embeddings.size(1) > 50:
                    pca = PCA(n_components=50)
                    embeddings_reduced = pca.fit_transform(embeddings.cpu().numpy())
                else:
                    embeddings_reduced = embeddings.cpu().numpy()
                
                # Then use t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings_reduced)
            else:
                embeddings_2d = embeddings.cpu().numpy()
            
            # Plot
            plt.figure(figsize=(10, 8))
            
            unique_labels = torch.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                           c=[colors[i]], label=f"Class {label.item()}", alpha=0.7, s=50)
            
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
            plt.show()
            
        except ImportError:
            print("scikit-learn not available for t-SNE visualization")
            # Fallback to PCA
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
            
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels.cpu().numpy(), alpha=0.7)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title(f"{title} (PCA)")
            plt.colorbar()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            
            plt.show()
