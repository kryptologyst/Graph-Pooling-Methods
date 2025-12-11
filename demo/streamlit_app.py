"""Streamlit demo application for graph pooling methods."""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data import GraphDataset, create_data_loaders
from models import GCNWithPooling, GATWithPooling, GraphSAGEWithPooling
from utils import set_seed, get_device
from utils.visualization import GraphVisualizer


def load_sample_data() -> Tuple[GraphDataset, DataLoader, DataLoader, DataLoader]:
    """Load sample data for demonstration.
    
    Returns:
        Tuple of (dataset, train_loader, val_loader, test_loader).
    """
    # Create synthetic dataset for demo
    dataset = GraphDataset(
        dataset_name="MUTAG",
        dataset_root="data",
        use_synthetic=True,
        synthetic_config={
            "num_graphs": 200,
            "min_nodes": 10,
            "max_nodes": 30,
            "num_classes": 2,
            "edge_prob": 0.3,
            "feature_dim": 7
        },
        seed=42
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset,
        batch_size=16,
        num_workers=0,
        pin_memory=False
    )
    
    return dataset, train_loader, val_loader, test_loader


def create_model(model_type: str, pooling_method: str, input_dim: int, num_classes: int) -> torch.nn.Module:
    """Create a model based on the specified type and pooling method.
    
    Args:
        model_type: Type of model ("GCN", "GAT", "GraphSAGE").
        pooling_method: Pooling method to use.
        input_dim: Input feature dimension.
        num_classes: Number of classes.
        
    Returns:
        PyTorch model.
    """
    if model_type == "GCN":
        return GCNWithPooling(
            input_dim=input_dim,
            hidden_dim=64,
            num_classes=num_classes,
            pooling_method=pooling_method,
            pooling_ratio=0.8
        )
    elif model_type == "GAT":
        return GATWithPooling(
            input_dim=input_dim,
            hidden_dim=64,
            num_classes=num_classes,
            pooling_method=pooling_method,
            pooling_ratio=0.8
        )
    elif model_type == "GraphSAGE":
        return GraphSAGEWithPooling(
            input_dim=input_dim,
            hidden_dim=64,
            num_classes=num_classes,
            pooling_method=pooling_method,
            pooling_ratio=0.8
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def visualize_graph_interactive(data: Data, title: str = "Graph") -> go.Figure:
    """Create interactive graph visualization using Plotly.
    
    Args:
        data: PyTorch Geometric Data object.
        title: Title of the graph.
        
    Returns:
        Plotly figure.
    """
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)
    
    # Get layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract node and edge information
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"Node {i}" for i in G.nodes()],
        textposition="middle center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=True,
            color=[],
            size=20,
            colorbar=dict(
                thickness=15,
                xanchor="left",
                titleside="right"
            ),
            line=dict(width=2)
        )
    )
    
    # Color nodes by degree
    node_adjacencies = []
    node_text = []
    for node in G.nodes():
        adjacencies = list(G.neighbors(node))
        node_adjacencies.append(len(adjacencies))
        node_text.append(f"Node {node}<br>Degree: {len(adjacencies)}")
    
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Interactive graph visualization",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='black', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Graph Pooling Methods Demo",
        page_icon="🕸️",
        layout="wide"
    )
    
    st.title("🕸️ Graph Pooling Methods Demo")
    st.markdown("Interactive demonstration of different graph pooling methods for graph classification")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["GCN", "GAT", "GraphSAGE"],
        help="Choose the base graph neural network architecture"
    )
    
    pooling_method = st.sidebar.selectbox(
        "Pooling Method",
        ["topk", "sag", "diff", "asap", "global", "hierarchical", "adaptive", "multiscale"],
        help="Choose the graph pooling method"
    )
    
    # Load data
    with st.spinner("Loading sample data..."):
        dataset, train_loader, val_loader, test_loader = load_sample_data()
    
    st.sidebar.success(f"Loaded {len(dataset)} graphs")
    
    # Dataset information
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"**Total Graphs:** {len(dataset)}")
    st.sidebar.write(f"**Node Features:** {dataset.num_node_features}")
    st.sidebar.write(f"**Classes:** {dataset.num_classes}")
    st.sidebar.write(f"**Train/Val/Test:** {len(dataset.train_indices)}/{len(dataset.val_indices)}/{len(dataset.test_indices)}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "🔍 Graph Visualization", "🤖 Model Training", "📈 Results"])
    
    with tab1:
        st.header("Dataset Overview")
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Graphs", len(dataset))
            st.metric("Node Features", dataset.num_node_features)
        
        with col2:
            st.metric("Classes", dataset.num_classes)
            st.metric("Train Graphs", len(dataset.train_indices))
        
        with col3:
            st.metric("Val Graphs", len(dataset.val_indices))
            st.metric("Test Graphs", len(dataset.test_indices))
        
        # Sample graphs
        st.subheader("Sample Graphs")
        
        # Show first few graphs
        sample_graphs = [dataset[i] for i in range(min(5, len(dataset)))]
        
        for i, graph in enumerate(sample_graphs):
            with st.expander(f"Graph {i+1} (Label: {graph.y.item()})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Nodes:** {graph.x.size(0)}")
                    st.write(f"**Edges:** {graph.edge_index.size(1) // 2}")
                    st.write(f"**Features:** {graph.x.size(1)}")
                
                with col2:
                    # Create interactive visualization
                    fig = visualize_graph_interactive(graph, f"Graph {i+1}")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Graph Visualization")
        
        # Graph selection
        graph_idx = st.selectbox(
            "Select Graph to Visualize",
            range(len(dataset)),
            format_func=lambda x: f"Graph {x+1} (Label: {dataset[x].y.item()})"
        )
        
        selected_graph = dataset[graph_idx]
        
        # Visualization options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Graph Properties")
            st.write(f"**Nodes:** {selected_graph.x.size(0)}")
            st.write(f"**Edges:** {selected_graph.edge_index.size(1) // 2}")
            st.write(f"**Label:** {selected_graph.y.item()}")
            st.write(f"**Features:** {selected_graph.x.size(1)}")
            
            # Node features
            st.subheader("Node Features")
            feature_df = pd.DataFrame(
                selected_graph.x.numpy(),
                columns=[f"Feature {i}" for i in range(selected_graph.x.size(1))]
            )
            st.dataframe(feature_df)
        
        with col2:
            st.subheader("Interactive Visualization")
            fig = visualize_graph_interactive(selected_graph, f"Graph {graph_idx+1}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Adjacency matrix
        st.subheader("Adjacency Matrix")
        G = to_networkx(selected_graph, to_undirected=True)
        adj_matrix = nx.adjacency_matrix(G).todense()
        
        fig = px.imshow(
            adj_matrix,
            title="Adjacency Matrix",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Model Training")
        
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {model_type}")
            st.write(f"**Pooling Method:** {pooling_method}")
            st.write(f"**Input Dimension:** {dataset.num_node_features}")
            st.write(f"**Output Classes:** {dataset.num_classes}")
        
        with col2:
            hidden_dim = st.slider("Hidden Dimension", 32, 128, 64)
            num_layers = st.slider("Number of Layers", 1, 4, 2)
            dropout = st.slider("Dropout Rate", 0.0, 0.8, 0.5)
        
        # Training parameters
        st.subheader("Training Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            learning_rate = st.number_input("Learning Rate", 0.001, 0.1, 0.01, 0.001)
            weight_decay = st.number_input("Weight Decay", 0.0, 0.01, 0.0005, 0.0001)
        
        with col2:
            max_epochs = st.number_input("Max Epochs", 10, 200, 50, 10)
            batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        
        with col3:
            patience = st.number_input("Early Stopping Patience", 5, 50, 10, 5)
            use_wandb = st.checkbox("Use Weights & Biases", False)
        
        # Training button
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training model..."):
                # Create model
                model = create_model(
                    model_type=model_type,
                    pooling_method=pooling_method,
                    input_dim=dataset.num_node_features,
                    num_classes=dataset.num_classes
                )
                
                # Create data loaders with selected batch size
                train_loader_new, val_loader_new, test_loader_new = create_data_loaders(
                    dataset,
                    batch_size=batch_size,
                    num_workers=0,
                    pin_memory=False
                )
                
                # Simple training loop (simplified for demo)
                device = get_device("cpu")  # Use CPU for demo
                model = model.to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                criterion = torch.nn.CrossEntropyLoss()
                
                # Training metrics
                train_losses = []
                val_losses = []
                train_accuracies = []
                val_accuracies = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for epoch in range(max_epochs):
                    # Training
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    train_total = 0
                    
                    for batch in train_loader_new:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        
                        logits = model(batch.x, batch.edge_index, batch.batch)
                        loss = criterion(logits, batch.y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_correct += (torch.argmax(logits, dim=1) == batch.y).sum().item()
                        train_total += batch.y.size(0)
                    
                    train_loss /= len(train_loader_new)
                    train_acc = train_correct / train_total
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for batch in val_loader_new:
                            batch = batch.to(device)
                            logits = model(batch.x, batch.edge_index, batch.batch)
                            loss = criterion(logits, batch.y)
                            
                            val_loss += loss.item()
                            val_correct += (torch.argmax(logits, dim=1) == batch.y).sum().item()
                            val_total += batch.y.size(0)
                    
                    val_loss /= len(val_loader_new)
                    val_acc = val_correct / val_total
                    
                    # Store metrics
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    train_accuracies.append(train_acc)
                    val_accuracies.append(val_acc)
                    
                    # Update progress
                    progress = (epoch + 1) / max_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch+1}/{max_epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Early stopping check
                    if epoch > patience:
                        if all(val_accuracies[-i] <= val_accuracies[-patience-1] for i in range(1, patience+1)):
                            st.info(f"Early stopping at epoch {epoch+1}")
                            break
                
                # Store results in session state
                st.session_state.training_results = {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_accuracies": train_accuracies,
                    "val_accuracies": val_accuracies,
                    "model_type": model_type,
                    "pooling_method": pooling_method,
                    "final_train_acc": train_acc,
                    "final_val_acc": val_acc
                }
                
                st.success("Training completed!")
    
    with tab4:
        st.header("Training Results")
        
        if "training_results" in st.session_state:
            results = st.session_state.training_results
            
            # Results summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Final Train Accuracy", f"{results['final_train_acc']:.4f}")
            
            with col2:
                st.metric("Final Val Accuracy", f"{results['final_val_acc']:.4f}")
            
            with col3:
                st.metric("Model Type", results['model_type'])
            
            with col4:
                st.metric("Pooling Method", results['pooling_method'])
            
            # Training curves
            st.subheader("Training Curves")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Loss", "Accuracy"),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            epochs = range(1, len(results['train_losses']) + 1)
            
            # Loss curves
            fig.add_trace(
                go.Scatter(x=epochs, y=results['train_losses'], name="Train Loss", line=dict(color="blue")),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=results['val_losses'], name="Val Loss", line=dict(color="red")),
                row=1, col=1
            )
            
            # Accuracy curves
            fig.add_trace(
                go.Scatter(x=epochs, y=results['train_accuracies'], name="Train Acc", line=dict(color="blue")),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=results['val_accuracies'], name="Val Acc", line=dict(color="red")),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            fig.update_xaxes(title_text="Epoch")
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("No training results available. Please train a model first.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Graph Pooling Methods Demo** - Interactive exploration of different graph pooling techniques")


if __name__ == "__main__":
    main()
