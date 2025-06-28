import os
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from azureml.core import Run

# Import our model (assuming it's copied to the compute target)
from models.gnn import EllipticGNN

def load_data(data_path):
    """Load and process the Elliptic dataset"""
    features_path = os.path.join(data_path, 'elliptic_txs_features.csv')
    edges_path = os.path.join(data_path, 'elliptic_txs_edgelist.csv')
    classes_path = os.path.join(data_path, 'elliptic_txs_classes.csv')
    
    # Read data
    features_df = pd.read_csv(features_path, header=None)
    edges_df = pd.read_csv(edges_path)
    classes_df = pd.read_csv(classes_path)
    
    # Process node features
    node_features = features_df.iloc[:, 2:].values
    node_features = torch.FloatTensor(node_features)
    
    # Create mapping from txId to index
    txid_to_idx = {txid: i for i, txid in enumerate(features_df.iloc[:, 0].values)}
    
    # Process edges
    edge_index = []
    for _, row in edges_df.iterrows():
        src, dst = row['txId1'], row['txId2']
        if src in txid_to_idx and dst in txid_to_idx:
            edge_index.append([txid_to_idx[src], txid_to_idx[dst]])
    
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    # Process labels
    labels_dict = {'unknown': 0, '1': 1, '2': 2}
    labels = []
    for txid in features_df.iloc[:, 0].values:
        label_row = classes_df[classes_df['txId'] == txid]
        if len(label_row) > 0:
            label = labels_dict[label_row['class'].values[0]]
        else:
            label = 0  # unknown
        labels.append(label)
    
    y = torch.tensor(labels, dtype=torch.long)
    
    # Create PyG data object
    data = Data(x=node_features, edge_index=edge_index, y=y)
    return data

def train(model, data, optimizer, device):
    model.train()
    
    # Forward pass
    data = data.to(device)
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    
    # Only use known labels for training (1: illicit, 2: licit)
    mask = data.y != 0
    loss = F.cross_entropy(out[mask], data.y[mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, data, device):
    model.eval()
    
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index)
        
        # Only evaluate on known labels
        mask = data.y != 0
        y_true = data.y[mask].cpu().numpy()
        y_pred = out[mask].argmax(dim=1).cpu().numpy()
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    return f1, precision, recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='Path to the data')
    parser.add_argument('--model-type', type=str, default='gcn', choices=['gcn', 'gat', 'sage'])
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    args = parser.parse_args()
    
    # Get the run context
    run = Run.get_context()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_data(args.data_path)
    print(f"Dataset loaded: {data}")
    
    # Model parameters
    in_channels = data.x.size(1)  # Number of features
    out_channels = 3  # unknown, illicit, licit
    
    # Initialize model
    model = EllipticGNN(
        in_channels, 
        args.hidden_channels, 
        out_channels, 
        model_type=args.model_type
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=5e-4
    )
    
    # Training loop
    best_f1 = 0
    losses = []
    f1_scores = []
    
    for epoch in range(1, args.epochs + 1):
        loss = train(model, data, optimizer, device)
        f1, precision, recall = evaluate(model, data, device)
        
        # Log metrics to Azure ML
        run.log('loss', loss)
        run.log('f1_score', f1)
        run.log('precision', precision)
        run.log('recall', recall)
        
        losses.append(loss)
        f1_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            # Save model
            os.makedirs('./outputs', exist_ok=True)
            torch.save(model.state_dict(), './outputs/best_model.pt')
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, F1: {f1:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}')
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(f1_scores)
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    
    # Save the figure
    plt.savefig('./outputs/training_curves.png')
    
    # Upload the figure to Azure ML
    run.log_image('Training Curves', plot=plt)
    
    # Register the model
    run.upload_file('best_model.pt', './outputs/best_model.pt')
    run.register_model(
        model_name='elliptic_gnn',
        model_path='best_model.pt',
        description=f'GNN model for Elliptic Bitcoin dataset ({args.model_type})'
    )
    
    # Complete the run
    run.complete()

if __name__ == '__main__':
    main()
