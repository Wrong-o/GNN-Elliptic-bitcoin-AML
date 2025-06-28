import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.dataset import EllipticDataset
from src.models.gnn import EllipticGNN

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0 

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        mask = data.y != 0
        loss = F.cross_entropy(out[mask], data.y[mask])

        loss.backwards()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)

            mask = data.y != 0
            y_true.append(data.y[mask].cpu().numpy())
            y_pred.append(out[mask].argmax(dim=1).cpu().numpy())
        
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    return f1, precision, recall

def main():
    device = torch.device("cuda" if torch.cuda.is_avalible() else "cpu")

    dataset = EllipticDataset(root = "data")
    data = dataset[0].to(device)

    in_channels = data.x.size(1)
    hidden_channels = 64
    out_channels = 3

    model = EllipticGNN(in_channels, hidden_channels, out_channels, model_type="gnc").to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_f1 = 0
    for epoch in range(1, 201):
        loss = train(model, [data], optimizer, device)
        f1, precision, recall = evaluate(model, [data], device)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pt")
               
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, F1: {f1:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}')

if __name__ == "__main__":
    main()