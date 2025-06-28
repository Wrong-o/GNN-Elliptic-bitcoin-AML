import os
import json
import torch
import numpy as np
from torch_geometric.data import Data
from models.gnn import EllipticGNN

def init():
    global model
    
    # Get model path
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_model.pt')
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EllipticGNN(166, 64, 3, model_type='gcn').to(device)  # Adjust input features as needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Extract features and edges
        features = torch.tensor(data['features'], dtype=torch.float)
        edges = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
        
        # Create PyG data object
        pyg_data = Data(x=features, edge_index=edges)
        
        # Make prediction
        with torch.no_grad():
            output = model(pyg_data.x, pyg_data.edge_index)
            probabilities = torch.nn.functional.softmax(output, dim=1).numpy()
            predictions = output.argmax(dim=1).numpy()
        
        # Return results
        return json.dumps({
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        return json.dumps({"error": str(e)})
