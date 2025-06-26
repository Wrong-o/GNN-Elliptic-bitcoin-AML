from azureml.core import Workspace
from azureml.core import Dataset

# Connect to your existing workspace
ws = Workspace.from_config()  # Make sure you have config.json in current directory

# Get the default datastore
datastore = ws.get_default_datastore()

# Upload the data to Azure storage
Dataset.File.upload_directory(
    src_dir='./data/raw',  # Local folder containing the Elliptic dataset
    target=(datastore, 'elliptic-dataset'),  # Path in Azure storage
    overwrite=True,
    show_progress=True
)

print(f"Data uploaded to {datastore.name}")

# Register as a dataset for easier access in ML pipelines

# Create a file dataset
elliptic_dataset = Dataset.File.from_files(
    path=(datastore, 'elliptic-dataset')
)

# Register the dataset
registered_dataset = elliptic_dataset.register(
    workspace=ws,
    name='elliptic_bitcoin_dataset',
    description='Elliptic Bitcoin Transaction Dataset',
    create_new_version=True
)

print(f"Dataset registered as {registered_dataset.name} (version {registered_dataset.version})")
