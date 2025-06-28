from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies

def main():
    # Connect to workspace
    print("main started!")
    ws = Workspace.from_config()
    
    # Get compute target
    compute_target = ComputeTarget(workspace=ws, name="helloCompute")
    print("Compute and workspace worked")
    print("Create python enviroment")
    
    # Create a Python environment for the experiment
    pytorch_env = Environment("pytorch-env")
    pytorch_env.python.conda_dependencies = CondaDependencies.create(
        conda_packages=['pytorch', 'pip'],
        pip_packages=[
            'torch-geometric',
            'torch-scatter',
            'torch-sparse',
            'pandas',
            'numpy',
            'scikit-learn',
            'matplotlib'
        ]
    )
    
    print("Configuring the run")
    # Configure the training run
    src_dir = './src'
    script_config = ScriptRunConfig(
        source_directory=src_dir,
        script='training/train_azure.py',
        compute_target=compute_target,
        environment=pytorch_env,
        arguments=[
            '--data-path', ws.datasets.get('elliptic_bitcoin_dataset').as_mount(),
            '--model-type', 'gcn',
            '--hidden-channels', '64',
            '--epochs', '200',
            '--learning-rate', '0.01'
        ]
    )
    
    # Submit experiment
    print("Submitting experiment")
    experiment = Experiment(workspace=ws, name='elliptic-gnn-training')
    print("Running experiment")
    run = experiment.submit(script_config)
    
    # Print run details
    print(f"Run ID: {run.id}")
    print(f"Run details page: {run.get_portal_url()}")
    
    # Wait for the run to complete
    run.wait_for_completion(show_output=True)

if __name__ == '__main__':
    main()
