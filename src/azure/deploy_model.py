from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

def main():
    # Connect to workspace
    ws = Workspace.from_config()
    
    # Get the model
    model = Model(ws, name="elliptic_gnn")
    
    # Define inference configuration
    inference_config = InferenceConfig(
        entry_script="src/azure/score.py",
        source_directory=".",
        conda_file="src/azure/conda_env.yml"
    )
    
    # Define deployment configuration
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        auth_enabled=True
    )
    
    # Deploy the model
    service = Model.deploy(
        workspace=ws,
        name='elliptic-gnn-service',
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config
    )
    
    service.wait_for_deployment(show_output=True)
    print(f"Service deployed: {service.scoring_uri}")

if __name__ == '__main__':
    main()
