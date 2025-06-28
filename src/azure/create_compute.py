from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Connect to workspace
ws = Workspace.from_config()

# Compute target name
compute_name = "helloCompute"

# Check if compute target already exists
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    
    # Define compute configuration
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='Standard_NC6',  # GPU VM
        min_nodes=0,
        max_nodes=2,
        idle_seconds_before_scaledown=1800
    )
    
    # Create the compute target
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

print(f"Compute target '{compute_name}' is ready!")
