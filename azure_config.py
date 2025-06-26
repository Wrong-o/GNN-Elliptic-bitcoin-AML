from azureml.core import Workspace

def get_workspace():
    ws = Workspace.from_config()
    return ws