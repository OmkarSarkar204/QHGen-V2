from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute, Environment
from azure.identity import DefaultAzureCredential
import logging

# --- 1. CONFIGURATION (FILL THESE IN) ---
# Go to your Azure Portal -> Machine Learning Workspace -> Overview to find these.
SUBSCRIPTION_ID = "42dcbcc8-8e39-4b60-8911-2d971b41f4f5"
RESOURCE_GROUP = "qhgen-rg"
WORKSPACE_NAME = "qhgen-quantum"
COMPUTE_NAME = "cpu-cluster"

def main():
    print(f"🔌 Connecting to Azure ML Workspace: {WORKSPACE_NAME}...")
    
    # Connect using your local Azure credentials
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

    # --- 2. ENSURE COMPUTE CLUSTER EXISTS ---
    # This checks if a computer exists in the cloud. If not, it creates one.
    try:
        print(f"Checking for compute target: {COMPUTE_NAME}...")
        ml_client.compute.get(COMPUTE_NAME)
        print("   Found existing compute target.")
    except Exception:
        print("   Compute target not found. Creating new one (this takes ~3 mins)...")
        # Create a basic CPU cluster (Standard_DS11_v2 is cheap and sufficient)
        cpu_cluster = AmlCompute(
            name=COMPUTE_NAME,
            type="amlcompute",
            size="Standard_DS11_v2",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(cpu_cluster).result()
        print("   Compute cluster created!")

    # --- 3. SUBMIT THE JOB ---
    print("🚀 Submitting Training Job to Azure Cloud...")
    
    # This command bundles your folder, uploads it, installs libs from conda.yaml, 
    # and runs 'models/surrogate/train.py' in the cloud.
    job = command(
        code="./",  # Upload current directory
        command="python models/surrogate/schnet_engine.py", # Running your engine script
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file="conda.yaml"
        ),
        compute=COMPUTE_NAME,
        display_name="QHGen-Alloy-Training",
        experiment_name="qhgen-discovery-v1"
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"{returned_job.studio_url}")

if __name__ == "__main__":
    main()