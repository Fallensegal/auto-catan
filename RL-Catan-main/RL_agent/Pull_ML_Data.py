import mlflow
import os
import pandas

def Pull_MLFLOW_Data_to_Folder(experiment_name):
    # Set experiment ID or name
    mlflow.set_tracking_uri(uri='http://192.168.161.128:8250')
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Ensure experiment exists
    if experiment is None:
        print(f"Experiment '{experiment_name}' does not exist.")
        exit()

    experiment_id = experiment.experiment_id

    # Get all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    outputfile = 'summary.csv'

    # Local directory to store artifacts
    output_dir = experiment_name
    os.makedirs(output_dir, exist_ok=True)
    Results_file = os.path.join(output_dir,outputfile)
    runs.to_csv(Results_file,index=False)
    # Loop through all runs
    for _, row in runs.iterrows():
        run_id = row["run_id"]
        
        # Fetch the run to get tags
        run = mlflow.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", run_id)  # Fallback to run_id if no name is set
        
        # Sanitize run name for file system compatibility
        sanitized_run_name = "".join(c if c.isalnum() or c in (' ', '.', '_') else "_" for c in run_name)
        
        # Create a directory using the run name
        run_artifacts_dir = os.path.join(output_dir, sanitized_run_name)
        os.makedirs(run_artifacts_dir, exist_ok=True)
        
        # Download artifacts for this run
        print(f"Downloading artifacts for run: {run_name} (ID: {run_id})")
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="",
            dst_path=run_artifacts_dir
        )

    print(f"All artifacts downloaded to '{output_dir}'.")
