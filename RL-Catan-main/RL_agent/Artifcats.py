import pandas as pd
import numpy as np
import mlflow
import torch
import os

def PushArtifacts(experiment_name, params,Model,Results,MLFLOW_ADDRESS,TrainingData=False,TestingData = False,TagID = 0):
    if (TrainingData == False and TestingData == False) or (TrainingData == True and TestingData == True):
        print("Error: Either TrainingData or TestingData must be True, but not both.")
        return 0
    os.makedirs("Artifacts", exist_ok=True)
    DF = pd.DataFrame(Results)
    DF.to_csv('Artifacts/Results.csv')
    torch.save(Model.state_dict(),'Artifacts/Model_Parameters.pth')
    mlflow.set_tracking_uri(uri=MLFLOW_ADDRESS)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_artifact('Results.csv')
        mlflow.log_artifact('Model_Parameters.pth')
        if TrainingData is True:
            mlflow.log_metric('average moves', np.mean(DF.moves))
            mlflow.log_metric('average reward per move', np.mean(DF.RewardsPerTurn))
            mlflow.log_metric('average loss', np.mean(DF.AverageQloss))
            mlflow.set_tag("Training Run", str(TagID))
        if TestingData is True:
            mlflow.log_metric('Win rate',DF.winrate)
            mlflow.log_metric('RL Avg Victroy Points ', DF.VP0)
            mlflow.log_metric('Random Avg Victroy Points ', DF.VP1)
            mlflow.set_tag("Testing Run", str(TagID))
