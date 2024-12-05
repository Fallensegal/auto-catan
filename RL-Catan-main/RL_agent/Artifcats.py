import pandas as pd
import numpy as np
import mlflow
import torch
import os

def PushArtifacts(experiment_name:str, params:dict,Model,Results:List[Dict],MLFLOW_ADDRESS:str,TrainingData:bool=False,TestingData:bool = False,TagID:int = 0) -> bool:
    if (TrainingData == False and TestingData == False) or (TrainingData == True and TestingData == True):
        print("Error: Either TrainingData or TestingData must be True, but not both.")
        return False
    os.makedirs("Artifacts", exist_ok=True)
    DF = pd.DataFrame(Results)
    DF.to_csv('Artifacts/Results.csv')
    torch.save(Model.state_dict(),'Artifacts/Model_Parameters.pth')
    if MLFLOW_ADDRESS is None: 
        print('not pushing to ML Flow. Returned artifacts in')
        return True
    else:
        mlflow.set_tracking_uri(uri=MLFLOW_ADDRESS)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_artifact('Results.csv')
            mlflow.log_artifact('Model_Parameters.pth')
            if TrainingData is True:
                mlflow.log_metric('average moves', np.mean(DF.game_length))
                mlflow.log_metric('average reward per move', np.mean(DF.Reward_over_episode))
                mlflow.log_metric('average loss', np.mean(DF.AverageQloss))
                mlflow.set_tag("Training Run", str(TagID))
            if TestingData is True:
                mlflow.log_metric('Win rate',np.sum(DF.player_0_win)/len(DF.player_0_win))
                mlflow.log_metric('RL Avg Victroy Points ', np.mean(DF.player_0_victory_points))
                mlflow.log_metric('Random Avg Victroy Points ', np.mean(DF.player_1_victory_points))
                mlflow.set_tag("Testing Run", str(TagID))
        return True