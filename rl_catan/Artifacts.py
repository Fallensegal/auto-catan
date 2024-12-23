import pandas as pd
import numpy as np
import mlflow
import torch
import os

def PushArtifacts(experiment_name:str, params:dict,Model,Results,MLFLOW_ADDRESS:str,RunName:str, TrainingData:bool=False,TestingData:bool = False) -> bool:
    if (TrainingData == False and TestingData == False) or (TrainingData == True and TestingData == True):
        print("Error: Either TrainingData or TestingData must be True, but not both.")
        return False
    os.makedirs("Artifacts", exist_ok=True)
    DF = pd.DataFrame(Results)
    DF.to_csv('Artifacts/Results.csv')
    torch.save(Model.state_dict(),'Artifacts/Model_Parameters.pth')
    if MLFLOW_ADDRESS is None: 
        print('not pushing to ML Flow. Returned artifacts in Artifacts directory')
        return True
    else:
        mlflow.set_tracking_uri(uri=MLFLOW_ADDRESS)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=RunName):
            mlflow.log_params(params)
            mlflow.log_artifact('Artifacts/Results.csv')
            mlflow.log_artifact('Artifacts/Model_Parameters.pth')
            if TrainingData is True:
                mlflow.log_metric('average moves', np.mean(DF.game_length))
                mlflow.log_metric('average reward per move', np.mean(DF.Reward_over_episode))
                mlflow.log_metric('average loss', np.mean(DF.average_model_loss))
                mlflow.set_tag("Training Run", str('Training'))
            if TestingData is True:
                mlflow.log_metric('Win rate',np.sum(DF.player_0_win)/len(DF.player_0_win))
                mlflow.log_metric('RL Avg Victroy Points ', np.mean(DF.player_0_victory_points))
                mlflow.log_metric('Random Avg Victroy Points ', np.mean(DF.player_1_victory_points))
                mlflow.set_tag("Testing Run", str('Testing'))
        return True