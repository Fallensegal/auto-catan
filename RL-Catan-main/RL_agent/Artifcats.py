import pandas as pd
import numpy as np
import mlflow

def PushArtifacts(experiment_name, params,Model,Results,MLFLOW_ADDRESS,TrainingData=False,TestingData = False):
    if (TrainingData == False and TestingData == False) or (TrainingData == True and TestingData == True):
        return 0
    DF = pd.DataFrame(Results)
    DF.to_csv('Results.csv')
    mlflow.set_tracking_uri(uri=MLFLOW_ADDRESS)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        if TrainingData is True:
            mlflow.log_artifact('Results.csv')
            mlflow.log_metric('average moves', np.mean(DF.moves))
            mlflow.log_metric('average reward per move', np.mean(DF.RewardsPerTurn))
            mlflow.log_metric('average loss', np.mean(DF.AverageQloss))
            mlflow.set_tag("Training Run", "Test that ML Flow is working...")
        if TestingData is True:
            mlflow.log_artifact('Results.csv')
            mlflow.log_metric('Win rate',DF.winrate)
            mlflow.log_metric('RL Avg Victroy Points ', DF.VP0)
            mlflow.log_metric('Random Avg Victroy Points ', DF.VP1)
            mlflow.set_tag("Training Run", "Test that ML Flow is working...")
