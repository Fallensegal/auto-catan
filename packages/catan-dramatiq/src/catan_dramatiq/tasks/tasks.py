import dramatiq
import torch
import time
from typing import Any
from catan_dramatiq.broker.core import redis_broker, results_backend
from catan_dramatiq.storage import mlflow
from catan_dramatiq.orm.config import TrainingInput
from catan_dramatiq.catan_exec.exec import Catan_Training
from catan_dramatiq.catan_model.DQN import DQN_MEDIUM, DQN_POOL, DQN_SMALL

redis_broker.add_middleware(results_backend)
dramatiq.set_broker(broker=redis_broker)

# S3 Bucket Operations

@dramatiq.actor(broker=redis_broker)
def create_s3_bucket(bucket_name: str) -> str:
    return f"testing dramatiq: {bucket_name}"

# ML Operations

@dramatiq.actor(broker=redis_broker)
def receive_pydantic_model(config: dict[str, Any]) -> str:
    return config['MLFLOW_ADDRESS']

@dramatiq.actor(broker=redis_broker)
def execute_rl_benchmark(config: dict[str, Any]) -> str:

    # Setup Device and Config
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    param_config = TrainingInput.model_validate(config)
    model: Any | None = None

    # Config Based Model Import
    if param_config.MODEL_SELECT == 'Small_Pooling':
        model = DQN_POOL()
    elif param_config.MODEL_SELECT == 'Medium':
        model = DQN_MEDIUM()
    else:
        model = DQN_SMALL()
    
    # Move model to device
    model.to(device)

    # Define Training
    training = Catan_Training(
        EPISODES_PER_LOOP=param_config.EPISODES_PER_LOOP,
        GAMES_PER_BENCHMARK=param_config.GAMES_PER_BENCHMARK,
        reward_function=param_config.REWARD_FUNCTION,
        device=device,
        model=model,
        stochastic_policy=param_config.STOCHASTIC,
        memory=param_config.MEMORY,
        LR_START=param_config.LR_START,
        LR_END=param_config.LR_END,
        LR_DECAY=param_config.LR_DECAY,
        EPS_START=param_config.EPS_START,
        EPS_END=param_config.EPS_END,
        EPS_DECAY=param_config.EPS_DECAY,
        GAMMA=param_config.GAMMA,
        BATCH_SIZE=param_config.BATCH_SIZE
    )

    start_time = time.time()

    # Training Loop
    for _ in range(param_config.TRAINING_LOOPS):
        training.train()
        if param_config.STOCHASTIC:
            exp_name = f"{param_config.REWARD_FUNCTION}_{param_config.MODEL_SELECT}_Stochastic"
        else:
            exp_name = f"{param_config.REWARD_FUNCTION}_{param_config.MODEL_SELECT}_Deterministic"

        mlflow.PushArtifacts(experiment_name=exp_name,
                             params=config,
                             Model=training.agent_policy_net,
                             Results=training.EpisodeData,
                             MLFLOW_ADDRESS=param_config.MLFLOW_ADDRESS,
                             RunName=None,
                             TrainingData=True,
                             TestingData=False)
        
        # Set Minimum win Rate
        training.min_win_rate = 0.01
        training_broke = training.benchmark()

        mlflow.PushArtifacts(
            experiment_name=exp_name,
            params=config,
            Model=model,
            Results=training.EpisodeData,
            MLFLOW_ADDRESS=param_config.MLFLOW_ADDRESS,
            RunName=None,
            TrainingData=False,
            TestingData=True
        )

        if training_broke:
            print("Training Stopped: Low Win Rate")
            break

    elapsed_time = time.time() - start_time
    print('Complete')
    print(f'steps over {training.total_episodes} episodes: {training.steps_done}')
    print(f'Elapsed time: {elapsed_time}')
    print(f'Optimizer steps: {len(training.game.average_q_value_loss)}')

    return f"Device: {device}, URI: {param_config.MLFLOW_ADDRESS}, TRAIN_SUCCESS: {str(True)}, ELAPSED_TIME: {elapsed_time}"

