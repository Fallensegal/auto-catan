import dramatiq
import torch
from typing import Any
from catan_dramatiq.broker.core import redis_broker, results_backend
from catan_dramatiq.catan_env.game import Game
from catan_dramatiq.storage import mlflow

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
    torch.manual_seed(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    return "Successful BenchMark"

