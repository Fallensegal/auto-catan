#import json
import dramatiq
from catan_dramatiq.broker.core import redis_broker, results_backend
from catan_dramatiq.orm.config import TrainingInput

redis_broker.add_middleware(results_backend)
dramatiq.set_broker(broker=redis_broker)

# S3 Bucket Operations

@dramatiq.actor(broker=redis_broker)
def create_s3_bucket(bucket_name: str) -> str:
    return f"testing dramatiq: {bucket_name}"

# ML Operations

@dramatiq.actor(broker=redis_broker)
def receive_pydantic_model(config: TrainingInput) -> str:
    config_dict = config.model_dump()
    return config_dict['MLFLOW_ADDRESS']