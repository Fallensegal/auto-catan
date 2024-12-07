#import json
import dramatiq
from catan_dramatiq.broker.core import redis_broker, results_backend

redis_broker.add_middleware(results_backend)
dramatiq.set_broker(broker=redis_broker)

@dramatiq.actor(broker=redis_broker)
def create_s3_bucket(bucket_name: str) -> str:
    return f"testing dramatiq: {bucket_name}"