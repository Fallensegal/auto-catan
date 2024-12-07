#import json
import dramatiq
from catan_dramatiq.broker import redis_broker

@dramatiq.actor(broker=redis_broker)
def create_s3_bucket(bucket_name: str) -> str:
    return f"testing dramatiq: {bucket_name}"