#import json
import dramatiq
#from catan_dramatiq.storage import s3storage

@dramatiq.actor()
def create_s3_bucket(bucket_name: str) -> str:
    return f"testing dramatiq: {bucket_name}"