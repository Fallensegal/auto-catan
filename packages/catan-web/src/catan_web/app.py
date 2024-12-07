from fastapi import FastAPI
from fastapi import APIRouter
from catan_dramatiq.tasks import tasks
from catan_dramatiq.broker import redis_broker

# Define App
app = FastAPI()
s3_router = APIRouter(prefix="/s3", tags=["s3"])

# Server Routes
@app.get("/healthz")
async def health_check() -> dict[str, str]:
    """Check status of Web-Server"""
    return {"status": "ok"}

@app.post("/greet")
async def greet_user(names: list[str]) -> dict[str,str]:
    """Greet the user when given a list of users"""
    output_messege: dict[str, str] = {}
    for name in names:
        output_messege[name] = f"Hello User: {name}"
    
    return output_messege

@app.get("/flush")
def flush_broker() -> bool:
    try:
        redis_broker.flush_all()
        return True
    except Exception:
        return False

# Storage Routes

@s3_router.post("/creat_bucket")
def create_s3_bucket(s3_name: str) -> dict[str, str]:
    msg = tasks.create_s3_bucket.send(bucket_name=s3_name)
    return {'Server:': msg.get_result(timeout=100_000, block=False)}

app.include_router(s3_router)