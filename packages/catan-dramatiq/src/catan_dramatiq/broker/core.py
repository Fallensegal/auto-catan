import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from redis import Redis

class AppSettings(BaseSettings):
    """Import all environment variables passed to pod and initialize the password for redis"""
    redis_password: SecretStr = SecretStr('default')


##################### Execution #######################

app_settings = AppSettings()

redis_client = Redis(
    host="chart-redis-master.catan.svc.cluster.local",
    password=app_settings.redis_password.get_secret_value(),
    port=6379,
    decode_responses=False,
    socket_connect_timeout=0.5,
)

results_backend = Results(backend=RedisBackend(client=redis_client), store_results=True)

redis_broker = RedisBroker(client=redis_client)
redis_broker.add_middleware(results_backend)
dramatiq.set_broker(broker=redis_broker)
